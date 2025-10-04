from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F


class RunningStats:
    """Track mean and variance for high-dimensional tensors using Welford grouping."""

    def __init__(self, size: int, dtype: torch.dtype = torch.float32, device: Union[torch.device, str] = "cpu"):
        self.count = 0
        self.dtype = dtype
        self.device = device
        self.mean = torch.zeros(size, dtype=dtype, device=device)
        self.M2 = torch.zeros(size, dtype=dtype, device=device)

    def update(self, batch: torch.Tensor):
        if batch is None or batch.numel() == 0:
            return
        batch = batch.to(self.dtype)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_delta = batch - batch_mean.unsqueeze(0)
        batch_M2 = (batch_delta * batch_delta).sum(dim=0)

        if self.count == 0:
            self.mean.copy_(batch_mean)
            self.M2.copy_(batch_M2)
            self.count = batch_count
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.M2 += batch_M2 + (delta * delta) * self.count * batch_count / total
        self.count = total

    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.zeros_like(self.mean)
        return self.M2 / (self.count - 1)


def _compute_top_token_ids(dataloader, tokenizer, max_classes=512):
    if max_classes is None or max_classes <= 0:
        return []

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return []

    counts = torch.zeros(vocab_size, dtype=torch.long)
    for batch in dataloader:
        input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        ids = input_ids.reshape(-1).cpu()
        valid = (ids >= 0) & (ids < vocab_size)
        if not valid.any():
            continue
        ids = ids[valid]
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))

    nonzero = (counts > 0).nonzero(as_tuple=False).squeeze(-1)
    if nonzero.numel() == 0:
        return []

    k = min(max_classes, nonzero.numel())
    return torch.topk(counts, k).indices.tolist()


def collect_neuronrank_statistics(model, dataloader, tokenizer, device, max_classes=512):
    """Capture post-activation statistics for each LLaMA MLP gate projection."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected the model to expose model.layers for NeuronRank scoring.")

    batches = list(dataloader)
    if not batches:
        raise ValueError("Calibration dataloader for NeuronRank produced no batches.")

    top_token_ids = _compute_top_token_ids(batches, tokenizer, max_classes=max_classes)
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    class_index_lookup = torch.full((vocab_size,), -1, dtype=torch.long) if top_token_ids else None
    if class_index_lookup is not None:
        for idx, tok in enumerate(top_token_ids):
            if 0 <= tok < vocab_size:
                class_index_lookup[tok] = idx

    num_classes = len(top_token_ids)
    print(f"loading calibration data ({len(batches)} batches, tracking {num_classes} token classes)")

    layer_stats = {}
    hooks = []
    current_batch_token_ids = None
    class_lookup_cache = {}

    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue

        layer_device = gate_proj.weight.device
        layer_dtype = torch.float32

        layer_stats[layer_idx] = {
            "sample": RunningStats(gate_proj.out_features, dtype=layer_dtype, device=layer_device),
            "token": RunningStats(gate_proj.out_features, dtype=layer_dtype, device=layer_device),
            "class_sum": torch.zeros((num_classes, gate_proj.out_features), dtype=layer_dtype, device=layer_device) if num_classes else None,
            "class_count": torch.zeros(num_classes, dtype=layer_dtype, device=layer_device) if num_classes else None,
        }

        def make_hook(idx):
            def hook(_module, _inputs, output):
                if output is None:
                    return
                act = F.silu(output)
                if act.dim() == 3:
                    per_sample = act.mean(dim=1)
                    per_token = act.flatten(0, 1)
                else:
                    per_sample = act
                    per_token = act

                per_sample_local = per_sample.detach().to(dtype=layer_dtype)
                per_token_local = per_token.detach().to(dtype=layer_dtype)

                layer_stats[idx]["sample"].update(per_sample_local)
                layer_stats[idx]["token"].update(per_token_local)

                if class_index_lookup is not None:
                    flat_ids = current_batch_token_ids
                    if flat_ids is None or flat_ids.numel() != per_token_local.shape[0]:
                        return
                    lookup = class_lookup_cache.get(per_sample_local.device)
                    if lookup is None:
                        lookup = class_index_lookup.to(per_sample_local.device)
                        class_lookup_cache[per_sample_local.device] = lookup
                    within_vocab = flat_ids < lookup.shape[0]
                    if not within_vocab.any():
                        return
                    masked_ids = flat_ids[within_vocab]
                    class_indices = lookup[masked_ids]
                    valid_mask = class_indices != -1
                    if not valid_mask.any():
                        return
                    idxs = class_indices[valid_mask]
                    token_values = per_token_local[within_vocab][valid_mask]
                    layer_stats[idx]["class_sum"].index_add_(0, idxs, token_values)
                    layer_stats[idx]["class_count"].index_add_(0, idxs, torch.ones(idxs.size(0), dtype=layer_dtype, device=token_values.device))

            return hook

        hooks.append(gate_proj.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            current_batch_token_ids = input_ids.reshape(-1)
            model(input_ids=input_ids, attention_mask=attention_mask)
            current_batch_token_ids = None

    for handle in hooks:
        handle.remove()

    stats = {}
    for idx, stat in layer_stats.items():
        sample_var = stat["sample"].variance().to(dtype=torch.float32, device="cpu")
        token_var = stat["token"].variance().to(dtype=torch.float32, device="cpu")
        class_var = None
        if stat.get("class_sum") is not None:
            counts = stat["class_count"]
            valid = counts > 0
            if valid.any():
                class_means = (stat["class_sum"][valid] / counts[valid].unsqueeze(1)).to(dtype=torch.float32, device="cpu")
                if class_means.size(0) > 1:
                    class_var = class_means.var(dim=0, unbiased=False)
                else:
                    class_var = torch.zeros_like(sample_var)
        stats[idx] = {
            "sample_variance": sample_var,
            "token_variance": token_var,
            "class_variance": class_var,
        }
    return stats


def _safe_scalar_pow(base: float, exponent: float) -> float:
    if exponent == 0.0:
        return 1.0
    if base <= 0.0:
        return 0.0
    return float(base) ** float(exponent)


def compute_neuronrank_scores(
    model,
    stats,
    token_weight=0.0,
    variance_exp=1.0,
    variance_multi=1.0,
    magnitude_multi=0.0,
) -> Dict[int, Dict[str, Optional[torch.Tensor]]]:
    scores: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue
        weight = gate_proj.weight.detach().to(dtype=torch.float32, device="cpu")
        row_norm = torch.norm(weight, p=2, dim=1)

        layer_stats = stats.get(layer_idx)
        if layer_stats is None:
            variance = torch.zeros_like(row_norm)
        else:
            variance = layer_stats["sample_variance"].to(row_norm.device)
            if token_weight > 0:
                variance = variance + token_weight * layer_stats["token_variance"].to(row_norm.device)
            class_variance = layer_stats.get("class_variance")
            if class_variance is not None:
                variance = variance + class_variance.to(row_norm.device)

        variance = variance.clamp(min=0.0)
        
        if variance_exp == 0.0:
            variance_term = torch.ones_like(variance)
        elif variance_exp == 1.0:
            variance_term = variance
        else:
            variance_term = torch.pow(variance.clamp(min=1e-12), variance_exp)
        
        variance_component = variance_term * variance_multi
        magnitude_component = row_norm * magnitude_multi
        channel_score = variance_component + magnitude_component

        scores[layer_idx] = {
            "channel": channel_score,
            "variance": variance if variance is not None else None,
        }
    return scores


def compute_neuronrank_class_scores(stats, min_value: float = 0.0):
    """Derive per-neuron scores directly from class-wise variance."""

    scores = {}
    for layer_idx, layer_stats in stats.items():
        class_variance = layer_stats.get("class_variance")
        if class_variance is None:
            continue
        scores[layer_idx] = class_variance.clamp_min(min_value)
    return scores


def apply_neuronrank_pruning(model, scores, sparsity_ratio):
    total_channels = 0
    total_pruned = 0

    for layer_idx, layer in enumerate(model.model.layers):
        layer_entry = scores.get(layer_idx)
        if not layer_entry:
            continue

        layer_score = layer_entry.get("channel") if isinstance(layer_entry, dict) else layer_entry
        if layer_score is None:
            continue
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        up_proj = getattr(layer.mlp, "up_proj", None)
        down_proj = getattr(layer.mlp, "down_proj", None)
        if gate_proj is None:
            continue

        num_channels = layer_score.numel()
        total_channels += num_channels
        num_to_prune = int(num_channels * sparsity_ratio)
        if num_to_prune <= 0:
            continue

        prune_idx = torch.argsort(layer_score)[:num_to_prune]
        prune_idx_list = prune_idx.tolist()

        gate_proj.weight.data[prune_idx_list, :] = 0
        if gate_proj.bias is not None:
            gate_proj.bias.data[prune_idx_list] = 0

        if up_proj is not None:
            up_proj.weight.data[prune_idx_list, :] = 0
            if up_proj.bias is not None:
                up_proj.bias.data[prune_idx_list] = 0

        if down_proj is not None:
            down_proj.weight.data[:, prune_idx_list] = 0

        layer_pct = 100.0 * num_to_prune / num_channels
        print(f"[NeuronRank] layer {layer_idx}: pruned {num_to_prune}/{num_channels} channels ({layer_pct:.2f}%)")
        total_pruned += num_to_prune

    return total_pruned, total_channels
