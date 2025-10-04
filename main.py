import argparse
import os 
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.data import get_loaders
from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model


class RunningStats:
    """Track mean and variance for high-dimensional tensors using Welford grouping."""

    def __init__(self, size: int):
        self.count = 0
        self.mean = torch.zeros(size, dtype=torch.float64)
        self.M2 = torch.zeros(size, dtype=torch.float64)

    def update(self, batch: torch.Tensor):
        if batch is None or batch.numel() == 0:
            return
        batch = batch.to(torch.float64)
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
    if max_classes <= 0:
        return []

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return []

    counts = torch.zeros(vocab_size, dtype=torch.long)
    for batch in dataloader:
        input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        ids = input_ids.reshape(-1).cpu()
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.long))

    nonzero = (counts > 0).nonzero(as_tuple=False).squeeze(-1)
    if nonzero.numel() == 0:
        return []

    k = min(max_classes, nonzero.numel())
    topk = torch.topk(counts, k).indices.tolist()
    return topk


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
            if tok < vocab_size:
                class_index_lookup[tok] = idx

    num_classes = len(top_token_ids)
    print(f"loading calibration data ({len(batches)} batches, tracking {num_classes} token classes)")
    layer_stats = {}
    hooks = []
    current_batch_token_ids = None

    for layer_idx, layer in enumerate(model.model.layers):
        gate_proj = getattr(layer.mlp, "gate_proj", None)
        if gate_proj is None:
            continue

        layer_stats[layer_idx] = {
            "sample": RunningStats(gate_proj.out_features),
            "token": RunningStats(gate_proj.out_features),
            "class_sum": torch.zeros((num_classes, gate_proj.out_features), dtype=torch.float64) if num_classes else None,
            "class_count": torch.zeros(num_classes, dtype=torch.float64) if num_classes else None,
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

                per_sample_cpu = per_sample.detach().to(dtype=torch.float32, device="cpu")
                per_token_cpu = per_token.detach().to(dtype=torch.float32, device="cpu")

                layer_stats[idx]["sample"].update(per_sample_cpu)
                layer_stats[idx]["token"].update(per_token_cpu)

                if class_index_lookup is not None:
                    flat_ids = current_batch_token_ids
                    if flat_ids is None:
                        return
                    if flat_ids.numel() != per_token_cpu.shape[0]:
                        return
                    within_vocab = flat_ids < class_index_lookup.shape[0]
                    if not within_vocab.any():
                        return
                    masked_ids = flat_ids[within_vocab]
                    class_indices = class_index_lookup[masked_ids]
                    valid_mask = class_indices != -1
                    if not valid_mask.any():
                        return
                    idxs = class_indices[valid_mask]
                    token_values = per_token_cpu[within_vocab][valid_mask].to(dtype=torch.float64)
                    layer_stats[idx]["class_sum"].index_add_(0, idxs, token_values)
                    layer_stats[idx]["class_count"].index_add_(0, idxs, torch.ones(idxs.size(0), dtype=torch.float64))

            return hook

        hooks.append(gate_proj.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    print("loading calibration data")
    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            current_batch_token_ids = input_ids.reshape(-1).cpu()
            model(input_ids=input_ids, attention_mask=attention_mask)
            current_batch_token_ids = None

    for handle in hooks:
        handle.remove()

    stats = {}
    for idx, stat in layer_stats.items():
        sample_var = stat["sample"].variance().to(dtype=torch.float32)
        token_var = stat["token"].variance().to(dtype=torch.float32)
        class_var = None
        if stat.get("class_sum") is not None:
            counts = stat["class_count"]
            valid = counts > 0
            if valid.any():
                class_means = (stat["class_sum"][valid] / counts[valid].unsqueeze(1)).to(dtype=torch.float32)
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


def compute_neuronrank_scores(model, stats, token_weight=0.0, discrimination_weight=2.0):
    scores = {}
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

        variance = variance.clamp(min=0.0).pow(discrimination_weight)
        scores[layer_idx] = variance * row_norm
    return scores


def apply_neuronrank_pruning(model, scores, sparsity_ratio):
    total_channels = 0
    total_pruned = 0

    for layer_idx, layer in enumerate(model.model.layers):
        layer_score = scores.get(layer_idx)
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


def prune_neuronrank(args, model, tokenizer, device):
    if args.sparsity_type != "unstructured":
        raise ValueError("NeuronRank pruning currently only supports unstructured sparsity type.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("collecting NeuronRank statistics")
    stats = collect_neuronrank_statistics(
        model,
        dataloader,
        tokenizer,
        device,
        max_classes=args.neuronrank_max_classes,
    )
    scores = compute_neuronrank_scores(
        model,
        stats,
        token_weight=args.neuronrank_token_weight,
        discrimination_weight=args.nr_discrimination_weight,
    )
    pruned, total = apply_neuronrank_pruning(model, scores, args.sparsity_ratio)
    model.config.use_cache = use_cache

    if total:
        pct = 100.0 * pruned / total
    else:
        pct = 0.0
    print(f"NeuronRank pruned {pruned}/{total} channels across MLPs ({pct:.2f}% structural sparsity)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "neuronrank"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--neuronrank_token_weight', type=float, default=0.0,
                        help='Additional weight for token-level variance when computing NeuronRank scores (0 disables token variance contribution).')
    parser.add_argument('--nr-discrimination-weight', dest='nr_discrimination_weight', type=float, default=2.0,
                        help='Exponent applied to the combined variance term in NeuronRank scoring.')
    parser.add_argument('--neuronrank-max-classes', type=int, default=512,
                        help='Maximum number of high-frequency token classes to track when computing NeuronRank statistics (0 disables class-aware variance).')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "neuronrank":
            prune_neuronrank(args, model, tokenizer, device)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()