import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.data import get_loaders
from lib.prune import (
    prune_wanda,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    prune_neuronrank_unstructured,
    check_sparsity,
)
from lib.eval import eval_ppl, eval_zero_shot
from lib.neuronrank import (
    collect_neuronrank_statistics,
    compute_neuronrank_scores,
    compute_neuronrank_class_scores,
    apply_neuronrank_pruning,
)

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


def prune_neuronrank(args, model, tokenizer, device):
    if args.sparsity_type != "unstructured":
        raise ValueError("NeuronRank pruning currently only supports unstructured sparsity type.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
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
        discrimination_multi=args.discrimination_multi,
        discrimination_exp=args.discrimination_exp,
        magnitude_multi=args.magnitude_multi,
    )
    pruned, total = apply_neuronrank_pruning(model, scores, args.sparsity_ratio)
    model.config.use_cache = use_cache

    pct = 100.0 * pruned / total if total else 0.0
    print(f"NeuronRank pruned {pruned}/{total} channels across MLPs ({pct:.2f}% structural sparsity)")


def prune_neuronrank_variance(args, model, tokenizer, device):
    if args.sparsity_type != "unstructured":
        raise ValueError("NeuronRank variance-only pruning currently only supports unstructured sparsity type.")
    if args.neuronrank_max_classes <= 0:
        raise ValueError("NeuronRank variance-only pruning requires --neuronrank-max-classes > 0 to collect token-class statistics.")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("collecting NeuronRank class variance statistics")
    stats = collect_neuronrank_statistics(
        model,
        dataloader,
        tokenizer,
        device,
        max_classes=args.neuronrank_max_classes,
    )
    scores = compute_neuronrank_class_scores(stats)
    if not scores:
        raise RuntimeError("NeuronRank variance-only pruning could not compute class variance scores (no classes collected).")

    pruned, total = apply_neuronrank_pruning(model, scores, args.sparsity_ratio)
    model.config.use_cache = use_cache

    pct = 100.0 * pruned / total if total else 0.0
    print(f"NeuronRank variance-only pruned {pruned}/{total} channels across MLPs ({pct:.2f}% structural sparsity)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "neuronrank", "neuronrank_unstructured", "neuronrank_variance"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--neuronrank_token_weight', type=float, default=0.0,
                        help='Additional weight for token-level variance when computing NeuronRank scores (0 disables token variance contribution).')
    parser.add_argument('--discrimination-multi', dest='discrimination_multi', type=float, default=1.0,
                        help='Multiplier applied to the variance-based discrimination term before exponentiation.')
    parser.add_argument('--discrimination-exp', dest='discrimination_exp', type=float, default=2.0,
                        help='Exponent applied to the scaled variance-based discrimination term.')
    parser.add_argument('--magnitude-multi', dest='magnitude_multi', type=float, default=1.0,
                        help='Multiplier applied to the weight-magnitude term in NeuronRank scoring.')
    parser.add_argument('--nr-discrimination-weight', dest='nr_discrimination_weight', type=float, default=None,
                        help='[Deprecated] Alias for --discrimination-exp; will be removed in a future release.')
    parser.add_argument('--neuronrank-max-classes', type=int, default=512,
                        help='Maximum number of high-frequency token classes to track when computing NeuronRank statistics (0 disables class-aware variance).')
    parser.add_argument('--nr-include-attention', dest='nr_include_attention', action='store_true',
                        help='When using NeuronRank unstructured pruning, also prune attention projection weights (default).')
    parser.add_argument('--nr-skip-attention', dest='nr_include_attention', action='store_false',
                        help='Skip pruning attention projection weights in NeuronRank unstructured mode.')
    parser.set_defaults(nr_include_attention=True)
    parser.add_argument('--nr-prune-lm-head', action='store_true',
                        help='Also prune the LM head using magnitude when running NeuronRank unstructured pruning.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    if getattr(args, "nr_discrimination_weight", None) is not None:
        if args.discrimination_exp == parser.get_default("discrimination_exp"):
            args.discrimination_exp = args.nr_discrimination_weight
        print("[Warning] --nr-discrimination-weight is deprecated; please use --discrimination-exp instead.")

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
        elif args.prune_method == "neuronrank_unstructured":
            prune_neuronrank_unstructured(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "neuronrank_variance":
            prune_neuronrank_variance(args, model, tokenizer, device)

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