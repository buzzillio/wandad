# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import gzip
import json
import os
from pathlib import Path
import numpy as np
import random
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from urllib.request import urlretrieve
import zipfile


'''

python main.py \
  --model baffo32/decapoda-research-llama-7B-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save out/llama_7b/unstructured/wanda/


'''


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def _ensure_wikitext_s3_file(filename):
    cache_dir = Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache"))) / "wandad" / "wikitext-2-raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_file = cache_dir / filename
    if target_file.exists():
        return str(target_file)

    zip_path = cache_dir / "wikitext-2-raw-v1.zip"
    if not zip_path.exists():
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        print(f"[info] Downloading wikitext-2-raw-v1 from {url}")
        urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        inner_path = f"wikitext-2-raw/wiki.{filename.split('.',1)[1]}"
        with zf.open(inner_path) as source, open(target_file, "wb") as dest:
            dest.write(source.read())
    return str(target_file)


def _load_wikitext_split(filename):
    try:
        path = hf_hub_download(
            repo_id="wikitext",
            filename=f"wikitext-2-raw-v1/{filename}",
            repo_type="dataset",
        )
    except Exception:
        path = _ensure_wikitext_s3_file(filename)

    with open(path, "r", encoding="utf-8") as f:
        # Mirror the original dataset: each line represents one sample, keep blanks
        lines = [line.rstrip("\n") for line in f]
    return lines


# Load wikitext2 dataset via datasets library only (while returning tokenized calibration tuples)
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = ds["train"]["text"]
    test_text = ds["test"]["text"]

    trainenc = tokenizer(" ".join(train_text), return_tensors='pt')
    testenc = tokenizer("\n\n".join(test_text), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc

# Load and process c4 dataset
# def get_c4(nsamples, seed, seqlen, tokenizer):
#     # Load train and validation datasets
#     traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
#     valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')


def _download_c4_shard(filename):
    return hf_hub_download(
        repo_id="allenai/c4",
        filename=f"en/{filename}",
        repo_type="dataset",
    )


def _iter_c4_records(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if text:
                yield text


def get_c4(nsamples, seed, seqlen, tokenizer):
    set_seed(seed)
    train_path = _download_c4_shard("c4-train.00000-of-01024.json.gz")
    val_path = _download_c4_shard("c4-validation.00000-of-00008.json.gz")

    candidate_tokens = []
    max_candidates = max(nsamples * 20, 2000)
    for text in _iter_c4_records(train_path):
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] > seqlen:
            candidate_tokens.append(enc.input_ids)
        if len(candidate_tokens) >= max_candidates:
            break

    if not candidate_tokens:
        raise RuntimeError("Unable to collect C4 calibration samples longer than seqlen")

    trainloader = []
    attempts = 0
    max_attempts = nsamples * 20
    while len(trainloader) < nsamples and attempts < max_attempts:
        attempts += 1
        enc = random.choice(candidate_tokens)
        length = enc.shape[1]
        if length <= seqlen:
            continue
        start = random.randint(0, length - seqlen - 1)
        end = start + seqlen
        inp = enc[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    if len(trainloader) < nsamples:
        raise RuntimeError(f"Could only generate {len(trainloader)} C4 calibration samples out of requested {nsamples}")

    val_texts = []
    for text in _iter_c4_records(val_path):
        val_texts.append(text)
        if len(val_texts) >= 1100:
            break

    if not val_texts:
        raise RuntimeError("Validation split of C4 did not yield any text records")

    valenc = tokenizer(" ".join(val_texts), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)