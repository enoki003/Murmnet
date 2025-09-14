"""
Procure external models and datasets into local Hugging Face cache.
- Models: Switch Transformer example (config/tokenizer only as placeholder)
- Datasets: SQuAD, CNN/DailyMail, SST-2

Notes:
- This repository's training backend for external MoE (hf_moe) is a stub. We cache artifacts to prepare future integration.
- Respect dataset/model licenses when using the artifacts beyond local testing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

CACHE_DIR = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")


def ensure_tokenizer(model_id: str) -> Dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(model_id)
    # touch decode to force tokenizer.json download if needed
    _ = tok.decode([tok.eos_token_id or 0]) if hasattr(tok, "eos_token_id") else None
    return {"model_id": model_id, "files": ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]}


def ensure_config(model_id: str) -> Dict[str, Any]:
    _ = AutoConfig.from_pretrained(model_id)
    return {"model_id": model_id, "files": ["config.json"]}


def procure_models() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    # Example Switch Transformer model id; replace with a valid small MoE as available
    switch_small = os.environ.get("MURMNET_MOE_MODEL", "google/switch-base-16")
    report["switch_transformer"] = {
        "tokenizer": ensure_tokenizer(switch_small),
        "config": ensure_config(switch_small),
    }
    return report


def procure_datasets() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    # SQuAD 1.1
    ds_squad = load_dataset("squad")
    report["squad"] = {"splits": {k: len(v) for k, v in ds_squad.items()}, "builder": "squad"}
    # CNN/DailyMail 3.0.0
    ds_cnn = load_dataset("cnn_dailymail", "3.0.0")
    report["cnn_dailymail"] = {"splits": {k: len(v) for k, v in ds_cnn.items()}, "builder": "cnn_dailymail/3.0.0"}
    # GLUE SST-2
    ds_sst2 = load_dataset("glue", "sst2")
    report["sst2"] = {"splits": {k: len(v) for k, v in ds_sst2.items()}, "builder": "glue/sst2"}
    return report


def main() -> None:
    print({"cache_dir": CACHE_DIR})
    models = procure_models()
    datasets = procure_datasets()
    print({"models": models})
    print({"datasets": datasets})


if __name__ == "__main__":
    main()
