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
from typing import Dict, List, Mapping, Optional, TypedDict, Sized, cast, Protocol

# datasets may be missing at type-check time in some environments
from ..tools.hf_compat import load_dataset_compat, auto_tokenizer_from_pretrained, auto_config_from_pretrained

# Prefer public transformers imports; fall back to internal paths for strict type checkers
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class _TokenizerDecodeLike(Protocol):
    def decode(self, token_ids: List[int] | int, *, skip_special_tokens: bool = ...) -> str: ...

CACHE_DIR = os.environ.get("HF_HOME") or os.path.join(Path.home(), ".cache", "huggingface")


class TokenizerReport(TypedDict):
    model_id: str
    files: List[str]


class ConfigReport(TypedDict):
    model_id: str
    files: List[str]


class ModelArtifacts(TypedDict):
    tokenizer: TokenizerReport
    config: ConfigReport


def ensure_tokenizer(model_id: str) -> TokenizerReport:
    tok: PreTrainedTokenizerBase = auto_tokenizer_from_pretrained(model_id)
    tdec = cast(_TokenizerDecodeLike, tok)
    # touch decode to force tokenizer.json download if needed
    eid = cast(Optional[int], getattr(tok, "eos_token_id", None))
    if isinstance(eid, int):
        _ = tdec.decode([int(eid)])
    else:
        _ = tdec.decode([0])
    return {"model_id": model_id, "files": ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]}


def ensure_config(model_id: str) -> ConfigReport:
    _ = auto_config_from_pretrained(model_id)
    return {"model_id": model_id, "files": ["config.json"]}


def procure_models() -> Dict[str, ModelArtifacts]:
    report: Dict[str, ModelArtifacts] = {}
    # Example Switch Transformer model id; replace with a valid small MoE as available
    switch_small = os.environ.get("MURMNET_MOE_MODEL", "google/switch-base-16")
    report["switch_transformer"] = {
        "tokenizer": ensure_tokenizer(switch_small),
        "config": ensure_config(switch_small),
    }
    return report


class DatasetSplits(TypedDict):
    splits: Dict[str, int]
    builder: str


def procure_datasets() -> Dict[str, DatasetSplits]:
    report: Dict[str, DatasetSplits] = {}
    # SQuAD 1.1
    ds_squad = cast(Mapping[str, Sized], load_dataset_compat("squad"))
    report["squad"] = {"splits": {k: int(len(v)) for k, v in ds_squad.items()}, "builder": "squad"}
    # CNN/DailyMail 3.0.0
    ds_cnn = cast(Mapping[str, Sized], load_dataset_compat("cnn_dailymail", "3.0.0"))
    report["cnn_dailymail"] = {"splits": {k: int(len(v)) for k, v in ds_cnn.items()}, "builder": "cnn_dailymail/3.0.0"}
    # GLUE SST-2
    ds_sst2 = cast(Mapping[str, Sized], load_dataset_compat("glue", "sst2"))
    report["sst2"] = {"splits": {k: int(len(v)) for k, v in ds_sst2.items()}, "builder": "glue/sst2"}
    return report


def main() -> None:
    print({"cache_dir": CACHE_DIR})
    models = procure_models()
    datasets = procure_datasets()
    print({"models": models})
    print({"datasets": datasets})


if __name__ == "__main__":
    main()
