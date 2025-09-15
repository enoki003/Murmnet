from typing import Dict, Tuple
import torch  # 型注釈内で "torch.Tensor" を使用するため参照を用意

from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# 本ファイルは過去のダミーデータ実装を廃止し、Hugging Face datasets ローダーの薄いラッパーに置き換えました。
# すべてのデータは `src/data/hf_loader.py` の `build_hf_dataloaders` に委譲されます。


def build_dataloaders(
    task: str,
    dataset_size: str,
    seq_len: int,
    micro_batch: int,
    num_workers: int = 0,
) -> Tuple[DataLoader[Dict[str, "torch.Tensor"]], DataLoader[Dict[str, "torch.Tensor"]], PreTrainedTokenizerBase]:
    from .hf_loader import build_hf_dataloaders

    # dataset_size は "small" または "full" を想定
    if dataset_size not in {"small", "full"}:
        # 不正値は "small" にフォールバック
        dataset_size = "small"

    return build_hf_dataloaders(task, dataset_size, seq_len, micro_batch, num_workers=num_workers)
