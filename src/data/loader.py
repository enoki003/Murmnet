from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class DummyLMSet(Dataset[Dict[str, Tensor]]):
    def __init__(self, size: int = 128, seq_len: int = 64, vocab: int = 32000, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.input_ids: Tensor = torch.randint(0, vocab, (size, seq_len), generator=g)
        self.labels: Tensor = self.input_ids.clone()

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def build_dataloaders(task: str, dataset_size: str, seq_len: int, micro_batch: int, num_workers: int = 2) -> Tuple[DataLoader[Dict[str, Tensor]], DataLoader[Dict[str, Tensor]]]:
    # Minimal: use dummy dataset; real datasets can be plugged with HuggingFace datasets later.
    if dataset_size == "dummy":
        train = DummyLMSet(size=128, seq_len=seq_len)
        dev = DummyLMSet(size=64, seq_len=seq_len, seed=123)
    else:
        # Placeholder: same as dummy for now
        train = DummyLMSet(size=1024 if dataset_size == "small" else 8192, seq_len=seq_len)
        dev = DummyLMSet(size=256, seq_len=seq_len, seed=123)

    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        return {"input_ids": input_ids, "labels": labels}

    train_loader: DataLoader[Dict[str, Tensor]] = DataLoader(train, batch_size=micro_batch, shuffle=True, num_workers=num_workers, collate_fn=collate)
    dev_loader: DataLoader[Dict[str, Tensor]] = DataLoader(dev, batch_size=micro_batch, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, dev_loader
