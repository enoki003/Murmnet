from typing import Dict, List, Tuple, Optional, Mapping, Sequence, TypeAlias, Protocol, runtime_checkable, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from ..tools.hf_compat import (
    HFDatasetLike,
    load_dataset_compat,
    auto_tokenizer_from_pretrained,
)

# Minimal protocol the mapped HF dataset objects satisfy after .map/select
@runtime_checkable
class _SupportsHF(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Mapping[str, object]: ...


def _get_eos_id(tok: PreTrainedTokenizerBase) -> int:
    eid = cast(Optional[int], getattr(tok, "eos_token_id", None))
    if isinstance(eid, int):
        return eid
    sid = cast(Optional[int], getattr(tok, "sep_token_id", None))
    if isinstance(sid, int):
        return sid
    uid = cast(Optional[int], getattr(tok, "unk_token_id", None))
    if isinstance(uid, int):
        return uid
    pid = cast(Optional[int], getattr(tok, "pad_token_id", None))
    if isinstance(pid, int):
        return pid
    return 0


# Type aliases for clarity and better type inference across modules
BatchDict: TypeAlias = Dict[str, Tensor]
Loader: TypeAlias = DataLoader[BatchDict]


def build_hf_dataloaders(
    task: str,
    dataset_size: str,
    seq_len: int,
    micro_batch: int,
    num_workers: int = 0,
) -> Tuple[DataLoader[Dict[str, Tensor]], DataLoader[Dict[str, Tensor]], PreTrainedTokenizerBase]:
    # Tokenizer (English defaults)
    tok_name = "gpt2"
    tok = auto_tokenizer_from_pretrained(tok_name)
    eos_id: int = _get_eos_id(tok)

    if task == "squad":
        ds = load_dataset_compat("squad")
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            q: str = str(ex["question"])  # hf schema
            c: str = str(ex["context"])   # hf schema
            answers_val = ex.get("answers")
            ans_list: List[str] = [""]
            if isinstance(answers_val, Mapping):
                answers_map = cast(Mapping[str, object], answers_val)
                text_val = answers_map.get("text")
                if isinstance(text_val, list):
                    seq = cast(Sequence[object], text_val)
                    ans_list = [str(t) for t in seq]
            ans: str = str(ans_list[0]) if len(ans_list) > 0 else ""
            inp = f"Q: {q}\nC: {c}\nA: "
            out: str = ans
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(out, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids = list(map(int, cast(List[int], enc_inp["input_ids"])))
            out_ids = list(map(int, cast(List[int], enc_out["input_ids"])))
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(_SupportsHF, ds["train"].map(encode, remove_columns=[]))
        dev = cast(_SupportsHF, ds["validation"].map(encode, remove_columns=[]))
    elif task == "cnndm":
        ds = load_dataset_compat("cnn_dailymail", "3.0.0")
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            art: str = str(ex["article"])     # hf schema
            summ: str = str(ex["highlights"])  # hf schema
            inp = f"Summarize:\n{art}\nSummary: "
            out: str = summ
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(out, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids = list(map(int, cast(List[int], enc_inp["input_ids"])))
            out_ids = list(map(int, cast(List[int], enc_out["input_ids"])))
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(_SupportsHF, ds["train"].map(encode, remove_columns=[]))
        dev = cast(_SupportsHF, ds["validation"].map(encode, remove_columns=[]))
    elif task == "sst2":
        ds = load_dataset_compat("glue", "sst2")
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            text: str = str(ex["sentence"])  # hf schema
            label_val = ex.get("label")
            label: int = int(label_val) if isinstance(label_val, (int, str)) else 0
            tgt = "positive" if label == 1 else "negative"
            inp = f"Review: {text}\nSentiment: "
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(tgt, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids = list(map(int, cast(List[int], enc_inp["input_ids"])))
            out_ids = list(map(int, cast(List[int], enc_out["input_ids"])))
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(_SupportsHF, ds["train"].map(encode, remove_columns=[]))
        dev = cast(_SupportsHF, ds["validation"].map(encode, remove_columns=[]))
    else:
        # Fallback to dummy handled elsewhere
        raise ValueError("HF loader supports only 'squad', 'cnndm', and 'sst2' in this minimal setup")

    if dataset_size == "small":
        train_like = cast(HFDatasetLike, train)
        dev_like = cast(HFDatasetLike, dev)
        train = cast(_SupportsHF, train_like.select(range(min(5000, len(train_like)))))
        dev = cast(_SupportsHF, dev_like.select(range(min(1000, len(dev_like)))))
    elif dataset_size == "full":
        # keep full size
        ...

    pad_raw = getattr(tok, "pad_token_id", None)
    pad_id: int = int(pad_raw) if isinstance(pad_raw, int) else eos_id

    def collate(batch: List[BatchDict]) -> BatchDict:
        input_ids = [batch[i]["input_ids"].to(dtype=torch.long) for i in range(len(batch))]
        labels = [batch[i]["labels"].to(dtype=torch.long) for i in range(len(batch))]
        # padding_value should match dtype (long)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}

    # Wrap HF datasets into a minimal torch Dataset to satisfy type checkers
    from torch.utils.data import Dataset as TorchDataset  # local import to avoid name clash

    class HFToTorchDataset(TorchDataset[BatchDict]):
        def __init__(self, ds: _SupportsHF):
            self.ds = ds
        def __len__(self) -> int:
            return int(len(self.ds))
        def __getitem__(self, idx: int) -> BatchDict:
            item = self.ds[idx]
            ids_list = list(map(int, cast(List[int], item["input_ids"])))
            lbl_list = list(map(int, cast(List[int], item["labels"])))
            return {
                "input_ids": torch.tensor(ids_list, dtype=torch.long),
                "labels": torch.tensor(lbl_list, dtype=torch.long),
            }

    train_torch = HFToTorchDataset(train)
    dev_torch = HFToTorchDataset(dev)

    train_loader: Loader = DataLoader(train_torch, batch_size=micro_batch, shuffle=True, num_workers=num_workers, collate_fn=collate)
    dev_loader: Loader = DataLoader(dev_torch, batch_size=micro_batch, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, dev_loader, tok
