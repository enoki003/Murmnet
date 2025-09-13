from typing import Dict, List, Tuple, Optional, Mapping, TypeAlias, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset as HFDataset, DatasetDict  # type: ignore[import-not-found]
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


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
    num_workers: int = 2,
) -> Tuple[DataLoader[Dict[str, Tensor]], DataLoader[Dict[str, Tensor]], PreTrainedTokenizerBase]:
    # Tokenizer (English defaults)
    tok_name = "gpt2"
    tok = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tok_name))  # type: ignore[reportUnknownMemberType]
    eos_id: int = _get_eos_id(tok)

    if task == "squad":
        ds = cast(DatasetDict, load_dataset("squad"))  # type: ignore[reportUnknownMemberType]
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            q: str = str(ex["question"])  # type: ignore[index]
            c: str = str(ex["context"])  # type: ignore[index]
            answers = ex.get("answers", {}) if isinstance(ex.get("answers", {}), dict) else {}  # type: ignore[arg-type]
            ans_list: List[str] = cast(List[str], answers.get("text", [""])) if isinstance(answers, dict) else [""]  # type: ignore[call-arg]
            ans: str = str(ans_list[0]) if len(ans_list) > 0 else ""
            inp = f"Q: {q}\nC: {c}\nA: "
            out: str = ans
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(out, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids: List[int] = cast(List[int], enc_inp["input_ids"])  # type: ignore[index]
            out_ids: List[int] = cast(List[int], enc_out["input_ids"])   # type: ignore[index]
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        train: HFDataset = cast(HFDataset, ds["train"].map(encode, remove_columns=ds["train"].column_names))  # type: ignore[reportUnknownMemberType]
        dev: HFDataset = cast(HFDataset, ds["validation"].map(encode, remove_columns=ds["validation"].column_names))  # type: ignore[reportUnknownMemberType]
    elif task == "cnndm":
        ds = cast(DatasetDict, load_dataset("cnn_dailymail", "3.0.0"))  # type: ignore[reportUnknownMemberType]
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            art: str = str(ex["article"])  # type: ignore[index]
            summ: str = str(ex["highlights"])  # type: ignore[index]
            inp = f"Summarize:\n{art}\nSummary: "
            out: str = summ
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(out, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids: List[int] = cast(List[int], enc_inp["input_ids"])  # type: ignore[index]
            out_ids: List[int] = cast(List[int], enc_out["input_ids"])   # type: ignore[index]
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(HFDataset, ds["train"].map(encode, remove_columns=ds["train"].column_names))  # type: ignore[reportUnknownMemberType]
        dev = cast(HFDataset, ds["validation"].map(encode, remove_columns=ds["validation"].column_names))  # type: ignore[reportUnknownMemberType]
    elif task == "sst2":
        ds = cast(DatasetDict, load_dataset("glue", "sst2"))  # type: ignore[reportUnknownMemberType]
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            text: str = str(ex["sentence"])  # type: ignore[index]
            label: int = int(ex["label"])  # type: ignore[index]
            tgt = "positive" if label == 1 else "negative"
            inp = f"Review: {text}\nSentiment: "
            enc_inp = tok(inp, truncation=True, max_length=seq_len, add_special_tokens=False)
            enc_out = tok(tgt, truncation=True, max_length=seq_len, add_special_tokens=False)
            inp_ids: List[int] = cast(List[int], enc_inp["input_ids"])  # type: ignore[index]
            out_ids: List[int] = cast(List[int], enc_out["input_ids"])   # type: ignore[index]
            input_ids: List[int] = inp_ids + out_ids + [eos_id]
            labels: List[int] = [-100] * len(inp_ids) + out_ids + [eos_id]
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            return {"input_ids": input_ids, "labels": labels}
        split_map = {"train": "train", "validation": "validation"}
        train = cast(HFDataset, ds[split_map["train"]].map(encode, remove_columns=ds["train"].column_names))  # type: ignore[reportUnknownMemberType]
        dev = cast(HFDataset, ds[split_map["validation"]].map(encode, remove_columns=ds["validation"].column_names))  # type: ignore[reportUnknownMemberType]
    else:
        # Fallback to dummy handled elsewhere
        raise ValueError("HF loader supports only 'squad', 'cnndm', and 'sst2' in this minimal setup")

    if dataset_size == "small":
        # HF datasets stubs are incomplete; silence unknown member type on select
        train = cast(HFDataset, train.select(range(min(5000, len(train)))))  # type: ignore[reportUnknownMemberType]
        dev = cast(HFDataset, dev.select(range(min(1000, len(dev)))))        # type: ignore[reportUnknownMemberType]
    elif dataset_size == "full":
        pass
    else:
        # 'dummy' shouldn't route here
        train = cast(HFDataset, train.select(range(min(512, len(train)))))   # type: ignore[reportUnknownMemberType]
        dev = cast(HFDataset, dev.select(range(min(128, len(dev)))))         # type: ignore[reportUnknownMemberType]

    pad_raw = getattr(tok, "pad_token_id", None)
    pad_id: int = int(pad_raw) if isinstance(pad_raw, int) else eos_id

    def collate(batch: List[BatchDict]) -> BatchDict:
        input_ids = [batch[i]["input_ids"].to(dtype=torch.long) for i in range(len(batch))]
        labels = [batch[i]["labels"].to(dtype=torch.long) for i in range(len(batch))]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=float(pad_id))
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}

    # Wrap HF datasets into a minimal torch Dataset to satisfy type checkers
    from torch.utils.data import Dataset as TorchDataset  # local import to avoid name clash

    class HFToTorchDataset(TorchDataset[BatchDict]):
        def __init__(self, ds: HFDataset):
            self.ds = ds
        def __len__(self) -> int:
            return int(len(self.ds))  # type: ignore[arg-type]
        def __getitem__(self, idx: int) -> BatchDict:
            item = cast(Mapping[str, object], self.ds[idx])  # type: ignore[index]
            ids_list = cast(List[int], item["input_ids"])  # type: ignore[index]
            lbl_list = cast(List[int], item["labels"])     # type: ignore[index]
            return {
                "input_ids": torch.tensor(ids_list, dtype=torch.long),
                "labels": torch.tensor(lbl_list, dtype=torch.long),
            }

    train_torch = HFToTorchDataset(train)
    dev_torch = HFToTorchDataset(dev)

    train_loader: Loader = DataLoader(train_torch, batch_size=micro_batch, shuffle=True, num_workers=num_workers, collate_fn=collate)
    dev_loader: Loader = DataLoader(dev_torch, batch_size=micro_batch, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, dev_loader, tok
