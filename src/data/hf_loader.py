from typing import Dict, List, Tuple, Optional, Mapping, Sequence, TypeAlias, Protocol, runtime_checkable, cast, Callable, Iterable, Any
import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import importlib

# Minimal protocol the mapped HF dataset objects satisfy after .map/select
@runtime_checkable
class _SupportsHF(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Mapping[str, object]: ...


# Minimal Protocols to keep typing strict while using HF directly
@runtime_checkable
class HFDatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Mapping[str, object]: ...
    def map(
        self,
        function: Callable[[Mapping[str, object]], Mapping[str, object]]
        | Callable[[Mapping[str, object]], Mapping[str, Sequence[int]]],
        *,
        remove_columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> "HFDatasetLike": ...
    def select(self, indices: Iterable[int]) -> "HFDatasetLike": ...


@runtime_checkable
class HFDatasetDictLike(Protocol):
    def __getitem__(self, key: str) -> HFDatasetLike: ...
    def keys(self) -> Iterable[str]: ...
    def items(self) -> Iterable[tuple[str, HFDatasetLike]]: ...


def _load_dataset_compat(builder: str, subset: Optional[str] = None) -> HFDatasetDictLike:
    datasets_mod = importlib.import_module("datasets")
    # Windows 環境では HF datasets のキャッシュ周りでファイル移動が失敗することがあるため、
    # map 時のキャッシュ書き込みを抑止する目的でグローバルにキャッシュ無効化（Windows のみ）
    try:
        if os.name == "nt":
            disable_caching = getattr(datasets_mod, "disable_caching", None)
            if callable(disable_caching):
                disable_caching()
    except Exception:
        # ベストエフォート（失敗しても致命的ではない）
        pass
    load_dataset_func: Callable[..., object] = getattr(datasets_mod, "load_dataset")
    ds_obj = load_dataset_func(builder, subset) if subset is not None else load_dataset_func(builder)
    return cast(HFDatasetDictLike, ds_obj)


def _auto_tokenizer_from_pretrained(model_id: str) -> PreTrainedTokenizerBase:
    tok_mod = importlib.import_module("transformers.models.auto.tokenization_auto")
    AutoTokenizer = getattr(tok_mod, "AutoTokenizer")
    tok_obj = AutoTokenizer.from_pretrained(model_id)
    return cast(PreTrainedTokenizerBase, tok_obj)


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
    # Subsampling controls (applied after encoding and size preset)
    train_fraction: float = 1.0,
    eval_fraction: float = 1.0,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    *,
    model_id: str = "google/switch-base-16",
) -> Tuple[DataLoader[Dict[str, Tensor]], DataLoader[Dict[str, Tensor]], PreTrainedTokenizerBase]:
    """Build seq2seq-style dataloaders for HF models.

    Returns batches with keys:
      - input_ids: encoder token ids
      - labels: decoder token ids (padded with -100 during collate)
    """
    tok = _auto_tokenizer_from_pretrained(model_id)
    # Try to ensure a pad token exists for batching
    if getattr(tok, "pad_token", None) is None:
        pad_fallback = getattr(tok, "eos_token", None) or getattr(tok, "sep_token", None) or getattr(tok, "unk_token", None)
        if pad_fallback is not None:
            setattr(tok, "pad_token", pad_fallback)
    eos_id: int = _get_eos_id(tok)

    if task == "squad":
        ds = _load_dataset_compat("squad")
        # Prepare sources (preselect small before map to avoid tokenizing whole split)
        train_src = cast(HFDatasetLike, ds["train"])  # type: ignore[index]
        dev_src = cast(HFDatasetLike, ds["validation"])  # type: ignore[index]
        if dataset_size == "small":
            try:
                train_src = train_src.select(range(min(5000, len(train_src))))
                dev_src = dev_src.select(range(min(1000, len(dev_src))))
            except Exception:
                pass
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
            src = f"Q: {q}\nC: {c}\nA:"
            tgt: str = ans
            enc_src = tok(src, truncation=True, max_length=seq_len, add_special_tokens=True)
            enc_tgt = tok(tgt, truncation=True, max_length=seq_len, add_special_tokens=True)
            input_ids = list(map(int, cast(List[int], enc_src["input_ids"])))
            labels = list(map(int, cast(List[int], enc_tgt["input_ids"])))
            if len(labels) == 0:
                labels = [eos_id]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(
            _SupportsHF,
            train_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-squad-train",
            ),
        )
        dev = cast(
            _SupportsHF,
            dev_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-squad-dev",
            ),
        )
    elif task == "cnndm":
        ds = _load_dataset_compat("cnn_dailymail", "3.0.0")
        train_src = cast(HFDatasetLike, ds["train"])  # type: ignore[index]
        dev_src = cast(HFDatasetLike, ds["validation"])  # type: ignore[index]
        if dataset_size == "small":
            try:
                train_src = train_src.select(range(min(5000, len(train_src))))
                dev_src = dev_src.select(range(min(1000, len(dev_src))))
            except Exception:
                pass
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            art: str = str(ex["article"])     # hf schema
            summ: str = str(ex["highlights"])  # hf schema
            src = f"Summarize:\n{art}\nSummary:"
            tgt: str = summ
            enc_src = tok(src, truncation=True, max_length=seq_len, add_special_tokens=True)
            enc_tgt = tok(tgt, truncation=True, max_length=seq_len, add_special_tokens=True)
            input_ids = list(map(int, cast(List[int], enc_src["input_ids"])))
            labels = list(map(int, cast(List[int], enc_tgt["input_ids"])))
            if len(labels) == 0:
                labels = [eos_id]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(
            _SupportsHF,
            train_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-cnndm-train",
            ),
        )
        dev = cast(
            _SupportsHF,
            dev_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-cnndm-dev",
            ),
        )
    elif task == "sst2":
        ds = _load_dataset_compat("glue", "sst2")
        train_src = cast(HFDatasetLike, ds["train"])  # type: ignore[index]
        dev_src = cast(HFDatasetLike, ds["validation"])  # type: ignore[index]
        if dataset_size == "small":
            try:
                train_src = train_src.select(range(min(5000, len(train_src))))
                dev_src = dev_src.select(range(min(1000, len(dev_src))))
            except Exception:
                pass
        def encode(ex: Mapping[str, object]) -> Dict[str, List[int]]:
            text: str = str(ex["sentence"])  # hf schema
            label_val = ex.get("label")
            label: int = int(label_val) if isinstance(label_val, (int, str)) else 0
            tgt = "positive" if label == 1 else "negative"
            src = f"Review: {text}\nSentiment:"
            enc_src = tok(src, truncation=True, max_length=seq_len, add_special_tokens=True)
            enc_tgt = tok(tgt, truncation=True, max_length=seq_len, add_special_tokens=True)
            input_ids = list(map(int, cast(List[int], enc_src["input_ids"])))
            labels = list(map(int, cast(List[int], enc_tgt["input_ids"])))
            if len(labels) == 0:
                labels = [eos_id]
            return {"input_ids": input_ids, "labels": labels}
        train = cast(
            _SupportsHF,
            train_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-sst2-train",
            ),
        )
        dev = cast(
            _SupportsHF,
            dev_src.map(
                encode,
                remove_columns=[],
                load_from_cache_file=False,
                keep_in_memory=True,
                num_proc=1,
                desc="tokenize-sst2-dev",
            ),
        )
    else:
        raise ValueError("HF loader supports only 'squad', 'cnndm', and 'sst2' in this minimal setup")

    if dataset_size == "small":
        train_like = cast(HFDatasetLike, train)
        dev_like = cast(HFDatasetLike, dev)
        train = cast(_SupportsHF, train_like.select(range(min(5000, len(train_like)))))
        dev = cast(_SupportsHF, dev_like.select(range(min(1000, len(dev_like)))))
    elif dataset_size == "full":
        # keep full size
        ...

    # Optional subsampling by fraction and/or cap
    def _apply_subsample(ds_sup: _SupportsHF, frac: float, cap: int) -> _SupportsHF:
        ds_like = cast(HFDatasetLike, ds_sup)
        n_total = len(ds_like)
        n = n_total
        if 0.0 < frac < 1.0:
            n = max(1, int(n_total * frac))
        if cap > 0:
            n = min(n, cap)
        if n < n_total:
            return cast(_SupportsHF, ds_like.select(range(n)))
        return ds_sup

    train = _apply_subsample(train, train_fraction, max_train_samples)
    dev = _apply_subsample(dev, eval_fraction, max_eval_samples)

    pad_raw = getattr(tok, "pad_token_id", None)
    pad_id: int = int(pad_raw) if isinstance(pad_raw, int) else eos_id

    def collate(batch: List[BatchDict]) -> BatchDict:
        inputs = [ex["input_ids"].to(dtype=torch.long) for ex in batch]
        lbls = [ex["labels"].to(dtype=torch.long) for ex in batch]
        inputs_pad = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        labels_pad = torch.nn.utils.rnn.pad_sequence(lbls, batch_first=True, padding_value=-100)
        return {"input_ids": inputs_pad, "labels": labels_pad}

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
