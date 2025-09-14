"""
Thin compatibility wrappers and Protocols for Hugging Face libraries to keep
our codebase strictly typed without resorting to type: ignore.

We expose minimal Protocols satisfied by datasets.Dataset / DatasetDict after
map/select, and typed constructors for AutoTokenizer/AutoConfig.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable, cast
import importlib

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

T = TypeVar("T")


@runtime_checkable
class HFDatasetLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Mapping[str, object]: ...
    def map(self, function: Callable[[Mapping[str, object]], Mapping[str, object]] | Callable[[Mapping[str, object]], Mapping[str, Sequence[int]]], *, remove_columns: Sequence[str] | None = None) -> "HFDatasetLike": ...
    def select(self, indices: Iterable[int]) -> "HFDatasetLike": ...


@runtime_checkable
class HFDatasetDictLike(Protocol):
    def __getitem__(self, key: str) -> HFDatasetLike: ...
    def keys(self) -> Iterable[str]: ...
    def items(self) -> Iterable[tuple[str, HFDatasetLike]]: ...


def load_dataset_compat(builder: str, subset: str | None = None) -> HFDatasetDictLike:
    datasets_mod = importlib.import_module("datasets")
    load_dataset_func: Callable[..., Any] = getattr(datasets_mod, "load_dataset")
    ds_obj = load_dataset_func(builder, subset) if subset is not None else load_dataset_func(builder)
    return cast(HFDatasetDictLike, ds_obj)


def auto_tokenizer_from_pretrained(model_id: str) -> PreTrainedTokenizerBase:
    # Resolve from the canonical module to satisfy strict export checks
    tok_mod = importlib.import_module("transformers.models.auto.tokenization_auto")
    AutoTokenizer = getattr(tok_mod, "AutoTokenizer")
    from_pretrained: Callable[..., Any] = getattr(AutoTokenizer, "from_pretrained")
    tok_obj = from_pretrained(model_id)
    return cast(PreTrainedTokenizerBase, tok_obj)


def auto_config_from_pretrained(model_id: str) -> object:
    cfg_mod = importlib.import_module("transformers.models.auto.configuration_auto")
    AutoConfig = getattr(cfg_mod, "AutoConfig")
    from_pretrained: Callable[..., Any] = getattr(AutoConfig, "from_pretrained")
    cfg_obj = from_pretrained(model_id)
    # We don't rely on static members from the config
    return cast(object, cfg_obj)
