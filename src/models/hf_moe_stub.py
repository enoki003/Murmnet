from typing import Any


class HFMoELoader:
    """Placeholder for HF MoE backend loader.

    実際のHFモデル読み込み/推論は、現在 train.py の --backend hf_moe 分岐で直接行います。
    このクラスは将来の整理のためのプレースホルダです。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "HFMoELoader is not used. Use --backend hf_moe in train.py for inference-only demo."
        )
