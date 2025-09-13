from typing import NoReturn


class HFMoELoader:
    """Stub for loading HF MoE models (Switch/Qwen2-MoE/etc.).

    現状のTransformersではSwitch TransformerはFlax実装が中心で、PyTorchでの微調整は未対応ケースが多いです。
    将来的にPyTorch対応MoEモデル（例: Qwen2-MoE系）を安全に差し替えるためのプレースホルダです。
    """

    def __init__(self, model_name: str = "google/switch-base-16", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def get_model(self) -> NoReturn:
        raise NotImplementedError(
            "HF MoE backend is not yet implemented in this repo. "
            "Switch Transformer on HF is primarily Flax/JAX; for PyTorch training, use TinyMoE backend for now."
        )
