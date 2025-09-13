import torch

def overuse_rate(util_probs: torch.Tensor, threshold: float = 0.2) -> float:
    """
    偏り超過率: ある専門家の平均利用確率が threshold を超える割合。
    util_probs: (E,)
    """
    return float((util_probs > threshold).float().mean().item())
