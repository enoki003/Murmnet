from typing import List

import torch


def self_consistency_rate(logits_list: List[torch.Tensor]) -> float:
    # logits_list: List[Tensor(B, T, V)]
    if len(logits_list) < 2:
        return 1.0
    preds = [l.argmax(dim=-1) for l in logits_list]
    base = preds[0]
    agree = [(p == base).float().mean().item() for p in preds[1:]]
    return float(sum(agree) / len(agree))


def embedding_variance(hidden_states: torch.Tensor) -> float:
    # hidden_states: (B, T, H)
    var = hidden_states.float().var(dim=(0, 1)).mean().item()
    return float(var)


def expert_entropy(util_probs: torch.Tensor) -> float:
    # util_probs: (E,)
    p = util_probs + 1e-9
    ent = -(p * (p.log())).sum().item()
    return float(ent)
