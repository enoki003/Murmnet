"""
Facade for Boids-style router regularization.

Provides a single function `boids_regularize(router_probs_per_layer, attn_mask, weight, **kwargs)`
that computes the regularization loss and returns (total_loss, components_dict).
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from torch import Tensor

from ..loss.boids_router_loss import compute_boids_router_loss


def boids_regularize(
    router_probs_per_layer: List[Tensor],
    attn_mask: Tensor,
    *,
    weight: float = 0.01,
    w_align: float = 1.0,
    w_separation: float = 1.0,
    w_entropy: float = 0.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    comps = compute_boids_router_loss(
        router_probs_per_layer,
        attn_mask,
        w_align=w_align,
        w_separation=w_separation,
        w_entropy=w_entropy,
    )
    total = comps["total"] * float(weight)
    return total, comps
# End of module
