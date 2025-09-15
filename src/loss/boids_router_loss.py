"""
Boids-style router regularization for HF Switch-Transformer routing probabilities.

This module doesn't depend on HF internals; it expects per-layer routing probabilities
of shape (batch, seq_len, num_experts), and an attention mask (batch, seq_len).

Loss components:
- separation (load-balance): encourage uniform expert utilization across tokens
- alignment (cohesion): encourage adjacent tokens to pick similar experts
- entropy (sharpening): encourage confident (low-entropy) assignments

All components are averaged across layers; each term is scale-invariant-ish and small.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import Tensor

EPS = 1e-8


def _masked_sum(x: Tensor, mask: Tensor, dim: Tuple[int, ...]) -> Tensor:
    # mask: (B, T) -> broadcast to x
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum(dim=dim)


def _per_expert_load(probs: Tensor, attn_mask: Tensor) -> Tensor:
    """Return per-expert fraction usage, shape (E,), sums to 1.
    probs: (B, T, E), attn_mask: (B, T)
    """
    _, _, _ = probs.shape
    token_counts = attn_mask.sum().clamp_min(1.0)
    per_expert = _masked_sum(probs, attn_mask, dim=(0, 1))  # (E)
    per_expert = per_expert / token_counts
    # normalize to sum 1 in case of numerical drift
    per_expert = per_expert / per_expert.sum().clamp_min(EPS)
    return per_expert


def load_balance_loss(probs: Tensor, attn_mask: Tensor) -> Tensor:
    """MSE between per-expert load and uniform.
    Encourages separation (avoid overcrowding) by distributing load.
    """
    e: int = int(probs.shape[-1])
    target: Tensor = probs.new_full((e,), 1.0 / float(e))
    per_expert = _per_expert_load(probs, attn_mask)
    return torch.mean((per_expert - target) ** 2)


def neighbor_alignment_loss(probs: Tensor, attn_mask: Tensor) -> Tensor:
    """Minimize 1 - cosine similarity between adjacent tokens' routing probs.
    Computes over valid neighbor positions only.
    """
    _, t, _ = probs.shape
    if t <= 1:
        return probs.new_tensor(0.0)
    # neighbors: (B, T-1, E)
    p0: Tensor = probs[:, :-1, :]
    p1: Tensor = probs[:, 1:, :]
    # mask for neighbor pairs
    m0: Tensor = attn_mask[:, :-1]
    m1: Tensor = attn_mask[:, 1:]
    m: Tensor = (m0 * m1).float()
    # cosine similarity
    num: Tensor = (p0 * p1).sum(dim=-1)
    # l2 norms via pow/sum/sqrt to keep type stubs happy
    n0 = p0.pow(2.0).sum(dim=-1).clamp_min(EPS).sqrt()
    n1 = p1.pow(2.0).sum(dim=-1).clamp_min(EPS).sqrt()
    den = (n0 * n1).clamp_min(EPS)
    cos = (num / den)
    diff = (1.0 - cos) * m
    denom = m.sum().clamp_min(1.0)
    return diff.sum() / denom


def entropy_loss(probs: Tensor, attn_mask: Tensor) -> Tensor:
    """Mean entropy across tokens (masked). Lower is sharper; encourage low entropy.
    """
    p: Tensor = probs.clamp_min(EPS)
    ent: Tensor = -(p * p.log()).sum(dim=-1)  # (B, T)
    denom: Tensor = attn_mask.sum().clamp_min(1.0)
    return (ent * attn_mask).sum() / denom


def compute_boids_router_loss(
    router_probs_per_layer: List[Tensor],
    attn_mask: Tensor,
    *,
    w_align: float = 1.0,
    w_separation: float = 1.0,
    w_entropy: float = 0.0,
) -> Dict[str, Tensor]:
    """Compute combined Boids-style regularization from a list of (B,T,E) tensors.

    Returns a dict with components and total: {total, align, separation, entropy}
    """
    if not router_probs_per_layer:
        zero = attn_mask.new_tensor(0.0)
        return {"total": zero, "align": zero, "separation": zero, "entropy": zero}
    align_acc = attn_mask.new_tensor(0.0)
    sep_acc = attn_mask.new_tensor(0.0)
    ent_acc = attn_mask.new_tensor(0.0)
    n = 0
    for probs in router_probs_per_layer:
        # ensure proper shape and dtype
        if probs.dim() != 3:
            continue
        if probs.size(0) != attn_mask.size(0):
            continue
        align_acc = align_acc + neighbor_alignment_loss(probs, attn_mask)
        sep_acc = sep_acc + load_balance_loss(probs, attn_mask)
        if w_entropy != 0.0:
            ent_acc = ent_acc + entropy_loss(probs, attn_mask)
        n += 1
    if n == 0:
        zero = attn_mask.new_tensor(0.0)
        return {"total": zero, "align": zero, "separation": zero, "entropy": zero}
    align = align_acc / n
    sep = sep_acc / n
    ent = ent_acc / max(1, n)
    total = w_align * align + w_separation * sep + w_entropy * ent
    return {"total": total, "align": align, "separation": sep, "entropy": ent}
# End of module
