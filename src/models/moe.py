from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    model_size: str = "small"  # tiny/small/base
    hidden_dim: int = 768
    ffn_dim: int = 3072
    num_layers: int = 6
    num_experts: int = 8
    top_k: int = 1
    router_dropout: float = 0.1
    load_balance_coef: float = 0.01
    vocab_size: int = 32000
    max_position: int = 2048
    # Router/MoE extras
    capacity_factor: float = 1.25
    temperature: float = 1.0
    fallback_self_ffn: bool = True


class TopKRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dropout: float = 0.0, temperature: float = 1.0) -> None:
        nn.Module.__init__(self)  # type: ignore[misc]
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(hidden_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.tau = temperature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, T, H)
        logits = self.linear(self.dropout(x))  # (B, T, E)
        g_soft = F.softmax(logits / self.tau, dim=-1)
        topk_idx = torch.topk(g_soft, k=self.top_k, dim=-1)[1]
        hard = torch.zeros_like(g_soft).scatter_(-1, topk_idx, 1.0)
        mask = hard.detach() - g_soft.detach() + g_soft
        g_top = g_soft * (mask > 0).float()
        g_top = g_top / (g_top.sum(dim=-1, keepdim=True) + 1e-9)
        return g_top, topk_idx, logits


class ExpertFFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        nn.Module.__init__(self)  # type: ignore[misc]
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int, router_dropout: float = 0.0, capacity_factor: float = 1.25, temperature: float = 1.0, fallback_self_ffn: bool = True) -> None:
        nn.Module.__init__(self)  # type: ignore[misc]
        self.router = TopKRouter(hidden_dim, num_experts, top_k, router_dropout, temperature)
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim, ffn_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.fallback_self_ffn = fallback_self_ffn
        self.self_ffn: Optional[ExpertFFN] = ExpertFFN(hidden_dim, ffn_dim) if fallback_self_ffn else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, H = x.shape
        gates, idx, logits = self.router(x)  # (B, T, E), (B, T, K), (B, T, E)
        N = B * T
        gates_f = gates.view(N, self.num_experts)
        idx_f = idx.view(N, self.top_k)
        x_f = x.view(N, H)
        device = x.device
        # Capacity per expert by primary assignment
        cap = max(1, int((N / self.num_experts) * self.capacity_factor))
        primary = idx_f[:, 0]
        keep_primary = torch.zeros(N, dtype=torch.bool, device=device)
        assign = torch.full((N,), -1, dtype=torch.long, device=device)
        used = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        for e in range(self.num_experts):
            mask_e = (primary == e).nonzero(as_tuple=False).squeeze(-1)
            if mask_e.numel() == 0:
                continue
            n = int(min(cap - int(used[e].item()), mask_e.numel()))
            if n > 0:
                sel = mask_e[:n]
                keep_primary[sel] = True
                assign[sel] = e
                used[e] += n
        # Backup to second expert
        overflow = ~keep_primary
        if self.top_k > 1:
            second = idx_f[:, 1]
            for e in range(self.num_experts):
                if used[e] >= cap:
                    continue
                mask_e = ((second == e) & overflow).nonzero(as_tuple=False).squeeze(-1)
                if mask_e.numel() == 0:
                    continue
                free = cap - int(used[e].item())
                n = int(min(free, mask_e.numel()))
                if n > 0:
                    sel = mask_e[:n]
                    assign[sel] = e
                    used[e] += n
                    overflow[sel] = False
        # Fallback to self FFN
        use_fallback = overflow & (self.self_ffn is not None)

        # Compute expert outputs (simple full compute)
        expert_outputs = [self.experts[e](x_f) for e in range(self.num_experts)]
        expert_stack = torch.stack(expert_outputs, dim=1)  # (N, E, H)
        one_hot = torch.zeros(N, self.num_experts, device=device, dtype=expert_stack.dtype)
        valid = assign >= 0
        one_hot[valid, assign[valid]] = 1.0
        y = (expert_stack * one_hot.unsqueeze(-1)).sum(dim=1)
        if use_fallback.any():
            assert self.self_ffn is not None
            y_fb = self.self_ffn(x_f[use_fallback])
            y[use_fallback] = y_fb
        y = y.view(B, T, H)

        util = one_hot.mean(dim=0)
        with torch.no_grad():
            f = torch.zeros(self.num_experts, device=device, dtype=expert_stack.dtype)
            f.scatter_add_(0, primary, torch.ones_like(primary, dtype=expert_stack.dtype))
            f = f / N
        P = gates_f.mean(dim=0)
        stats: Dict[str, torch.Tensor] = {"f": f, "P": P, "logits": logits.view(N, self.num_experts), "gates_soft": gates_f, "topk_idx": idx_f}
        return y, util, stats


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, moe: Optional[MoELayer] = None) -> None:
        nn.Module.__init__(self)  # type: ignore[misc]
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp: Optional[nn.Sequential] = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim)) if moe is None else None
        self.moe = moe

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        h, _ = self.attn(x, x, x)
        x = x + h
        x = self.ln1(x)
        if self.moe is None:
            assert self.mlp is not None
            h = self.mlp(x)
            x = self.ln2(x + h)
            util = None
            stats = None
        else:
            h, util, stats = self.moe(x)
            x = self.ln2(x + h)
        return x, util, stats


class TinyMoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        nn.Module.__init__(self)  # type: ignore[misc]
        self.cfg = cfg
        H = cfg.hidden_dim
        self.emb = nn.Embedding(cfg.vocab_size, H)
        self.pos = nn.Embedding(cfg.max_position, H)
        blocks: List[TransformerBlock] = []
        heads = 12 if H >= 768 else 8
        for i in range(cfg.num_layers):
            # Insert MoE every other block for simplicity
            moe = MoELayer(
                H,
                cfg.ffn_dim,
                cfg.num_experts,
                cfg.top_k,
                cfg.router_dropout,
                capacity_factor=cfg.capacity_factor,
                temperature=cfg.temperature,
                fallback_self_ffn=cfg.fallback_self_ffn,
            ) if (i % 2 == 1) else None
            blocks.append(TransformerBlock(H, heads, cfg.ffn_dim, moe))
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(H)
        self.head = nn.Linear(H, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        T = input_ids.size(1)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.emb(input_ids) + self.pos(pos)
        utils: List[torch.Tensor] = []
        moe_stats: List[Dict[str, torch.Tensor]] = []
        for blk in self.blocks:
            x, util, stats = blk(x)
            if util is not None:
                utils.append(util)
            if stats is not None:
                moe_stats.append(stats)
        x = self.ln(x)
        logits = self.head(x)
        if len(utils) == 0:
            util_stack = torch.zeros(self.cfg.num_experts, device=logits.device)
        else:
            util_stack = torch.stack(utils).mean(dim=0)
        if return_hidden:
            return logits, util_stack, x, moe_stats
        return logits, util_stack
