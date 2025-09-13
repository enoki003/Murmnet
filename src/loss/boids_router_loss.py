import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class BoidsRouterLoss(nn.Module):
    def __init__(
        self,
        lambda_coh: float = 0.1,
        lambda_sep: float = 0.05,
        lambda_ali: float = 0.01,
        k_nn: int = 8,
        sep_tau: float = 1.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.lc: float = lambda_coh
        self.ls: float = lambda_sep
        self.la: float = lambda_ali
        self.k: int = k_nn
        self.tau: float = sep_tau
        self.eps: float = eps

    @staticmethod
    def _js_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum(-1)
        kl_qm = (q * (q / m).log()).sum(-1)
        return 0.5 * (kl_pm + kl_qm)

    def forward(
        self,
        z: torch.Tensor,
        gates_soft: torch.Tensor,
        topk_idx: torch.Tensor,
        num_experts: int,
    ) -> Dict[str, torch.Tensor]:
        """
        z: (N, d) routing embeddings
        gates_soft: (N, E) soft gate distribution (softmax)
        topk_idx: (N, k) hard top-k expert ids
        num_experts: int
        returns: dict with 'coh','sep','ali','boids'
        """
        N, E = gates_soft.shape
        # Cohesion via kNN on cosine
        z_norm = F.normalize(z, p=2, dim=-1)
        sim = z_norm @ z_norm.t()
        sim.fill_diagonal_(0.0)
        k = min(self.k, max(1, N - 1))
        knn_val, knn_idx = torch.topk(sim, k=k, dim=-1, largest=True)
        gi = gates_soft.unsqueeze(1).expand(-1, knn_idx.size(1), -1)
        gj = gates_soft[knn_idx]
        js = self._js_div(gi, gj, self.eps)
        L_coh = (knn_val * js).mean()

        # Separation: expert usage balance from hard top-1 indices
        counts = torch.zeros(E, device=gates_soft.device, dtype=gates_soft.dtype)
        counts.scatter_add_(0, topk_idx[:, 0].reshape(-1), torch.ones_like(topk_idx[:, 0], dtype=gates_soft.dtype))
        n_bar = counts.mean().clamp_min(self.eps)
        over = (counts / n_bar) - self.tau
        L_sep = torch.clamp(over, min=0).pow(2).mean()

        # Alignment: gates vs batch mean
        g_bar = gates_soft.mean(dim=0, keepdim=True)
        L_ali = self._js_div(gates_soft, g_bar.expand_as(gates_soft), self.eps).mean()

        loss = self.lc * L_coh + self.ls * L_sep + self.la * L_ali
        return {"coh": L_coh, "sep": L_sep, "ali": L_ali, "boids": loss}
