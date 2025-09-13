from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class BoidsConfig:
    lambda_c: float = 0.1  # Cohesion
    lambda_s: float = 0.05  # Separation
    lambda_a: float = 0.01  # Alignment
    k_nn: int = 8
    sep_tau: float = 1.5
    warmup_frac: float = 0.0  # 0..1 of total steps


class BoidsRegularizer:
    """
    Simplified Boids-like regularizer.
    Works on features shaped (B, T, H) or (N, H).
    Encourages variance (cohesion proxy) and low pairwise similarity (separation proxy).
    """

    def __init__(self, cfg: BoidsConfig) -> None:
        self.cfg = cfg
        self._scale: float = 1.0

    def step(self, global_step: int, total_steps: int) -> None:
        if self.cfg.warmup_frac <= 0:
            self._scale = 1.0
            return
        warmup_steps = max(1, int(total_steps * self.cfg.warmup_frac))
        self._scale = min(1.0, global_step / warmup_steps)

    def __call__(self, feats: Tensor) -> Tensor:
        # Flatten if (B, T, H)
        if feats.dim() == 3:
            bsz: int = int(feats.size(0))
            tsz: int = int(feats.size(1))
            hid: int = int(feats.size(2))
            x: Tensor = feats.reshape(bsz * tsz, hid)
        else:
            x = feats
        # Normalize for stability
        x = F.normalize(x, dim=-1)
        # Feature variance as cohesion proxy
        mean: Tensor = x.mean(dim=0, keepdim=True)
        var: Tensor = ((x - mean) * (x - mean)).mean()
        # Pairwise cosine similarity mean as separation proxy
        sim: Tensor = x @ x.t()
        N: int = int(sim.size(0))
        if N > 1:
            eye: Tensor = torch.eye(N, device=sim.device, dtype=sim.dtype)
            sim_no_diag: Tensor = sim - eye
            sim_mean: Tensor = sim_no_diag.mean()
        else:
            sim_mean = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # Use var again as a simple alignment proxy
        one: Tensor = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        loss: Tensor = (
            torch.tensor(self.cfg.lambda_c, device=x.device, dtype=x.dtype) * var
            + torch.tensor(self.cfg.lambda_s, device=x.device, dtype=x.dtype) * (one - sim_mean)
            + torch.tensor(self.cfg.lambda_a, device=x.device, dtype=x.dtype) * var
        ) * torch.tensor(self._scale, device=x.device, dtype=x.dtype)
        return loss
        def __call__(self, feats: Tensor) -> Tensor:
            # Flatten if (B, T, H)
            if feats.dim() == 3:
                bsz: int = int(feats.size(0))
                tsz: int = int(feats.size(1))
                hid: int = int(feats.size(2))
                x: Tensor = feats.reshape(bsz * tsz, hid)
            else:
                x = feats
            # Normalize for stability
            x = F.normalize(x, dim=-1)
            # Simple proxy Boids loss using feature variance and pairwise dot-product statistics
            mean: Tensor = x.mean(dim=0, keepdim=True)
            var: Tensor = ((x - mean) * (x - mean)).mean()
            # Encourage spread across samples (lower pairwise similarity)
            sim: Tensor = (x @ x.t())
            N: int = int(sim.size(0))
            if N > 1:
                sim = sim - torch.eye(N, device=sim.device, dtype=sim.dtype)
                sim_mean: Tensor = sim.mean()
            else:
                sim_mean = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            # Combine terms to mimic cohesion (var), separation (1-sim), alignment (use var again)
            loss: Tensor = (
                self.cfg.lambda_c * var
                + self.cfg.lambda_s * (1.0 - sim_mean)
                + self.cfg.lambda_a * var
            ) * self._scale
            return loss
