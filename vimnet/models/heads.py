from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainNextStepHead(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )

    def forward(self, ev_token: torch.Tensor) -> torch.Tensor:
        return self.mlp(ev_token)


@dataclass
class ScheduledSamplingConfig:
    max_free_running_ratio: float = 0.2
    warmup_frac: float = 0.3  # of total steps


class GRUTrajectoryHead(nn.Module):
    """Autoregressive GRU decoder that outputs future deltas and integrates to positions."""

    def __init__(self, dim: int = 256, hidden: int = 256, horizon: int = 50) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.init_h = nn.Linear(dim, hidden)
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(
        self,
        ev_context: torch.Tensor,  # (B,dim)
        teacher_deltas: Optional[torch.Tensor] = None,  # (B,T,2)
        free_running_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            pred_deltas: (B,T,2)
            pred_pos: (B,T,2) cumulative sum from origin
        """
        B, D = ev_context.shape
        T = self.horizon
        h0 = torch.tanh(self.init_h(ev_context)).unsqueeze(0)  # (1,B,H)
        # start input delta = 0
        prev = torch.zeros((B, 1, 2), device=ev_context.device, dtype=ev_context.dtype)
        outs = []
        h = h0
        for t in range(T):
            out, h = self.gru(prev, h)  # out: (B,1,H)
            delta = self.out(out[:, 0])  # (B,2)
            outs.append(delta)
            if teacher_deltas is not None:
                use_free = torch.rand((B,), device=ev_context.device) < free_running_ratio
                teacher = teacher_deltas[:, t]
                prev_delta = torch.where(use_free.unsqueeze(-1), delta, teacher)
            else:
                prev_delta = delta
            prev = prev_delta.unsqueeze(1)
        pred_deltas = torch.stack(outs, dim=1)  # (B,T,2)
        pred_pos = torch.cumsum(pred_deltas, dim=1)
        return pred_deltas, pred_pos


class LaneChangeHead(nn.Module):
    def __init__(self, dim: int = 256, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, ev_context: torch.Tensor) -> torch.Tensor:
        return self.fc(ev_context)


def label_smoothed_ce(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross entropy with optional label smoothing and class weights.

    Args:
        logits: (B,C)
        target: (B,) int
        smoothing: in [0,1)
        class_weights: (C,) float tensor on same device
    """
    C = logits.shape[-1]
    logp = F.log_softmax(logits, dim=-1)
    if smoothing > 0.0:
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(smoothing / (C - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
        nll = -(true_dist * logp).sum(dim=-1)
    else:
        nll = F.nll_loss(logp, target, reduction="none")

    if class_weights is not None:
        w = class_weights[target]  # (B,)
        nll = nll * w

    return nll.mean()
