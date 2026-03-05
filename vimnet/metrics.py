from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@torch.no_grad()
def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """Compute ADE/FDE for point trajectories.

    Args:
        pred: (B,T,2)
        gt: (B,T,2)
    """
    diff = pred - gt
    dist = torch.linalg.norm(diff, dim=-1)  # (B,T)
    ade = dist.mean().item()
    fde = dist[:, -1].mean().item()
    return {"ADE": ade, "FDE": fde}


@torch.no_grad()
def horizon_metrics(pred: torch.Tensor, gt: torch.Tensor, hz: float = 10.0) -> Dict[str, float]:
    """Return ADE/FDE at 1..5 seconds given 10Hz trajectories."""
    out: Dict[str, float] = {}
    for sec in [1, 2, 3, 4, 5]:
        t = int(sec * hz)
        t = min(t, pred.shape[1])
        m = ade_fde(pred[:, :t], gt[:, :t])
        out[f"ADE@{sec}s"] = m["ADE"]
        out[f"FDE@{sec}s"] = m["FDE"]
    return out


@torch.no_grad()
def accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred_logits.argmax(dim=-1)
    return (pred == target).float().mean().item()
