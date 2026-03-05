from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TemporalGRUAdapter(nn.Module):
    """Apply a GRU across time to EV embeddings (B,T,D) -> (B,D)."""

    def __init__(self, dim: int = 256, hidden: int | None = None) -> None:
        super().__init__()
        h = int(hidden or dim)
        self.gru = nn.GRU(input_size=dim, hidden_size=h, num_layers=1, batch_first=True)
        self.proj = nn.Linear(h, dim) if h != dim else nn.Identity()

    def forward(self, ev_seq: torch.Tensor) -> torch.Tensor:
        # ev_seq: (B,T,D)
        out, h = self.gru(ev_seq)  # h: (1,B,H)
        return self.proj(h[0])
