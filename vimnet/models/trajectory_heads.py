from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import GRUTrajectoryHead


class MLPTrajectoryHead(nn.Module):
    """2-layer MLP that outputs the whole horizon of deltas in one shot."""

    def __init__(self, dim: int = 256, horizon: int = 50) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2 * self.horizon),
        )

    def forward(self, ev_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D = ev_context.shape
        out = self.mlp(ev_context).view(B, self.horizon, 2)
        pos = torch.cumsum(out, dim=1)
        return out, pos


class TransformerTrajectoryHead(nn.Module):
    """Lightweight transformer decoder with learned queries and causal mask."""

    def __init__(self, dim: int = 256, horizon: int = 50, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.query = nn.Parameter(torch.randn(self.horizon, dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=4 * dim, dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out = nn.Linear(dim, 2)

        # causal mask for target sequence
        self.register_buffer("tgt_mask", torch.triu(torch.ones(self.horizon, self.horizon), diagonal=1).bool(), persistent=False)

    def forward(self, ev_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # memory is a single token
        memory = ev_context.unsqueeze(1)  # (B,1,D)
        tgt = self.query.unsqueeze(0).expand(ev_context.shape[0], -1, -1)  # (B,T,D)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self.tgt_mask)
        deltas = self.out(out)
        pos = torch.cumsum(deltas, dim=1)
        return deltas, pos
