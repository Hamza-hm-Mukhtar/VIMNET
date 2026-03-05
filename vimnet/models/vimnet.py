from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import AttentionConfig, TransformerEncoderLayer
from .positional_encoding import sinusoidal_time_embedding


@dataclass
class VIMNETConfig:
    num_slots: int = 9
    dim: int = 256
    depth: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    ffn_mult: int = 4
    stochastic_depth: float = 0.05

    # ablation toggles
    enable_spatial: bool = True
    enable_temporal: bool = True
    use_distance_bias: bool = True
    use_type_bias: bool = True
    use_actpas_bias: bool = True
    fixed_spatial_weights: bool = False  # VIMNET-FixedWeights

    use_slot_type_embeddings: bool = True  # VIMNET-NoTypes ablation
    use_role_embeddings: bool = True       # VIMNET-NoTypes ablation


class VIMNETEncoder(nn.Module):
    def __init__(self, cfg: VIMNETConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Sequential(
            nn.Linear(2, cfg.dim),
            nn.GELU(),
            nn.LayerNorm(cfg.dim),
            nn.Dropout(cfg.dropout),
        )

        # Slot / role embeddings
        self.slot_embed = nn.Embedding(cfg.num_slots, cfg.dim)
        self.role_embed = nn.Embedding(2, cfg.dim)  # 0=EV, 1=neighbor

        attn_cfg = AttentionConfig(
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            enable_spatial=cfg.enable_spatial,
            enable_temporal=cfg.enable_temporal,
            use_distance_bias=cfg.use_distance_bias,
            use_type_bias=cfg.use_type_bias,
            use_actpas_bias=cfg.use_actpas_bias,
            fixed_spatial_weights=cfg.fixed_spatial_weights,
        )

        # Stochastic depth schedule
        dpr = [cfg.stochastic_depth * (i / max(1, cfg.depth - 1)) for i in range(cfg.depth)]
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_slots=cfg.num_slots,
                    cfg=attn_cfg,
                    ffn_mult=cfg.ffn_mult,
                    drop_path_prob=dpr[i],
                )
                for i in range(cfg.depth)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.dim)

    def forward(self, obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            obs_xy: (B,T,N,2) float
            obs_mask: (B,T,N) bool
        Returns:
            enc: (B,T,N,D)
        """
        B, T, N, _ = obs_xy.shape
        assert N == self.cfg.num_slots, f"Expected num_slots={self.cfg.num_slots}, got {N}"

        x = self.embed(obs_xy)  # (B,T,N,D)

        # Time embedding
        pe = sinusoidal_time_embedding(T, self.cfg.dim, device=obs_xy.device)  # (T,D)
        x = x + pe.view(1, T, 1, self.cfg.dim)

        if self.cfg.use_slot_type_embeddings:
            slot_ids = torch.arange(N, device=obs_xy.device, dtype=torch.long)
            x = x + self.slot_embed(slot_ids).view(1, 1, N, self.cfg.dim)

        if self.cfg.use_role_embeddings:
            role_ids = torch.zeros((N,), device=obs_xy.device, dtype=torch.long)
            role_ids[1:] = 1
            x = x + self.role_embed(role_ids).view(1, 1, N, self.cfg.dim)

        # mask invalid tokens to zero
        x = x * obs_mask.unsqueeze(-1).to(x.dtype)

        for layer in self.layers:
            x = layer(x, valid_mask=obs_mask, obs_xy=obs_xy)

        x = self.ln_f(x)
        x = x * obs_mask.unsqueeze(-1).to(x.dtype)
        return x


class VIMNET(nn.Module):
    """Full model wrapper with encoder and optional heads."""

    def __init__(self, encoder: VIMNETEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, obs_xy: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_xy, obs_mask)
