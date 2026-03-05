from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    rand = rand.floor()
    return x.div(keep_prob) * rand


@dataclass
class AttentionConfig:
    dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1

    enable_spatial: bool = True
    enable_temporal: bool = True

    use_distance_bias: bool = True
    use_type_bias: bool = True
    use_actpas_bias: bool = True

    fixed_spatial_weights: bool = False  # VIMNET-FixedWeights ablation

    invdist_eps: float = 1e-3
    invdist_clip: float = 2.5


def build_block_sparse_mask(T: int, N: int, enable_spatial: bool, enable_temporal: bool, device: torch.device) -> torch.Tensor:
    """Build additive attention mask (1,1,L,L) with 0 for allowed and -inf for disallowed.

    Allowed edges:
      - temporal: within each slot, causal (t attends to <=t)
      - spatial: within each time t, EV<->neighbor only (star)
    """
    L = T * N
    mask = torch.full((L, L), fill_value=float("-inf"), device=device)
    # Always allow self
    mask.fill_diagonal_(0.0)

    # Temporal causal per slot
    if enable_temporal:
        for s in range(N):
            for t_q in range(T):
                q = t_q * N + s
                # keys from 0..t_q
                k0 = 0 * N + s
                k1 = (t_q + 1) * N + s
                mask[q, k0:k1:N] = 0.0  # step by N to select same slot
        # Note: above uses slice with step N; works if contiguous.

    # Spatial star within frame
    if enable_spatial and N > 1:
        for t in range(T):
            base = t * N
            ev = base + 0
            # EV attends to all slots within frame
            mask[ev, base : base + N] = 0.0
            # Each neighbor attends to EV
            for j in range(1, N):
                mask[base + j, ev] = 0.0

    return mask.view(1, 1, L, L)


class BlockSparseMHSA(nn.Module):
    """Multi-head self-attention with a fixed block-sparse mask and optional edge biases."""

    def __init__(self, num_slots: int, cfg: AttentionConfig) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.cfg = cfg
        assert cfg.dim % cfg.num_heads == 0
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=True)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

        # Learnable type/direction bias for spatial EV<->neighbor edges.
        # Shape: (num_heads, 2, num_slots) where 2 is direction:
        #   dir=0: EV(query)->neighbor(key)
        #   dir=1: neighbor(query)->EV(key)
        self.edge_type_bias = nn.Parameter(torch.zeros(cfg.num_heads, 2, self.num_slots))

        # Cache masks by (T, enable_spatial, enable_temporal)
        self._mask_cache: Dict[Tuple[int, bool, bool], torch.Tensor] = {}

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, self.cfg.enable_spatial, self.cfg.enable_temporal)
        m = self._mask_cache.get(key)
        if m is None or m.device != device:
            m = build_block_sparse_mask(T, self.num_slots, self.cfg.enable_spatial, self.cfg.enable_temporal, device)
            self._mask_cache[key] = m
        return m

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, obs_xy: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (B,T,N,D)
            valid_mask: (B,T,N) bool
            obs_xy: (B,T,N,2) positions in ego frame (used for distance bias)
        """
        B, T, N, D = x.shape
        assert N == self.num_slots
        L = T * N

        x_flat = x.reshape(B, L, D)

        qkv = self.qkv(x_flat)  # (B,L,3D)
        qkv = qkv.view(B, L, 3, self.cfg.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,L,hd)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,L,L)

        # Apply block-sparse mask
        attn = attn + self._get_mask(T, attn.device)

        # Key padding mask for invalid tokens: disallow attending to invalid keys
        valid_flat = valid_mask.reshape(B, L)  # bool
        # shape to (B,1,1,L)
        key_mask = (~valid_flat).view(B, 1, 1, L)
        attn = attn.masked_fill(key_mask, float("-inf"))

        # Distance bias + type bias for spatial EV<->neighbor edges
        if self.cfg.enable_spatial and N > 1:
            # compute invdist per (B,T,N)
            if self.cfg.use_distance_bias or self.cfg.fixed_spatial_weights:
                ev = obs_xy[:, :, 0:1, :]  # (B,T,1,2)
                rel = obs_xy - ev
                dist = torch.linalg.norm(rel, dim=-1)  # (B,T,N)
                inv = 1.0 / (dist + self.cfg.invdist_eps)
                inv = torch.clamp(inv, max=self.cfg.invdist_clip)
                inv = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
            else:
                inv = None

            # Build bias tensor only for needed entries (small loops)
            if (self.cfg.use_distance_bias or self.cfg.fixed_spatial_weights) and inv is not None:
                # We'll add bias as (B,1,L,L) for broadcast across heads.
                bias = torch.zeros((B, L, L), device=attn.device, dtype=attn.dtype)
                for t in range(T):
                    base = t * N
                    ev_idx = base + 0
                    for j in range(1, N):
                        key = base + j
                        b = inv[:, t, j].to(attn.dtype)
                        bias[:, ev_idx, key] = b
                        bias[:, key, ev_idx] = b
                # Apply bias
                if self.cfg.fixed_spatial_weights:
                    # overwrite spatial logits with distance-only bias (no learned qk).
                    for t in range(T):
                        base = t * N
                        ev_idx = base + 0
                        for j in range(1, N):
                            key = base + j
                            b = inv[:, t, j].to(attn.dtype)
                            attn[:, :, ev_idx, key] = b.view(B, 1)
                            attn[:, :, key, ev_idx] = b.view(B, 1)
                else:
                    attn = attn + bias.view(B, 1, L, L)

            if self.cfg.use_type_bias:
                # Learnable per-head bias; apply only on spatial EV<->neighbor edges
                # Build (1,H,L,L) and broadcast over batch
                tb = torch.zeros((self.cfg.num_heads, L, L), device=attn.device, dtype=attn.dtype)
                # direction handling
                for t in range(T):
                    base = t * N
                    ev_idx = base + 0
                    for j in range(1, N):
                        neigh_idx = base + j
                        # EV(query)->neighbor(key): dir=0, sector slot=j
                        tb[:, ev_idx, neigh_idx] = self.edge_type_bias[:, 0 if self.cfg.use_actpas_bias else 0, j]
                        # neighbor(query)->EV(key): dir=1 if act/pass enabled else 0
                        dir_idx = 1 if self.cfg.use_actpas_bias else 0
                        tb[:, neigh_idx, ev_idx] = self.edge_type_bias[:, dir_idx, j]
                attn = attn + tb.view(1, self.cfg.num_heads, L, L)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B,H,L,hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Zero out invalid query outputs to keep NULL tokens inert
        out = out * valid_flat.to(out.dtype).unsqueeze(-1)

        return out.view(B, T, N, D)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_slots: int, cfg: AttentionConfig, ffn_mult: int = 4, drop_path_prob: float = 0.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.drop_path_prob = float(drop_path_prob)
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = BlockSparseMHSA(num_slots=num_slots, cfg=cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, ffn_mult * cfg.dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_mult * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, obs_xy: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.attn(self.ln1(x), valid_mask=valid_mask, obs_xy=obs_xy)
        x = x + drop_path(h, self.drop_path_prob, self.training)
        # Pre-norm FFN
        h2 = self.ffn(self.ln2(x))
        x = x + drop_path(h2, self.drop_path_prob, self.training)
        return x
