from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class AugmentConfig:
    enabled: bool = True
    gaussian_jitter_std: float = 0.05  # meters
    token_dropout_p: float = 0.05      # drop neighbor tokens per step
    random_fov_shrink: bool = True
    fov_lon_min_m: float = 18.0
    fov_lon_max_m: float = 22.0
    fov_lat_m: float = 10.0


class ShardedNPZDataset(Dataset):
    """Dataset reading fixed-size sequences stored in NPZ shards.

    Each shard is expected to contain the following arrays:
        obs: (B, T_obs, N, 2)
        obs_mask: (B, T_obs, N)  bool
        fut: (B, T_pred, 2)      (only for finetune)
        lc_label: (B,)           (only for finetune)
        y_next: (B, 2)           (only for pretrain)
    """

    def __init__(
        self,
        shard_dir: Path,
        task: str = "finetune",
        augment: Optional[AugmentConfig] = None,
        obs_len: int = 20,
        pred_len: int = 50,
        pretrain_ctx: int = 12,
        training: bool = True,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.task = task
        self.augment = augment or AugmentConfig(enabled=training)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pretrain_ctx = pretrain_ctx
        self.training = training
        self.rng = np.random.default_rng(seed)

        self.shards: List[Dict] = []
        for p in sorted(self.shard_dir.glob("*.npz")):
            meta = np.load(p, allow_pickle=False)
            n = int(meta["obs"].shape[0])
            self.shards.append({"path": p.name, "size": n})
        self._cum = np.cumsum([s["size"] for s in self.shards]).tolist()

    def __len__(self) -> int:
        return int(self._cum[-1]) if self._cum else 0

    def _locate(self, idx: int) -> Tuple[int, int]:
        # Return (shard_index, inner_index)
        lo, hi = 0, len(self._cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_i = lo
        prev = self._cum[shard_i - 1] if shard_i > 0 else 0
        inner = idx - prev
        return shard_i, inner

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_i, inner = self._locate(idx)
        shard_path = self.shard_dir / self.shards[shard_i]["path"]
        data = np.load(shard_path, allow_pickle=False)

        obs = torch.from_numpy(data["obs"][inner]).float()  # (T,N,2)
        obs_mask = torch.from_numpy(data["obs_mask"][inner]).bool()  # (T,N)

        sample: Dict[str, torch.Tensor] = {"obs": obs, "obs_mask": obs_mask}

        if self.task == "finetune":
            fut = torch.from_numpy(data["fut"][inner]).float()
            lc = torch.from_numpy(data["lc_label"][inner]).long()
            sample.update({"fut": fut, "lc_label": lc})
        elif self.task == "pretrain":
            y = torch.from_numpy(data["y_next"][inner]).float()
            sample.update({"y_next": y})
        else:
            raise ValueError(f"Unknown task={self.task}")

        if self.augment.enabled:
            obs, obs_mask = self._augment(obs, obs_mask)
            sample["obs"] = obs
            sample["obs_mask"] = obs_mask

        return sample

    def _augment(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gaussian jitter
        if self.augment.gaussian_jitter_std > 0:
            obs = obs + torch.randn_like(obs) * float(self.augment.gaussian_jitter_std)

        # Random FoV shrink (masks out tokens outside a randomly sampled longitudinal range)
        if self.augment.random_fov_shrink:
            lon = float(self.rng.uniform(self.augment.fov_lon_min_m, self.augment.fov_lon_max_m))
            lat = float(self.augment.fov_lat_m)
            inside = (obs[..., 0].abs() <= lon) & (obs[..., 1].abs() <= lat)
            # always keep EV slot (index 0)
            inside[:, 0] = True
            mask = mask & inside
            obs = obs * mask.unsqueeze(-1).float()

        # Token dropout (drop neighbors, keep EV)
        if self.augment.token_dropout_p > 0:
            drop = torch.rand(mask.shape) < float(self.augment.token_dropout_p)
            drop[:, 0] = False
            mask = mask & ~drop
            obs = obs * mask.unsqueeze(-1).float()

        return obs, mask


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
