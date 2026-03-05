from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ShardSpec:
    path: str
    num_samples: int


class ShardWriter:
    """Write samples into NPZ shards + an index.json.

    This keeps preprocessing memory-friendly and makes training load fast.

    Each shard is expected to contain numpy arrays with the same first dimension
    (num_samples in that shard).

    Example keys we use in this repo:
      - obs: (B,H_obs,N,2) float32
      - obs_mask: (B,H_obs,N) bool
      - fut: (B,H_pred,2) float32
      - lc_label: (B,) int64
      - meta_track_id: (B,) int64
      - meta_rec_id: (B,) int64
      - meta_t0: (B,) int64
    """

    def __init__(self, out_dir: str | Path, shard_size: int = 4096, compress: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.shard_size = int(shard_size)
        self.compress = bool(compress)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._buf: Dict[str, List[np.ndarray]] = {}
        self._num_in_buf = 0
        self._shards: List[ShardSpec] = []
        self._shard_idx = 0

    def add(self, sample: Dict[str, np.ndarray]) -> None:
        if self._num_in_buf == 0:
            self._buf = {k: [] for k in sample.keys()}
        for k, v in sample.items():
            self._buf[k].append(v)
        self._num_in_buf += 1
        if self._num_in_buf >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if self._num_in_buf == 0:
            return
        arrays: Dict[str, np.ndarray] = {}
        for k, lst in self._buf.items():
            arrays[k] = np.stack(lst, axis=0)
        shard_name = f"shard_{self._shard_idx:05d}.npz"
        out_path = self.out_dir / shard_name
        if self.compress:
            np.savez_compressed(out_path, **arrays)
        else:
            np.savez(out_path, **arrays)
        self._shards.append(ShardSpec(path=shard_name, num_samples=self._num_in_buf))
        self._shard_idx += 1
        self._buf = {}
        self._num_in_buf = 0

    def close(self) -> None:
        self.flush()
        index = {
            "format": "vimnet_npz_shards_v1",
            "shard_size": self.shard_size,
            "compress": self.compress,
            "shards": [s.__dict__ for s in self._shards],
        }
        with (self.out_dir / "index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
