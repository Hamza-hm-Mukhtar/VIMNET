from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Canonical 8-sector layout used by the paper.
SECTORS_8 = ["LP", "P", "RP", "LF", "F", "RF", "L", "R"]

# 6-sector (cardinal + diagonals merged)
SECTORS_6 = ["P", "F", "L", "R", "DiagL", "DiagR"]

# 4-sector baseline
SECTORS_4 = ["P", "F", "L", "R"]


def sector_names(num_sectors: int) -> List[str]:
    if num_sectors == 8:
        return list(SECTORS_8)
    if num_sectors == 6:
        return list(SECTORS_6)
    if num_sectors == 4:
        return list(SECTORS_4)
    raise ValueError(f"Unsupported num_sectors={num_sectors}. Use 4/6/8.")


@dataclass(frozen=True)
class NeighborhoodConfig:
    """Neighborhood selection configuration for STSG construction."""

    mode: str = "sector"  # sector | radius | knn
    num_sectors: int = 8  # only used for sector mode
    max_neighbors: int = 8  # used for knn/radius (and as cap)
    knn_k: int = 8  # only used for knn
    radius_m: float = 50.0  # only used for radius

    # Field-of-view clip relative to target at each frame (in meters, in target-anchored coords)
    fov_lon_m: float = 20.0  # +/- longitudinal
    fov_lat_m: float = 10.0  # +/- lateral

    def num_neighbors(self) -> int:
        if self.mode == "sector":
            return int(self.num_sectors)
        return int(self.max_neighbors)

    def num_slots(self) -> int:
        return 1 + self.num_neighbors()

    @property
    def sector_list(self) -> List[str]:
        if self.mode != "sector":
            raise ValueError("sector_list is only valid when mode='sector'.")
        return sector_names(self.num_sectors)


def _within_fov(rel_xy: np.ndarray, fov_lon: float, fov_lat: float) -> np.ndarray:
    x = rel_xy[:, 0]
    y = rel_xy[:, 1]
    return (np.abs(x) <= fov_lon) & (np.abs(y) <= fov_lat)


def _angle_to_sector8(angle_rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(angle_rad)
    out = np.full_like(deg, fill_value=-1, dtype=np.int64)

    out[(deg > 157.5) | (deg <= -157.5)] = 4  # F
    out[(deg > -157.5) & (deg <= -112.5)] = 5  # RF
    out[(deg > -112.5) & (deg <= -67.5)] = 7  # R
    out[(deg > -67.5) & (deg <= -22.5)] = 2  # RP
    out[(deg > -22.5) & (deg <= 22.5)] = 1  # P
    out[(deg > 22.5) & (deg <= 67.5)] = 0  # LP
    out[(deg > 67.5) & (deg <= 112.5)] = 6  # L
    out[(deg > 112.5) & (deg <= 157.5)] = 3  # LF
    return out


def _sector8_to_sector6(sec8: np.ndarray) -> np.ndarray:
    out = np.full_like(sec8, fill_value=-1, dtype=np.int64)
    out[(sec8 == 1)] = 0  # P
    out[(sec8 == 4)] = 1  # F
    out[(sec8 == 6)] = 2  # L
    out[(sec8 == 7)] = 3  # R
    out[(sec8 == 0) | (sec8 == 3)] = 4  # DiagL
    out[(sec8 == 2) | (sec8 == 5)] = 5  # DiagR
    return out


def _sector8_to_sector4(sec8: np.ndarray) -> np.ndarray:
    out = np.full_like(sec8, fill_value=-1, dtype=np.int64)
    out[(sec8 == 0) | (sec8 == 1) | (sec8 == 2)] = 0  # P
    out[(sec8 == 3) | (sec8 == 4) | (sec8 == 5)] = 1  # F
    out[(sec8 == 6)] = 2  # L
    out[(sec8 == 7)] = 3  # R
    return out


def select_sectorized_neighbors(rel_xy: np.ndarray, cfg: NeighborhoodConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Select at most one neighbor per sector."""
    assert cfg.mode == "sector"
    S = cfg.num_neighbors()

    in_fov = _within_fov(rel_xy, cfg.fov_lon_m, cfg.fov_lat_m)
    rel_xy_f = rel_xy[in_fov]
    if rel_xy_f.shape[0] == 0:
        return np.full((S,), -1, dtype=np.int64), np.full((S,), np.inf, dtype=np.float32)

    ang = np.arctan2(rel_xy_f[:, 1], rel_xy_f[:, 0])
    sec8 = _angle_to_sector8(ang)

    if cfg.num_sectors == 8:
        sec = sec8
    elif cfg.num_sectors == 6:
        sec = _sector8_to_sector6(sec8)
    elif cfg.num_sectors == 4:
        sec = _sector8_to_sector4(sec8)
    else:
        raise ValueError(f"Unsupported num_sectors={cfg.num_sectors}")

    dist = np.linalg.norm(rel_xy_f, axis=1)

    sel_local = np.full((S,), -1, dtype=np.int64)
    sel_dist = np.full((S,), np.inf, dtype=np.float32)
    for i in range(rel_xy_f.shape[0]):
        s = int(sec[i])
        if s < 0:
            continue
        if dist[i] < sel_dist[s]:
            sel_dist[s] = float(dist[i])
            sel_local[s] = i

    orig_idx = np.flatnonzero(in_fov)
    sel_idx = np.full((S,), -1, dtype=np.int64)
    for s in range(S):
        if sel_local[s] >= 0:
            sel_idx[s] = int(orig_idx[sel_local[s]])

    return sel_idx, sel_dist


def select_knn_neighbors(rel_xy: np.ndarray, cfg: NeighborhoodConfig) -> Tuple[np.ndarray, np.ndarray]:
    assert cfg.mode == "knn"
    K = int(cfg.max_neighbors)
    in_fov = _within_fov(rel_xy, cfg.fov_lon_m, cfg.fov_lat_m)
    rel_xy_f = rel_xy[in_fov]
    if rel_xy_f.shape[0] == 0:
        return np.full((K,), -1, dtype=np.int64), np.full((K,), np.inf, dtype=np.float32)
    dist = np.linalg.norm(rel_xy_f, axis=1)
    order = np.argsort(dist)
    k = min(int(cfg.knn_k), order.shape[0], K)
    chosen_local = order[:k]
    chosen_dist = dist[chosen_local]
    orig_idx = np.flatnonzero(in_fov)
    chosen_idx = orig_idx[chosen_local]

    out_idx = np.full((K,), -1, dtype=np.int64)
    out_dist = np.full((K,), np.inf, dtype=np.float32)
    out_idx[:k] = chosen_idx
    out_dist[:k] = chosen_dist
    return out_idx, out_dist


def select_radius_neighbors(rel_xy: np.ndarray, cfg: NeighborhoodConfig) -> Tuple[np.ndarray, np.ndarray]:
    assert cfg.mode == "radius"
    K = int(cfg.max_neighbors)
    in_fov = _within_fov(rel_xy, cfg.fov_lon_m, cfg.fov_lat_m)
    rel_xy_f = rel_xy[in_fov]
    if rel_xy_f.shape[0] == 0:
        return np.full((K,), -1, dtype=np.int64), np.full((K,), np.inf, dtype=np.float32)
    dist = np.linalg.norm(rel_xy_f, axis=1)
    keep = dist <= float(cfg.radius_m)
    if not np.any(keep):
        return np.full((K,), -1, dtype=np.int64), np.full((K,), np.inf, dtype=np.float32)
    dist2 = dist[keep]
    order = np.argsort(dist2)
    k = min(order.shape[0], K)
    chosen_local = np.flatnonzero(keep)[order[:k]]
    chosen_dist = dist[chosen_local]
    orig_idx = np.flatnonzero(in_fov)
    chosen_idx = orig_idx[chosen_local]

    out_idx = np.full((K,), -1, dtype=np.int64)
    out_dist = np.full((K,), np.inf, dtype=np.float32)
    out_idx[:k] = chosen_idx
    out_dist[:k] = chosen_dist[:k]
    return out_idx, out_dist


def compute_inv_distance_bias(dist: np.ndarray, eps: float = 1e-3, clip: float = 2.5) -> np.ndarray:
    w = 1.0 / (dist + eps)
    w = np.minimum(w, clip)
    w = np.where(np.isfinite(dist), w, 0.0)
    return w.astype(np.float32)
