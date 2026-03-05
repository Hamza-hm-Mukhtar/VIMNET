from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .shards import ShardWriter
from .stsg import NeighborhoodConfig, select_knn_neighbors, select_radius_neighbors, select_sectorized_neighbors
from .transforms import ego_normalize, resample_xy, savgol_smooth_xy, compute_kinematics


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class PNeumaPreprocessConfig:
    raw_dir: Path
    out_dir: Path

    hz: float = 10.0
    obs_sec: float = 2.0
    pred_sec: float = 5.0
    stride_sec: float = 0.5

    sg_window: int = 9
    sg_poly: int = 2

    max_speed_mps: float = 55.0
    max_lat_acc_mps2: float = 8.0

    lc_threshold_m: float = 1.75

    neighborhood: NeighborhoodConfig = NeighborhoodConfig(mode="sector", num_sectors=8, fov_lon_m=20.0, fov_lat_m=10.0)

    shard_size: int = 2048
    compress: bool = False

    include_types: Optional[List[str]] = None  # e.g., ["car", "truck", ...] depending on dataset encoding
    coords: str = "auto"  # "auto" | "latlon" | "metric"



def parse_pneuma_track_row(row: Sequence) -> Optional[Dict[str, object]]:
    """Parse a single pNEUMA 'wide' row into arrays.

    Expected format (based on public conversion scripts):
        [track_id, type, traveled_distance, avg_speed, (lat, lon, speed, lon_acc, lat_acc, time)*]

    Returns:
        dict with keys: track_id, obj_type, times_s, xy
    """
    if len(row) < 10:
        return None

    track_id = row[0]
    obj_type = row[1]

    # Convert trailing columns to numeric (coerce errors to NaN)
    rest = pd.to_numeric(pd.Series(list(row[4:])), errors="coerce").to_numpy(dtype=np.float32)
    if rest.size < 6:
        return None
    n = (rest.size // 6) * 6
    rest = rest[:n].reshape(-1, 6)

    x = rest[:, 0]
    y = rest[:, 1]
    t = rest[:, 5]

    valid = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return None
    t = t[valid]
    x = x[valid]
    y = y[valid]

    # Ensure strictly increasing times (drop duplicates)
    order = np.argsort(t)
    t = t[order]
    xy = np.stack([x[order], y[order]], axis=1).astype(np.float32)

    # Remove duplicate timestamps
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 1e-6
    t = t[keep]
    xy = xy[keep]

    if t.shape[0] < 5:
        return None

    return {
        "track_id": int(track_id),
        "obj_type": str(obj_type),
        "times_s": t.astype(np.float32),
        "xy": xy.astype(np.float32),
    }



def _looks_like_latlon(xy: np.ndarray) -> bool:
    """Heuristic check for lat/lon degree coordinates.

    pNEUMA CSVs often store *latitude/longitude* (degrees) as the first two columns of each 6-field time group.
    If we treat those as meters, FoV and kinematics become incorrect.
    """
    if xy.size == 0:
        return False
    x = xy[:, 0]
    y = xy[:, 1]
    if not (np.isfinite(x).any() and np.isfinite(y).any()):
        return False
    # Basic lat/lon bounds
    if float(np.nanmax(np.abs(x))) > 90.0 or float(np.nanmax(np.abs(y))) > 180.0:
        return False
    # Typical urban tile spans far less than 1 degree
    if float(np.nanmax(x) - np.nanmin(x)) > 1.0:
        return False
    if float(np.nanmax(y) - np.nanmin(y)) > 1.0:
        return False
    # Avoid mis-detecting small metric coordinates near (0, 0)
    if abs(float(np.nanmean(x))) < 1.0 and abs(float(np.nanmean(y))) < 1.0:
        return False
    return True


def latlon_to_xy_m(latlon: np.ndarray, origin_latlon: Tuple[float, float]) -> np.ndarray:
    """Project lat/lon degrees to local meters with an equirectangular approximation.

    This is sufficient for small areas (e.g., a single pNEUMA tile) and keeps preprocessing light.

    Args:
        latlon: (T, 2) with columns [lat_deg, lon_deg]
        origin_latlon: (lat0_deg, lon0_deg) used as local origin
    Returns:
        xy_m: (T, 2) with columns [x_east_m, y_north_m]
    """
    lat0_deg, lon0_deg = origin_latlon
    R = 6378137.0
    lat = np.deg2rad(latlon[:, 0].astype(np.float64))
    lon = np.deg2rad(latlon[:, 1].astype(np.float64))
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))
    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return np.stack([x, y], axis=1).astype(np.float32)
def _infer_lc_label_from_fut(fut_xy_ego: np.ndarray, thresh: float) -> int:
    y_end = float(fut_xy_ego[-1, 1])
    if y_end > thresh:
        return 0  # left
    if y_end < -thresh:
        return 2  # right
    return 1  # keep


def preprocess_pneuma(
    cfg: PNeumaPreprocessConfig,
    split_files: Dict[str, List[str]],
) -> None:
    """Preprocess pNEUMA CSV tiles into sharded NPZs."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / cfg.hz
    H_obs = int(round(cfg.obs_sec * cfg.hz))
    H_pred = int(round(cfg.pred_sec * cfg.hz))
    stride_frames = int(round(cfg.stride_sec * cfg.hz))

    for split, files in split_files.items():
        out_split = cfg.out_dir / "pneuma" / split
        writer = ShardWriter(out_split, shard_size=cfg.shard_size, compress=cfg.compress)
        num_written = 0

        for fname in tqdm(files, desc=f"pNEUMA[{split}] tiles"):
            tile_path = cfg.raw_dir / fname
            if not tile_path.exists():
                raise FileNotFoundError(f"Missing {tile_path}")

            # First pass: parse and resample each track into (frame -> xy) dict.
            track_dict: Dict[int, Dict[str, object]] = {}
            # frame -> list of (id, xy_world)
            frame_to_ids: Dict[int, List[int]] = {}
            frame_to_xy: Dict[int, List[np.ndarray]] = {}

            coords = str(cfg.coords).lower()
            if coords not in {"auto", "latlon", "metric"}:
                raise ValueError(f"cfg.coords must be one of auto|latlon|metric, got: {cfg.coords}")
            coords_effective = coords  # resolved from auto on the first parsed track
            origin_latlon: Optional[Tuple[float, float]] = None

            for chunk in pd.read_csv(tile_path, header=None, chunksize=2048):
                for row in chunk.itertuples(index=False, name=None):
                    parsed = parse_pneuma_track_row(row)
                    if parsed is None:
                        continue
                    if cfg.include_types is not None and parsed["obj_type"] not in cfg.include_types:
                        continue
                    tid = int(parsed["track_id"])
                    t = parsed["times_s"]
                    xy = parsed["xy"]

                    # Coordinate normalization: pNEUMA CSVs often store latitude/longitude (degrees).
                    # We project to local meters per tile so that FoV and kinematics are in SI units.
                    if coords_effective == "auto":
                        coords_effective = "latlon" if _looks_like_latlon(xy) else "metric"
                    if coords_effective == "latlon":
                        if origin_latlon is None:
                            origin_latlon = (float(xy[0, 0]), float(xy[0, 1]))
                        xy = latlon_to_xy_m(xy, origin_latlon)

                    # Resample to 10 Hz and smooth
                    t_r, xy_r = resample_xy(t, xy, dt=dt)
                    xy_r = savgol_smooth_xy(xy_r, window=cfg.sg_window, polyorder=cfg.sg_poly)

                    kin = compute_kinematics(t_r, xy_r)
                    speed = kin["speed"]
                    lat_acc = kin["lat_acc"]
                    if np.nanmax(speed) > cfg.max_speed_mps or np.nanmax(lat_acc) > cfg.max_lat_acc_mps2:
                        continue

                    frames = np.array([_round_half_up(float(tt) * cfg.hz) for tt in t_r], dtype=np.int32)

                    track_dict[tid] = {
                        "frames": frames,
                        "t": t_r,
                        "xy": xy_r,
                        "heading": kin["heading"],
                        "speed": speed,
                    }

                    for fr, pos in zip(frames, xy_r):
                        frame_to_ids.setdefault(int(fr), []).append(tid)
                        frame_to_xy.setdefault(int(fr), []).append(pos)

            if not track_dict:
                continue

            # Convert frame_to_* lists to numpy arrays for fast lookup
            frame_to_ids_np: Dict[int, np.ndarray] = {}
            frame_to_xy_np: Dict[int, np.ndarray] = {}
            for fr in frame_to_ids.keys():
                frame_to_ids_np[fr] = np.array(frame_to_ids[fr], dtype=np.int64)
                frame_to_xy_np[fr] = np.stack(frame_to_xy[fr], axis=0).astype(np.float32)

            # Second pass: generate samples per track
            for tid, td in track_dict.items():
                frames = td["frames"].astype(np.int32)
                xy = td["xy"].astype(np.float32)
                heading = td["heading"].astype(np.float32)

                f_min = int(frames[0])
                f_max = int(frames[-1])

                # We assume frames are contiguous after resampling (may have gaps if rounding),
                # so we build a mapping frame -> index for this track.
                frame_to_idx = {int(fr): i for i, fr in enumerate(frames)}

                start_f0 = f_min + (H_obs - 1)
                end_f0 = f_max - H_pred
                if end_f0 <= start_f0:
                    continue

                for f0 in range(start_f0, end_f0 + 1, stride_frames):
                    # Ensure required frames exist in this track
                    need_obs = [f0 - (H_obs - 1 - i) for i in range(H_obs)]
                    need_fut = [f0 + (i + 1) for i in range(H_pred)]
                    if any(fr not in frame_to_idx for fr in need_obs):
                        continue
                    if any(fr not in frame_to_idx for fr in need_fut):
                        continue

                    idx_obs = np.array([frame_to_idx[fr] for fr in need_obs], dtype=np.int32)
                    idx_fut = np.array([frame_to_idx[fr] for fr in need_fut], dtype=np.int32)

                    ev_obs_world = xy[idx_obs]
                    ev_fut_world = xy[idx_fut]

                    anchor_xy = ev_obs_world[-1].copy()
                    anchor_yaw = float(heading[idx_obs[-1]])

                    # Select neighbors geometrically at each obs frame
                    S = cfg.neighborhood.num_neighbors()
                    neigh_world = np.zeros((H_obs, S, 2), dtype=np.float32)
                    neigh_valid = np.zeros((H_obs, S), dtype=bool)

                    for t_i, fr in enumerate(need_obs):
                        ids = frame_to_ids_np.get(int(fr))
                        pos = frame_to_xy_np.get(int(fr))
                        if ids is None or pos is None:
                            continue
                        keep = ids != tid
                        ids2 = ids[keep]
                        pos2 = pos[keep]
                        if pos2.shape[0] == 0:
                            continue

                        # ego transform all candidates and EV at this time
                        pos2_ego = ego_normalize(pos2, anchor_xy, anchor_yaw)
                        ev_ego = ego_normalize(ev_obs_world[t_i:t_i+1], anchor_xy, anchor_yaw)[0]
                        rel = pos2_ego - ev_ego

                        if cfg.neighborhood.mode == "sector":
                            sel_idx, _ = select_sectorized_neighbors(rel, cfg.neighborhood)
                        elif cfg.neighborhood.mode == "knn":
                            sel_idx, _ = select_knn_neighbors(rel, cfg.neighborhood)
                        elif cfg.neighborhood.mode == "radius":
                            sel_idx, _ = select_radius_neighbors(rel, cfg.neighborhood)
                        else:
                            raise ValueError(f"Unknown neighborhood mode {cfg.neighborhood.mode}")

                        for s in range(S):
                            j = int(sel_idx[s])
                            if j >= 0:
                                neigh_world[t_i, s] = pos2[j]
                                neigh_valid[t_i, s] = True

                    # Ego normalize full obs slots
                    ev_obs_ego = ego_normalize(ev_obs_world, anchor_xy, anchor_yaw)
                    neigh_ego = ego_normalize(neigh_world.reshape(-1, 2), anchor_xy, anchor_yaw).reshape(neigh_world.shape)

                    N = 1 + S
                    obs = np.zeros((H_obs, N, 2), dtype=np.float32)
                    obs_mask = np.zeros((H_obs, N), dtype=bool)
                    obs[:, 0] = ev_obs_ego
                    obs_mask[:, 0] = True
                    obs[:, 1:] = neigh_ego
                    obs_mask[:, 1:] = neigh_valid

                    fut = ego_normalize(ev_fut_world, anchor_xy, anchor_yaw).astype(np.float32)
                    lc = _infer_lc_label_from_fut(fut, cfg.lc_threshold_m)

                    writer.add(
                        {
                            "obs": obs,
                            "obs_mask": obs_mask,
                            "fut": fut,
                            "lc_label": np.array(lc, dtype=np.int64),
                            "meta_track_id": np.array(int(tid), dtype=np.int64),
                            "meta_rec_id": np.array(-1, dtype=np.int64),  # tile not numeric
                            "meta_t0": np.array(int(f0), dtype=np.int64),
                        }
                    )
                    num_written += 1

        writer.close()
        print(f"[pNEUMA] Wrote {num_written} samples to {out_split}")
