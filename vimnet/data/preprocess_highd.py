from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .highd import HighDPaths, load_recording_meta, load_tracks
from .shards import ShardWriter
from .stsg import NeighborhoodConfig, select_knn_neighbors, select_radius_neighbors, select_sectorized_neighbors
from .transforms import ego_normalize, savgol_smooth_xy


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class HighDPreprocessConfig:
    raw_dir: Path
    out_dir: Path

    hz: float = 10.0
    obs_sec: float = 2.0
    pred_sec: float = 5.0
    stride_sec: float = 0.5

    # smoothing and filtering
    sg_window: int = 9
    sg_poly: int = 2
    max_speed_mps: float = 55.0
    max_lat_acc_mps2: float = 8.0

    # lane-change label from lateral displacement
    lc_threshold_m: float = 1.75

    neighborhood: NeighborhoodConfig = NeighborhoodConfig(mode="sector", num_sectors=8, fov_lon_m=20.0, fov_lat_m=10.0)

    shard_size: int = 2048
    compress: bool = False

    # If True, use the 8 neighbor-id columns provided by highD when possible (only for 8-sector mode).
    prefer_highd_neighbors: bool = True


def _neighbor_cols_ordered() -> List[str]:
    # Canonical order: [LP,P,RP,LF,F,RF,L,R]
    return [
        "leftPrecedingId",
        "precedingId",
        "rightPrecedingId",
        "leftFollowingId",
        "followingId",
        "rightFollowingId",
        "leftAlongsideId",
        "rightAlongsideId",
    ]


def _compute_anchor_yaw(row: pd.Series) -> float:
    vx = float(row.get("xVelocity", 0.0))
    vy = float(row.get("yVelocity", 0.0))
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        # fallback: yaw=0 (will still work, but less invariant)
        return 0.0
    return float(math.atan2(vy, vx))


def _infer_lc_label_from_fut(fut_xy_ego: np.ndarray, thresh: float) -> int:
    y_end = float(fut_xy_ego[-1, 1])
    if y_end > thresh:
        return 0  # left
    if y_end < -thresh:
        return 2  # right
    return 1  # keep


def preprocess_highd(
    cfg: HighDPreprocessConfig,
    split_recording_ids: Dict[str, List[int]],
) -> None:
    """Preprocess highD into sharded NPZs.

    Args:
        cfg: preprocessing config
        split_recording_ids: dict with keys {train,val,test} -> list of recording ids.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # time discretization
    dt = 1.0 / cfg.hz
    H_obs = int(round(cfg.obs_sec * cfg.hz))
    H_pred = int(round(cfg.pred_sec * cfg.hz))
    stride_frames_at_hz = int(round(cfg.stride_sec * cfg.hz))

    # obs offsets include t0 (0.0) as last element
    obs_times = np.array([-(H_obs - 1 - i) * dt for i in range(H_obs)], dtype=np.float32)
    fut_times = np.array([(i + 1) * dt for i in range(H_pred)], dtype=np.float32)

    for split, rec_ids in split_recording_ids.items():
        out_split = cfg.out_dir / "highd" / split
        writer = ShardWriter(out_split, shard_size=cfg.shard_size, compress=cfg.compress)
        num_written = 0

        for rec_id in tqdm(rec_ids, desc=f"highD[{split}] recordings"):
            paths = HighDPaths(cfg.raw_dir, int(rec_id))
            if not paths.tracks.exists():
                raise FileNotFoundError(f"Missing {paths.tracks}")
            rec_meta = load_recording_meta(paths.recording_meta)
            frame_rate = float(rec_meta.get("frameRate", 25.0))

            usecols = [
                "frame",
                "id",
                "x",
                "y",
                "xVelocity",
                "yVelocity",
                "xAcceleration",
                "yAcceleration",
                "laneId",
                "precedingId",
                "followingId",
                "leftPrecedingId",
                "leftAlongsideId",
                "leftFollowingId",
                "rightPrecedingId",
                "rightAlongsideId",
                "rightFollowingId",
            ]
            tracks = load_tracks(paths.tracks, usecols=usecols)
            tracks = tracks.sort_values(["id", "frame"], kind="mergesort")
            # positions lookup
            pos_df = tracks[["frame", "id", "x", "y"]].set_index(["frame", "id"]).sort_index()

            # group by track id
            for tid, g in tracks.groupby("id", sort=False):
                g = g.sort_values("frame", kind="mergesort")
                # Use frame range
                f_min = int(g["frame"].iloc[0])
                f_max = int(g["frame"].iloc[-1])

                # Determine max offset in original frames due to rounding
                obs_off = np.array([_round_half_up(float(t) * frame_rate) for t in obs_times], dtype=np.int32)
                fut_off = np.array([_round_half_up(float(t) * frame_rate) for t in fut_times], dtype=np.int32)
                past_req = int(abs(obs_off.min()))
                fut_req = int(fut_off.max())

                start_f0 = f_min + past_req
                end_f0 = f_max - fut_req
                if end_f0 <= start_f0:
                    continue

                # stride in original frames (approx based on seconds)
                stride_frames = _round_half_up(cfg.stride_sec * frame_rate)
                if stride_frames <= 0:
                    stride_frames = 1

                g_idx = g.set_index("frame")

                for f0 in range(start_f0, end_f0 + 1, stride_frames):
                    frames_obs = (f0 + obs_off).astype(np.int32)
                    frames_fut = (f0 + fut_off).astype(np.int32)

                    # quick existence check
                    if frames_obs[0] < f_min or frames_fut[-1] > f_max:
                        continue

                    # EV positions and kinematics
                    try:
                        ev_obs_world = g_idx.loc[frames_obs, ["x", "y"]].to_numpy(dtype=np.float32)
                        ev_fut_world = g_idx.loc[frames_fut, ["x", "y"]].to_numpy(dtype=np.float32)
                        row_f0 = g_idx.loc[f0]
                    except KeyError:
                        # some frames might be missing in corrupted tracks
                        continue

                    # Filter kinematics
                    speed = np.sqrt(np.square(g_idx.loc[frames_obs, "xVelocity"].to_numpy(dtype=np.float32)) +
                                    np.square(g_idx.loc[frames_obs, "yVelocity"].to_numpy(dtype=np.float32)))
                    if np.nanmax(speed) > cfg.max_speed_mps:
                        continue
                    ax = g_idx.loc[frames_obs, "xAcceleration"].to_numpy(dtype=np.float32)
                    ay = g_idx.loc[frames_obs, "yAcceleration"].to_numpy(dtype=np.float32)
                    # lateral acc magnitude proxy: |v x a| / |v|
                    vx = g_idx.loc[frames_obs, "xVelocity"].to_numpy(dtype=np.float32)
                    vy = g_idx.loc[frames_obs, "yVelocity"].to_numpy(dtype=np.float32)
                    cross = vx * ay - vy * ax
                    lat_acc = np.abs(cross) / (speed + 1e-3)
                    if np.nanmax(lat_acc) > cfg.max_lat_acc_mps2:
                        continue

                    # Smooth target positions over the whole segment (obs+fut) for stability
                    ev_all = np.concatenate([ev_obs_world, ev_fut_world], axis=0)
                    ev_all = savgol_smooth_xy(ev_all, window=cfg.sg_window, polyorder=cfg.sg_poly)
                    ev_obs_world = ev_all[:H_obs]
                    ev_fut_world = ev_all[H_obs:]

                    anchor_xy = ev_obs_world[-1].copy()  # at f0 (approx)
                    anchor_yaw = _compute_anchor_yaw(row_f0)

                    # Neighbor extraction
                    if cfg.prefer_highd_neighbors and cfg.neighborhood.mode == "sector" and cfg.neighborhood.num_sectors == 8:
                        neigh_cols = _neighbor_cols_ordered()
                        neigh_ids = g_idx.loc[frames_obs, neigh_cols].to_numpy(dtype=np.int64)  # (H_obs,8)
                        # treat 0 as missing
                        neigh_ids = np.where(neigh_ids > 0, neigh_ids, -1)

                        # Lookup neighbor positions via reindex
                        H, S = neigh_ids.shape
                        frames_rep = np.repeat(frames_obs, S)
                        neigh_flat = neigh_ids.reshape(-1)
                        mi = pd.MultiIndex.from_arrays([frames_rep, neigh_flat], names=["frame", "id"])
                        neigh_xy_world = pos_df.reindex(mi)[["x", "y"]].to_numpy(dtype=np.float32).reshape(H, S, 2)
                        neigh_valid = np.isfinite(neigh_xy_world[..., 0]) & (neigh_ids >= 0)
                        neigh_xy_world = np.nan_to_num(neigh_xy_world, nan=0.0)
                    else:
                        # Geometric selection for each frame
                        S = cfg.neighborhood.num_neighbors()
                        neigh_xy_world = np.zeros((H_obs, S, 2), dtype=np.float32)
                        neigh_valid = np.zeros((H_obs, S), dtype=bool)

                        for t, fr in enumerate(frames_obs):
                            # all vehicles in this frame
                            frame_df = pos_df.loc[fr]  # index becomes id
                            ids = frame_df.index.to_numpy(dtype=np.int64)
                            xy = frame_df[["x", "y"]].to_numpy(dtype=np.float32)
                            # remove target
                            keep = ids != tid
                            ids = ids[keep]
                            xy = xy[keep]
                            if xy.shape[0] == 0:
                                continue
                            # compute rel in ego frame anchored at f0
                            xy_ego = ego_normalize(xy, anchor_xy, anchor_yaw)
                            ev_ego = ego_normalize(ev_obs_world[t:t+1], anchor_xy, anchor_yaw)[0]
                            rel = xy_ego - ev_ego  # (M,2) relative to EV at this frame

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
                                    neigh_xy_world[t, s] = xy[j]
                                    neigh_valid[t, s] = True

                    # Ego normalize to anchor (f0)
                    ev_obs_ego = ego_normalize(ev_obs_world, anchor_xy, anchor_yaw)  # (H_obs,2)
                    neigh_ego = ego_normalize(neigh_xy_world.reshape(-1, 2), anchor_xy, anchor_yaw).reshape(neigh_xy_world.shape)

                    # Build obs tensor (H_obs, N, 2)
                    N = 1 + neigh_ego.shape[1]
                    obs = np.zeros((H_obs, N, 2), dtype=np.float32)
                    obs_mask = np.zeros((H_obs, N), dtype=bool)
                    obs[:, 0] = ev_obs_ego
                    obs_mask[:, 0] = True
                    obs[:, 1:] = neigh_ego
                    obs_mask[:, 1:] = neigh_valid

                    # Future in ego frame
                    fut = ego_normalize(ev_fut_world, anchor_xy, anchor_yaw).astype(np.float32)

                    lc = _infer_lc_label_from_fut(fut, cfg.lc_threshold_m)

                    writer.add(
                        {
                            "obs": obs,
                            "obs_mask": obs_mask,
                            "fut": fut,
                            "lc_label": np.array(lc, dtype=np.int64),
                            "meta_track_id": np.array(int(tid), dtype=np.int64),
                            "meta_rec_id": np.array(int(rec_id), dtype=np.int64),
                            "meta_t0": np.array(int(f0), dtype=np.int64),
                        }
                    )
                    num_written += 1

        writer.close()
        print(f"[highD] Wrote {num_written} samples to {out_split}")
