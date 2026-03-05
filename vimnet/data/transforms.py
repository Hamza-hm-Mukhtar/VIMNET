from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_xy(xy: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky–Golay smoothing to x and y independently."""
    if xy.shape[0] < window:
        return xy.astype(np.float32)
    x = savgol_filter(xy[:, 0], window_length=window, polyorder=polyorder, mode="interp")
    y = savgol_filter(xy[:, 1], window_length=window, polyorder=polyorder, mode="interp")
    return np.stack([x, y], axis=1).astype(np.float32)


def resample_xy(times_s: np.ndarray, xy: np.ndarray, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a trajectory to uniform dt using linear interpolation."""
    t0 = float(times_s[0])
    t1 = float(times_s[-1])
    new_times = np.arange(t0, t1 + 1e-6, dt, dtype=np.float32)
    new_x = np.interp(new_times, times_s, xy[:, 0]).astype(np.float32)
    new_y = np.interp(new_times, times_s, xy[:, 1]).astype(np.float32)
    return new_times, np.stack([new_x, new_y], axis=1)


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if x.shape[0] < 2:
        return out
    out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    out[0] = (x[1] - x[0]) / dt
    out[-1] = (x[-1] - x[-2]) / dt
    return out


def compute_kinematics(times_s: np.ndarray, xy: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute velocity, acceleration, speed, heading, lateral acceleration."""
    dt = float(np.median(np.diff(times_s))) if times_s.shape[0] >= 2 else 0.1
    vx = _finite_diff(xy[:, 0], dt)
    vy = _finite_diff(xy[:, 1], dt)
    ax = _finite_diff(vx, dt)
    ay = _finite_diff(vy, dt)
    speed = np.sqrt(vx * vx + vy * vy)
    heading = np.arctan2(vy, vx)  # radians
    cross = vx * ay - vy * ax
    lat_acc = np.abs(cross) / (speed + 1e-3)
    return {
        "vx": vx.astype(np.float32),
        "vy": vy.astype(np.float32),
        "ax": ax.astype(np.float32),
        "ay": ay.astype(np.float32),
        "speed": speed.astype(np.float32),
        "heading": heading.astype(np.float32),
        "lat_acc": lat_acc.astype(np.float32),
        "dt": np.array([dt], dtype=np.float32),
    }


def rotation_matrix(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def ego_normalize(xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Translate by anchor and rotate so anchor_yaw aligns with +x."""
    R = rotation_matrix(-anchor_yaw)  # world -> ego
    flat = xy.reshape(-1, 2) - anchor_xy.reshape(1, 2)
    out = flat @ R.T
    return out.reshape(xy.shape).astype(np.float32)
