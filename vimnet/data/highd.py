from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HighDPaths:
    """Helper to resolve the three CSVs for a given recording id."""

    raw_dir: Path
    rec_id: int

    @property
    def prefix(self) -> str:
        return f"{self.rec_id:02d}"

    @property
    def tracks(self) -> Path:
        return self.raw_dir / f"{self.prefix}_tracks.csv"

    @property
    def tracks_meta(self) -> Path:
        return self.raw_dir / f"{self.prefix}_tracksMeta.csv"

    @property
    def recording_meta(self) -> Path:
        return self.raw_dir / f"{self.prefix}_recordingMeta.csv"


def load_recording_meta(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    # recordingMeta is a single-row csv
    row = df.iloc[0].to_dict()
    return {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in row.items()}


def load_tracks_meta(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_tracks(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols)


def expected_tracks_columns() -> List[str]:
    """Columns as defined by highD 'data format' document."""
    return [
        "frame",
        "id",
        "x",
        "y",
        "width",
        "height",
        "xVelocity",
        "yVelocity",
        "xAcceleration",
        "yAcceleration",
        "frontSightDistance",
        "backSightDistance",
        "dhw",
        "thw",
        "ttc",
        "precedingXVelocity",
        "precedingId",
        "followingId",
        "leftPrecedingId",
        "leftAlongsideId",
        "leftFollowingId",
        "rightPrecedingId",
        "rightAlongsideId",
        "rightFollowingId",
        "laneId",
    ]
