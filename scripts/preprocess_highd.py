#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vimnet.data.preprocess_highd import HighDPreprocessConfig, preprocess_highd
from vimnet.data.stsg import NeighborhoodConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/preprocess/highd.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    cfg_dict = yaml.safe_load(cfg_path.read_text()) or {}

    neigh = cfg_dict.get("neighborhood", {}) or {}
    neighborhood = NeighborhoodConfig(
        mode=neigh.get("mode", "sector"),
        num_sectors=int(neigh.get("num_sectors", 8)),
        fov_lon_m=float(neigh.get("fov_lon_m", 20.0)),
        fov_lat_m=float(neigh.get("fov_lat_m", 10.0)),
        knn_k=int(neigh.get("knn_k", 8)),
        radius_m=float(neigh.get("radius_m", 50.0)),
    )

    cfg = HighDPreprocessConfig(
        raw_dir=Path(cfg_dict["raw_dir"]).expanduser(),
        out_dir=Path(cfg_dict["out_dir"]).expanduser(),
        hz=float(cfg_dict.get("hz", 10.0)),
        obs_sec=float(cfg_dict.get("obs_sec", 2.0)),
        pred_sec=float(cfg_dict.get("pred_sec", 5.0)),
        stride_sec=float(cfg_dict.get("stride_sec", 0.5)),
        sg_window=int(cfg_dict.get("sg_window", 9)),
        sg_poly=int(cfg_dict.get("sg_poly", 2)),
        max_speed_mps=float(cfg_dict.get("max_speed_mps", 55.0)),
        max_lat_acc_mps2=float(cfg_dict.get("max_lat_acc_mps2", 8.0)),
        lc_threshold_m=float(cfg_dict.get("lc_threshold_m", 1.75)),
        neighborhood=neighborhood,
        shard_size=int(cfg_dict.get("shard_size", 2048)),
        compress=bool(cfg_dict.get("compress", False)),
        prefer_highd_neighbors=bool(cfg_dict.get("prefer_highd_neighbors", True)),
    )

    splits_yaml = Path(cfg_dict.get("splits_yaml", "configs/splits/highd.yaml"))
    splits = yaml.safe_load(splits_yaml.read_text()) or {}
    preprocess_highd(cfg, split_recording_ids={"train": splits.get("train", []), "val": splits.get("val", []), "test": splits.get("test", [])})


if __name__ == "__main__":
    main()
