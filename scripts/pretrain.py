#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vimnet.models.vimnet import VIMNETConfig
from vimnet.train.pretrain import PretrainConfig, run_pretrain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/train/pretrain.yaml")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(Path(args.cfg).read_text()) or {}
    model_yaml = Path(cfg_dict.get("model_yaml", "configs/model/vimnet_base.yaml"))
    model_dict = yaml.safe_load(model_yaml.read_text()) or {}

    model_cfg = VIMNETConfig(**model_dict)

    cfg = PretrainConfig(
        train_shard_dir=Path(cfg_dict["train_shard_dir"]),
        val_shard_dir=Path(cfg_dict["val_shard_dir"]),
        out_dir=Path(cfg_dict["out_dir"]),
        seed=int(cfg_dict.get("seed", 1)),
        device=str(cfg_dict.get("device", "cuda")),
        num_workers=int(cfg_dict.get("num_workers", 4)),
        obs_len=int(cfg_dict.get("obs_len", 20)),
        pretrain_ctx=int(cfg_dict.get("pretrain_ctx", 12)),
        batch_size=int(cfg_dict.get("batch_size", 128)),
        lr=float(cfg_dict.get("lr", 3e-4)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.01)),
        grad_clip=float(cfg_dict.get("grad_clip", 1.0)),
        max_steps=int(cfg_dict.get("max_steps", 200000)),
        warmup_frac=float(cfg_dict.get("warmup_frac", 0.05)),
        eval_every=int(cfg_dict.get("eval_every", 2000)),
        model=model_cfg,
    )

    run_pretrain(cfg)


if __name__ == "__main__":
    main()
