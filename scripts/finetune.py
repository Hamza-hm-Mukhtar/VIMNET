#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vimnet.models.vimnet import VIMNETConfig
from vimnet.train.finetune import FinetuneConfig, run_finetune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/train/finetune_highd.yaml")
    args = ap.parse_args()

    cfg_dict = yaml.safe_load(Path(args.cfg).read_text()) or {}
    model_yaml = Path(cfg_dict.get("model_yaml", "configs/model/vimnet_base.yaml"))
    model_dict = yaml.safe_load(model_yaml.read_text()) or {}
    model_cfg = VIMNETConfig(**model_dict)

    ckpt = cfg_dict.get("pretrained_ckpt", None)
    ckpt_path = Path(ckpt) if ckpt else None

    cfg = FinetuneConfig(
        train_shard_dir=Path(cfg_dict["train_shard_dir"]),
        val_shard_dir=Path(cfg_dict["val_shard_dir"]),
        test_shard_dir=Path(cfg_dict["test_shard_dir"]),
        out_dir=Path(cfg_dict["out_dir"]),
        pretrained_ckpt=ckpt_path,
        seed=int(cfg_dict.get("seed", 1)),
        device=str(cfg_dict.get("device", "cuda")),
        num_workers=int(cfg_dict.get("num_workers", 4)),
        obs_len=int(cfg_dict.get("obs_len", 20)),
        pred_len=int(cfg_dict.get("pred_len", 50)),
        batch_size=int(cfg_dict.get("batch_size", 64)),
        lr_encoder=float(cfg_dict.get("lr_encoder", 2e-4)),
        lr_heads=float(cfg_dict.get("lr_heads", 5e-4)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.01)),
        grad_clip=float(cfg_dict.get("grad_clip", 1.0)),
        epochs=int(cfg_dict.get("epochs", 60)),
        warmup_frac=float(cfg_dict.get("warmup_frac", 0.05)),
        early_stop_patience=int(cfg_dict.get("early_stop_patience", 8)),
        train_tp=bool(cfg_dict.get("train_tp", True)),
        train_lc=bool(cfg_dict.get("train_lc", True)),
        lambda_lc=float(cfg_dict.get("lambda_lc", 0.5)),
        label_smoothing=float(cfg_dict.get("label_smoothing", 0.05)),
        max_free_running_ratio=float(cfg_dict.get("max_free_running_ratio", 0.2)),
        free_running_warmup_frac=float(cfg_dict.get("free_running_warmup_frac", 0.3)),
        context_pool=str(cfg_dict.get("context_pool", "last")),
        use_temporal_gru_adapter=bool(cfg_dict.get("use_temporal_gru_adapter", False)),
        trajectory_head=str(cfg_dict.get("trajectory_head", "gru")),
        freeze_encoder=bool(cfg_dict.get("freeze_encoder", False)),
        model=model_cfg,
    )

    run_finetune(cfg)


if __name__ == "__main__":
    main()
