#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vimnet.models.vimnet import VIMNETConfig
from vimnet.train.finetune import FinetuneConfig, run_finetune


def load_finetune_cfg(path: Path) -> FinetuneConfig:
    d = yaml.safe_load(path.read_text()) or {}
    model_yaml = Path(d.get("model_yaml", "configs/model/vimnet_base.yaml"))
    model_dict = yaml.safe_load(model_yaml.read_text()) or {}
    model_cfg = VIMNETConfig(**model_dict)

    ckpt = d.get("pretrained_ckpt", None)
    ckpt_path = Path(ckpt) if ckpt else None

    return FinetuneConfig(
        train_shard_dir=Path(d["train_shard_dir"]),
        val_shard_dir=Path(d["val_shard_dir"]),
        test_shard_dir=Path(d["test_shard_dir"]),
        out_dir=Path(d["out_dir"]),
        pretrained_ckpt=ckpt_path,
        seed=int(d.get("seed", 1)),
        device=str(d.get("device", "cuda")),
        num_workers=int(d.get("num_workers", 4)),
        obs_len=int(d.get("obs_len", 20)),
        pred_len=int(d.get("pred_len", 50)),
        batch_size=int(d.get("batch_size", 64)),
        lr_encoder=float(d.get("lr_encoder", 2e-4)),
        lr_heads=float(d.get("lr_heads", 5e-4)),
        weight_decay=float(d.get("weight_decay", 0.01)),
        grad_clip=float(d.get("grad_clip", 1.0)),
        epochs=int(d.get("epochs", 60)),
        warmup_frac=float(d.get("warmup_frac", 0.05)),
        early_stop_patience=int(d.get("early_stop_patience", 8)),
        train_tp=bool(d.get("train_tp", True)),
        train_lc=bool(d.get("train_lc", True)),
        lambda_lc=float(d.get("lambda_lc", 0.5)),
        label_smoothing=float(d.get("label_smoothing", 0.05)),
        max_free_running_ratio=float(d.get("max_free_running_ratio", 0.2)),
        free_running_warmup_frac=float(d.get("free_running_warmup_frac", 0.3)),
        context_pool=str(d.get("context_pool", "last")),
        use_temporal_gru_adapter=bool(d.get("use_temporal_gru_adapter", False)),
        trajectory_head=str(d.get("trajectory_head", "gru")),
        freeze_encoder=bool(d.get("freeze_encoder", False)),
        model=model_cfg,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cfgs", nargs="+", help="List of finetune YAML configs")
    args = ap.parse_args()

    for cfg_path in args.cfgs:
        cfg = load_finetune_cfg(Path(cfg_path))
        print(f"\n=== Running {cfg_path} ===")
        run_finetune(cfg)


if __name__ == "__main__":
    main()
