from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import AugmentConfig, ShardedNPZDataset, collate_batch
from ..metrics import ade_fde, accuracy
from ..models.adapters import TemporalGRUAdapter
from ..models.heads import GRUTrajectoryHead, LaneChangeHead, label_smoothed_ce
from ..models.trajectory_heads import MLPTrajectoryHead, TransformerTrajectoryHead
from ..models.vimnet import VIMNETConfig, VIMNETEncoder
from .schedule import get_cosine_schedule_with_warmup, set_seed


@dataclass
class FinetuneConfig:
    train_shard_dir: Path
    val_shard_dir: Path
    test_shard_dir: Path
    out_dir: Path

    pretrained_ckpt: Optional[Path] = None  # path to pretrain checkpoint

    seed: int = 1
    device: str = "cuda"
    num_workers: int = 4

    # data
    obs_len: int = 20
    pred_len: int = 50
    batch_size: int = 64

    # optimization
    lr_encoder: float = 2e-4
    lr_heads: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    epochs: int = 60
    warmup_frac: float = 0.05
    early_stop_patience: int = 8

    # multi-task
    train_tp: bool = True
    train_lc: bool = True
    lambda_lc: float = 0.5
    label_smoothing: float = 0.05

    # scheduled sampling (GRU head)
    max_free_running_ratio: float = 0.2
    free_running_warmup_frac: float = 0.3

    # representation pooling / ablations
    context_pool: str = "last"  # last | mean
    use_temporal_gru_adapter: bool = False  # GSAN-style temporal modeling

    # trajectory head type
    trajectory_head: str = "gru"  # gru | mlp | transformer

    # model
    model: VIMNETConfig = VIMNETConfig()

    freeze_encoder: bool = False  # option


def _compute_class_weights(ds: ShardedNPZDataset, num_classes: int = 3) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for shard in ds.shards:
        path = ds.shard_dir / shard["path"]
        data = np.load(path, allow_pickle=False)
        labels = data["lc_label"].astype(np.int64).reshape(-1)
        for c in range(num_classes):
            counts[c] += int((labels == c).sum())
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float32)
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def _fut_to_deltas(fut: torch.Tensor) -> torch.Tensor:
    first = fut[:, 0:1, :]
    rest = fut[:, 1:, :] - fut[:, :-1, :]
    return torch.cat([first, rest], dim=1)


def _get_context(enc: torch.Tensor, cfg: FinetuneConfig, adapter: Optional[TemporalGRUAdapter]) -> torch.Tensor:
    # enc: (B,T,N,D)
    ev_seq = enc[:, :, 0]  # (B,T,D)
    if cfg.use_temporal_gru_adapter:
        assert adapter is not None
        return adapter(ev_seq)
    if cfg.context_pool == "mean":
        return ev_seq.mean(dim=1)
    return ev_seq[:, -1]


@torch.no_grad()
def evaluate(model: VIMNETEncoder, tp_head: nn.Module, lc_head: LaneChangeHead, adapter: Optional[TemporalGRUAdapter], loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    tp_head.eval()
    lc_head.eval()
    if adapter is not None:
        adapter.eval()

    ade_list = []
    fde_list = []
    acc_list = []
    for batch in loader:
        obs = batch["obs"].to(device)
        mask = batch["obs_mask"].to(device)
        fut = batch["fut"].to(device)
        lc = batch["lc_label"].to(device)

        enc = model(obs, mask)
        ev = _get_context(enc, cfg=loader.dataset._finetune_cfg, adapter=adapter)  # type: ignore[attr-defined]

        # trajectory
        if isinstance(tp_head, GRUTrajectoryHead):
            _, pred_pos = tp_head(ev, teacher_deltas=None, free_running_ratio=1.0)
        else:
            _, pred_pos = tp_head(ev)

        logits = lc_head(ev)

        m = ade_fde(pred_pos, fut)
        ade_list.append(m["ADE"])
        fde_list.append(m["FDE"])
        acc_list.append(accuracy(logits, lc))

    return {
        "ADE": float(np.mean(ade_list)),
        "FDE": float(np.mean(fde_list)),
        "LC_SR": float(np.mean(acc_list)) * 100.0,
    }


def run_finetune(cfg: FinetuneConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_ds = ShardedNPZDataset(
        cfg.train_shard_dir,
        task="finetune",
        augment=AugmentConfig(enabled=True),
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        training=True,
        seed=cfg.seed,
    )
    val_ds = ShardedNPZDataset(
        cfg.val_shard_dir,
        task="finetune",
        augment=AugmentConfig(enabled=False),
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        training=False,
        seed=cfg.seed + 1,
    )
    test_ds = ShardedNPZDataset(
        cfg.test_shard_dir,
        task="finetune",
        augment=AugmentConfig(enabled=False),
        obs_len=cfg.obs_len,
        pred_len=cfg.pred_len,
        training=False,
        seed=cfg.seed + 2,
    )

    # Hack: attach cfg so evaluate can access pooling settings via loader.dataset
    train_ds._finetune_cfg = cfg  # type: ignore[attr-defined]
    val_ds._finetune_cfg = cfg  # type: ignore[attr-defined]
    test_ds._finetune_cfg = cfg  # type: ignore[attr-defined]

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)

    model = VIMNETEncoder(cfg.model).to(device)

    # trajectory head selection
    if cfg.trajectory_head == "gru":
        tp_head: nn.Module = GRUTrajectoryHead(dim=cfg.model.dim, hidden=cfg.model.dim, horizon=cfg.pred_len).to(device)
    elif cfg.trajectory_head == "mlp":
        tp_head = MLPTrajectoryHead(dim=cfg.model.dim, horizon=cfg.pred_len).to(device)
    elif cfg.trajectory_head == "transformer":
        tp_head = TransformerTrajectoryHead(dim=cfg.model.dim, horizon=cfg.pred_len, num_layers=2, num_heads=cfg.model.num_heads, dropout=cfg.model.dropout).to(device)
    else:
        raise ValueError(f"Unknown trajectory_head={cfg.trajectory_head}")

    lc_head = LaneChangeHead(dim=cfg.model.dim, num_classes=3).to(device)

    adapter = TemporalGRUAdapter(dim=cfg.model.dim).to(device) if cfg.use_temporal_gru_adapter else None

    # Load pretrain
    if cfg.pretrained_ckpt is not None and cfg.pretrained_ckpt.exists():
        ck = torch.load(cfg.pretrained_ckpt, map_location="cpu")
        model.load_state_dict(ck.get("encoder", ck), strict=False)
        print(f"Loaded pretrained encoder from {cfg.pretrained_ckpt}")

    if cfg.freeze_encoder:
        for p in model.parameters():
            p.requires_grad = False

    class_w = _compute_class_weights(train_ds).to(device)

    params = []
    if not cfg.freeze_encoder:
        params.append({"params": model.parameters(), "lr": cfg.lr_encoder})
    if adapter is not None:
        params.append({"params": adapter.parameters(), "lr": cfg.lr_encoder})
    params.append({"params": tp_head.parameters(), "lr": cfg.lr_heads})
    params.append({"params": lc_head.parameters(), "lr": cfg.lr_heads})

    opt = torch.optim.AdamW(params, betas=cfg.betas, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_frac * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_fde = float("inf")
    best_ckpt = cfg.out_dir / "best_finetune.pt"
    patience = 0

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        tp_head.train()
        lc_head.train()
        if adapter is not None:
            adapter.train()

        pbar = tqdm(train_loader, desc=f"finetune epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            obs = batch["obs"].to(device, non_blocking=True)
            mask = batch["obs_mask"].to(device, non_blocking=True)
            fut = batch["fut"].to(device, non_blocking=True)
            lc = batch["lc_label"].to(device, non_blocking=True)

            teacher_deltas = _fut_to_deltas(fut)

            # scheduled sampling ratio (only for GRU head)
            frac = min(1.0, step / max(1, int(cfg.free_running_warmup_frac * total_steps)))
            free_ratio = cfg.max_free_running_ratio * frac

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                enc = model(obs, mask)
                ev = _get_context(enc, cfg, adapter)

                tp_loss = torch.tensor(0.0, device=device)
                lc_loss = torch.tensor(0.0, device=device)

                if cfg.train_tp:
                    if isinstance(tp_head, GRUTrajectoryHead):
                        pred_deltas, _ = tp_head(ev, teacher_deltas=teacher_deltas, free_running_ratio=free_ratio)
                    else:
                        pred_deltas, _ = tp_head(ev)
                    tp_loss = torch.mean((pred_deltas - teacher_deltas) ** 2)

                if cfg.train_lc:
                    logits = lc_head(ev)
                    lc_loss = label_smoothed_ce(logits, lc, smoothing=cfg.label_smoothing, class_weights=class_w)

                loss = tp_loss + (cfg.lambda_lc * lc_loss if cfg.train_lc else 0.0)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(tp_head.parameters()) + list(lc_head.parameters()) + (list(adapter.parameters()) if adapter is not None else []),
                cfg.grad_clip,
            )
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "tp": f"{tp_loss.item():.4f}", "lc": f"{lc_loss.item():.4f}", "fr": f"{free_ratio:.2f}"})

        # Validation
        val_metrics = evaluate(model, tp_head, lc_head, adapter, val_loader, device)
        print(f"[val] epoch={epoch+1} {val_metrics}")
        if val_metrics["FDE"] < best_fde:
            best_fde = val_metrics["FDE"]
            patience = 0
            torch.save(
                {
                    "encoder": model.state_dict(),
                    "tp_head": tp_head.state_dict(),
                    "lc_head": lc_head.state_dict(),
                    "adapter": adapter.state_dict() if adapter is not None else None,
                    "cfg": cfg.__dict__,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (best FDE={best_fde:.4f})")
                break

    # Test best
    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["encoder"], strict=True)
    tp_head.load_state_dict(ck["tp_head"], strict=True)
    lc_head.load_state_dict(ck["lc_head"], strict=True)
    if adapter is not None and ck.get("adapter") is not None:
        adapter.load_state_dict(ck["adapter"], strict=True)

    test_metrics = evaluate(model, tp_head, lc_head, adapter, test_loader, device)
    print(f"[test] {test_metrics}")
    (cfg.out_dir / "test_metrics.json").write_text(json.dumps({"test": test_metrics, "cfg": cfg.__dict__}, indent=2))
    return best_ckpt
