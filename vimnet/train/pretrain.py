from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import AugmentConfig, ShardedNPZDataset, collate_batch
from ..models.heads import PretrainNextStepHead
from ..models.vimnet import VIMNETConfig, VIMNETEncoder
from .schedule import get_cosine_schedule_with_warmup, set_seed


@dataclass
class PretrainConfig:
    train_shard_dir: Path
    val_shard_dir: Path
    out_dir: Path

    seed: int = 1
    device: str = "cuda"
    num_workers: int = 4

    # data
    obs_len: int = 20
    pretrain_ctx: int = 12
    batch_size: int = 128

    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    max_steps: int = 200_000
    warmup_frac: float = 0.05
    eval_every: int = 2000

    # model
    model: VIMNETConfig = VIMNETConfig()


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean()


@torch.no_grad()
def evaluate(model: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    head.eval()
    losses = []
    for batch in loader:
        obs = batch["obs"].to(device)
        mask = batch["obs_mask"].to(device)
        y = batch["y_next"].to(device)
        enc = model(obs, mask)  # (B,T,N,D)
        ev = enc[:, -1, 0]  # last timestep EV
        pred = head(ev)
        losses.append(l2_loss(pred, y).item())
    return float(sum(losses) / max(1, len(losses)))


def run_pretrain(cfg: PretrainConfig) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_ds = ShardedNPZDataset(
        cfg.train_shard_dir,
        task="pretrain",
        augment=AugmentConfig(enabled=True),
        obs_len=cfg.obs_len,
        pretrain_ctx=cfg.pretrain_ctx,
        training=True,
        seed=cfg.seed,
    )
    val_ds = ShardedNPZDataset(
        cfg.val_shard_dir,
        task="pretrain",
        augment=AugmentConfig(enabled=False),
        obs_len=cfg.obs_len,
        pretrain_ctx=cfg.pretrain_ctx,
        training=False,
        seed=cfg.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_batch)

    encoder = VIMNETEncoder(cfg.model).to(device)
    head = PretrainNextStepHead(dim=cfg.model.dim).to(device)

    opt = torch.optim.AdamW( list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    warmup_steps = int(cfg.warmup_frac * cfg.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, cfg.max_steps)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_ckpt = cfg.out_dir / "best_pretrain.pt"

    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="pretrain")
    while step < cfg.max_steps:
        for batch in train_loader:
            encoder.train()
            head.train()
            obs = batch["obs"].to(device, non_blocking=True)
            mask = batch["obs_mask"].to(device, non_blocking=True)
            y = batch["y_next"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                enc = encoder(obs, mask)
                ev = enc[:, -1, 0]
                pred = head(ev)
                loss = l2_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{sched.get_last_lr()[0]:.2e}"})

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val = evaluate(encoder, head, val_loader, device)
                if val < best_val:
                    best_val = val
                    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "cfg": cfg.__dict__}, best_ckpt)
                print(f"\n[pretrain] step={step} val={val:.4f} best={best_val:.4f}")

            if step >= cfg.max_steps:
                break

    pbar.close()
    return best_ckpt
