"""
Three-phase training loop for Turner / Constable art classifier.

Phases:
  1. Warm-up: freeze backbone, train head only (fast convergence of random head)
  2. Full fine-tuning: differential LRs across backbone stages
  3. Polish: same as Phase 2 at 10x lower LR

Usage:
    python src/training/train.py --config configs/swin.yaml
    python src/training/train.py --config configs/efficientnet.yaml --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ARTISTS, get_dataloader, mixup_batch
from src.models.classifier import ArtClassifier, build_model, get_param_groups
from src.training.losses import build_loss, get_class_weights
from src.training.metrics import MetricsAccumulator, format_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: ArtClassifier,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    mixup_alpha: float = 0.0,
    accumulate_steps: int = 1,
    grad_clip: float = 1.0,
    scheduler=None,
) -> dict:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    accum = MetricsAccumulator()
    optimizer_step_count = 0

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(tqdm(loader, leave=False)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # MixUp (train only)
            if is_train and mixup_alpha > 0:
                images, soft_labels = mixup_batch(images, labels, alpha=mixup_alpha)
                logits = model(images)
                loss = criterion(logits, soft_labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                (loss / accumulate_steps).backward()
                if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == len(loader):
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step_count += 1

            # Metrics
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
                batch_labels = labels if labels.dim() == 1 else labels.argmax(dim=-1)
                accum.update(
                    labels=batch_labels.cpu().numpy(),
                    preds=preds.cpu().numpy(),
                    probs=probs[:, 1].cpu().numpy(),  # prob of Constable
                    loss=loss.item(),
                    batch_size=images.size(0),
                )

    if is_train and scheduler is not None:
        scheduler.step()

    return accum.compute()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, data_dir: str = "data/processed", resume: str | None = None) -> None:
    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_loader = get_dataloader(data_dir, "train", batch_size, num_workers, image_size)
    val_loader = get_dataloader(data_dir, "val", batch_size, num_workers, image_size)

    # Class weights from training set
    train_labels = train_loader.dataset.labels
    class_weights = get_class_weights(train_labels, device=str(device))
    print(f"Class weights: Turner={class_weights[0]:.3f}, Constable={class_weights[1]:.3f}")

    # --- Model ---
    model_cfg = cfg["model"]
    model = build_model(
        name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        drop_rate=model_cfg.get("drop_rate", 0.4),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
    ).to(device)

    if resume:
        print(f"Resuming from {resume}")
        load_checkpoint(model, Path(resume), device)

    # --- Loss ---
    criterion = build_loss(
        loss_type="label_smoothing",
        smoothing=cfg["training"]["label_smoothing"],
        class_weights=class_weights,
    )

    # --- Checkpointing setup ---
    ckpt_dir = Path(cfg["checkpoints"]["dir"]) / model_cfg["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    patience_count = 0
    patience = cfg["training"]["early_stop_patience"]

    tr_cfg = cfg["training"]
    mixup_alpha = tr_cfg.get("mixup_alpha", 0.4)
    accumulate = tr_cfg.get("accumulate_grad_batches", 4)
    grad_clip = tr_cfg.get("grad_clip", 1.0)

    history = []

    # ==========================================================================
    # PHASE 1 — Warm-up: train only the classification head
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Classifier warm-up (backbone frozen)")
    print("=" * 60)

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=tr_cfg["phase1_lr"],
        weight_decay=tr_cfg["weight_decay"],
    )

    for epoch in range(1, tr_cfg["phase1_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device,
                                  mixup_alpha=0.0,  # no MixUp in warm-up
                                  accumulate_steps=accumulate, grad_clip=grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        _log_epoch("phase1", epoch, train_metrics, val_metrics)
        history.append({"phase": 1, "epoch": epoch, "train": train_metrics, "val": val_metrics})

    # ==========================================================================
    # PHASE 2 — Full fine-tuning with differential LRs
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Full fine-tuning (differential LRs)")
    print("=" * 60)

    model.unfreeze_backbone()
    param_groups = get_param_groups(
        model,
        lr_head=tr_cfg["phase2_lr_head"],
        lr_late_stages=tr_cfg["phase2_lr_late_stages"],
        lr_early_stages=tr_cfg["phase2_lr_early_stages"],
        weight_decay=tr_cfg["weight_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=tr_cfg["T_0"], T_mult=tr_cfg["T_mult"]
    )

    for epoch in range(1, tr_cfg["phase2_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device,
                                  mixup_alpha=mixup_alpha,
                                  accumulate_steps=accumulate, grad_clip=grad_clip,
                                  scheduler=scheduler)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        _log_epoch("phase2", epoch, train_metrics, val_metrics)
        history.append({"phase": 2, "epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_auc = val_metrics.get("auc", 0.0)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            ckpt_dir / "best.pth")
            print(f"  ✓ New best val AUC: {best_auc:.4f} — checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ==========================================================================
    # PHASE 3 — Polish at very low LR
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Final polish (10x lower LR)")
    print("=" * 60)

    # Reload best checkpoint for phase 3
    best_ckpt = ckpt_dir / "best.pth"
    if best_ckpt.exists():
        load_checkpoint(model, best_ckpt, device)
        print(f"Loaded best checkpoint (val AUC {best_auc:.4f}) for phase 3")

    factor = tr_cfg.get("phase3_lr_factor", 0.1)
    param_groups = get_param_groups(
        model,
        lr_head=tr_cfg["phase2_lr_head"] * factor,
        lr_late_stages=tr_cfg["phase2_lr_late_stages"] * factor,
        lr_early_stages=tr_cfg["phase2_lr_early_stages"] * factor,
        weight_decay=tr_cfg["weight_decay"],
    )
    optimizer = torch.optim.AdamW(param_groups)
    patience_count = 0

    for epoch in range(1, tr_cfg["phase3_epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device,
                                  mixup_alpha=mixup_alpha * 0.5,
                                  accumulate_steps=accumulate, grad_clip=grad_clip)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        _log_epoch("phase3", epoch, train_metrics, val_metrics)
        history.append({"phase": 3, "epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_auc = val_metrics.get("auc", 0.0)
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_metrics,
                            ckpt_dir / "best.pth")
            print(f"  ✓ New best val AUC: {best_auc:.4f} — checkpoint saved")

    # Save training history
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")


def _log_epoch(phase: str, epoch: int, train_m: dict, val_m: dict) -> None:
    train_acc = train_m["accuracy"] * 100
    val_acc = val_m["accuracy"] * 100
    val_auc = val_m.get("auc", float("nan"))
    val_f1 = val_m.get("f1_macro", float("nan"))
    print(
        f"  [{phase} ep{epoch:3d}] "
        f"train_loss={train_m['loss']:.4f} train_acc={train_acc:.1f}% | "
        f"val_loss={val_m['loss']:.4f} val_acc={val_acc:.1f}% "
        f"val_auc={val_auc:.4f} val_f1={val_f1:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, data_dir=args.data_dir, resume=args.resume)


if __name__ == "__main__":
    main()
