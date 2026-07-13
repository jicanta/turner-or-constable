"""
Frozen-backbone feature probe for Turner / Constable classification.

Extracts features once with a strong self-supervised backbone (DINOv2 by
default), then trains the small classification head on the cached features.
This is dramatically cheaper than fine-tuning on CPU and — on a dataset this
small — usually more accurate, since the backbone can't overfit.

Pipeline:
  1. Extract features for train (several augmented views per image), val and
     test (center view + horizontal flip for TTA). Features are cached to
     checkpoints/<name>/features.npz so re-runs skip extraction.
  2. Small hyperparameter sweep (dropout x weight decay) on val AUC.
  3. Retrain best config, report val/test metrics (test uses flip-TTA).
  4. Export a standard ArtClassifier checkpoint (backbone + trained head) to
     checkpoints/<name>/best.pth — usable by evaluate_test.py, predict.py
     and app.py unchanged.

Usage:
    python src/training/probe.py                       # DINOv2 ViT-S/14 @ 336
    python src/training/probe.py --train-views 4 --image-size 336
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ArtDataset, get_transforms
from src.models.classifier import ArtClassifier
from src.training.metrics import compute_metrics, format_metrics

DEFAULT_BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_backbone(name: str, device: torch.device) -> nn.Module:
    import timm
    try:
        backbone = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
    except TypeError:
        backbone = timm.create_model(name, pretrained=True, num_classes=0)
    return backbone.eval().to(device)


@torch.no_grad()
def extract_split(
    backbone: nn.Module,
    data_dir: str,
    split: str,
    image_size: int,
    device: torch.device,
    views: int = 1,
    augment: bool = False,
    flip: bool = False,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [n_views*N, D], labels [n_views*N]) for a split.

    augment=True uses the training augmentation pipeline per view;
    flip=True extracts a horizontally-flipped copy instead of augmenting.
    """
    transform = get_transforms("train" if augment else "val", image_size)
    feats, labels = [], []

    for view in range(views):
        ds = ArtDataset(data_dir, split, transform=transform, image_size=image_size)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        for images, lbls in tqdm(loader, desc=f"{split} view {view + 1}/{views}", leave=False):
            if flip:
                images = torch.flip(images, dims=[3])
            f = backbone(images.to(device))
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(feats), np.concatenate(labels)


def dataset_fingerprint(data_dir: str) -> str:
    """Hash of every split's file list, so a re-split invalidates the cache."""
    import hashlib
    h = hashlib.md5()
    for split in ("train", "val", "test"):
        ds = ArtDataset(data_dir, split, transform=get_transforms("val", 224))
        for p, label in ds.samples:
            h.update(f"{split}/{label}/{p.name}".encode())
    return h.hexdigest()


def extract_all(args, device: torch.device) -> dict[str, np.ndarray]:
    cache = Path(args.checkpoint_dir) / args.name / "features.npz"
    meta = {"backbone": args.backbone, "image_size": args.image_size,
            "train_views": args.train_views, "data": dataset_fingerprint(args.data_dir)}

    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if json.loads(str(data["meta"])) == meta:
            print(f"Using cached features from {cache}")
            return {k: data[k] for k in data.files if k != "meta"}
        print("Feature cache is stale (different backbone/size/views) — re-extracting")

    backbone = build_backbone(args.backbone, device)
    out = {}
    # Train: one canonical center view + (train_views - 1) augmented views
    out["X_train"], out["y_train"] = extract_split(
        backbone, args.data_dir, "train", args.image_size, device, views=1)
    if args.train_views > 1:
        Xa, ya = extract_split(backbone, args.data_dir, "train", args.image_size, device,
                               views=args.train_views - 1, augment=True)
        out["X_train"] = np.concatenate([out["X_train"], Xa])
        out["y_train"] = np.concatenate([out["y_train"], ya])

    out["X_val"], out["y_val"] = extract_split(
        backbone, args.data_dir, "val", args.image_size, device)
    out["X_test"], out["y_test"] = extract_split(
        backbone, args.data_dir, "test", args.image_size, device)
    out["X_test_flip"], _ = extract_split(
        backbone, args.data_dir, "test", args.image_size, device, flip=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, meta=json.dumps(meta), **out)
    print(f"Features cached to {cache}")
    del backbone
    return out


# ---------------------------------------------------------------------------
# Head training on cached features
# ---------------------------------------------------------------------------

def make_head(feature_dim: int, hidden_dim: int, drop_rate: float) -> nn.Sequential:
    """Same structure as ArtClassifier.head so weights transfer directly."""
    head = nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p=drop_rate),
        nn.Linear(hidden_dim, 2),
    )
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
    return head


def train_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int,
    drop_rate: float,
    weight_decay: float,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[nn.Sequential, float]:
    """Train the head full-batch on cached features; return (head, best val AUC)."""
    torch.manual_seed(seed)
    head = make_head(X_train.shape[1], hidden_dim, drop_rate)

    counts = torch.bincount(y_train, minlength=2).float()
    class_weights = counts.sum() / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(X_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            probs = torch.softmax(head(X_val), dim=-1)[:, 1].numpy()
        m = compute_metrics(y_val.numpy(), (probs > 0.5).astype(int), probs)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head, best_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Frozen-feature probe training")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--name", default="dinov2_probe", help="checkpoint subdirectory name")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--train-views", type=int, default=4,
                        help="feature views per training image (1 center + N-1 augmented)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    print(f"Backbone: {args.backbone} @ {args.image_size}px, device: {device}")

    data = extract_all(args, device)
    X_train = torch.from_numpy(data["X_train"]).float()
    y_train = torch.from_numpy(data["y_train"]).long()
    X_val = torch.from_numpy(data["X_val"]).float()
    y_val = torch.from_numpy(data["y_val"]).long()
    X_test = torch.from_numpy(data["X_test"]).float()
    y_test = torch.from_numpy(data["y_test"]).long()
    X_test_flip = torch.from_numpy(data["X_test_flip"]).float()

    print(f"Train features: {tuple(X_train.shape)} "
          f"({int((y_train == 0).sum())} turner / {int((y_train == 1).sum())} constable views)")

    # --- Hyperparameter sweep on val AUC ---
    print("\nSweeping head hyperparameters...")
    best = {"auc": -1.0}
    for drop_rate in (0.2, 0.4):
        for weight_decay in (1e-3, 1e-2, 5e-2):
            _, auc = train_head(X_train, y_train, X_val, y_val,
                                args.hidden_dim, drop_rate, weight_decay,
                                epochs=args.epochs, seed=args.seed)
            print(f"  drop={drop_rate} wd={weight_decay:g}: val AUC {auc:.4f}")
            if auc > best["auc"]:
                best = {"auc": auc, "drop_rate": drop_rate, "weight_decay": weight_decay}

    print(f"\nBest config: drop={best['drop_rate']} wd={best['weight_decay']:g} "
          f"(val AUC {best['auc']:.4f})")
    head, val_auc = train_head(X_train, y_train, X_val, y_val,
                               args.hidden_dim, best["drop_rate"], best["weight_decay"],
                               epochs=args.epochs, seed=args.seed)

    # --- Test evaluation (flip TTA) ---
    head.eval()
    with torch.no_grad():
        probs = (torch.softmax(head(X_test), dim=-1) +
                 torch.softmax(head(X_test_flip), dim=-1))[:, 1].numpy() / 2.0
    test_metrics = compute_metrics(y_test.numpy(), (probs > 0.5).astype(int), probs)
    print("\n" + "=" * 60)
    print("PROBE TEST RESULTS (flip TTA)")
    print("=" * 60)
    print(format_metrics(test_metrics, prefix="Test"))

    # --- Export as a standard ArtClassifier checkpoint ---
    print("\nExporting full checkpoint (backbone + head)...")
    model = ArtClassifier(
        backbone_name=args.backbone,
        pretrained=True,
        drop_rate=best["drop_rate"],
        head_hidden_dim=args.hidden_dim,
    )
    model.head.load_state_dict(head.state_dict())

    out_dir = Path(args.checkpoint_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "metrics": {"val_auc": val_auc, **{f"test_{k}": v for k, v in test_metrics.items()
                                           if not isinstance(v, (list, np.ndarray))}},
        "config": {
            "backbone": args.backbone,
            "image_size": args.image_size,
            "hidden_dim": args.hidden_dim,
            **best,
        },
    }, out_dir / "best.pth")

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "backbone": args.backbone,
            "image_size": args.image_size,
            "train_views": args.train_views,
            "best_config": best,
            "val_auc": val_auc,
            "test": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in test_metrics.items()},
        }, f, indent=2)

    print(f"Checkpoint: {out_dir / 'best.pth'}")
    print(f"Evaluate with: python evaluate_test.py --checkpoint {out_dir / 'best.pth'} "
          f"--model-name {args.backbone} --image-size {args.image_size}")


if __name__ == "__main__":
    main()
