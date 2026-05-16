"""
PyTorch Dataset for Turner / Constable classification.

Provides:
  - ArtDataset          — loads images from processed/{split}/{artist}/
  - get_transforms      — albumentations pipelines for train / val / test
  - get_dataloader      — DataLoader factory with WeightedRandomSampler for train
  - mixup_batch         — MixUp augmentation applied at the batch level
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ARTISTS = ["turner", "constable"]
LABEL_MAP = {artist: idx for idx, artist in enumerate(ARTISTS)}  # turner=0, constable=1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, image_size: int = 384, resize_size: int = 512) -> A.Compose:
    """Return an albumentations pipeline for the given split.

    Args:
        split: one of "train", "val", "test"
        image_size: crop/resize target fed to the model
        resize_size: intermediate resize before crop (training only)
    """
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = ToTensorV2()

    if split == "train":
        # Randomly resize before crop so the model sees both fine brushwork
        # (large image → tight 224px crop) and overall composition
        # (small image → wide 224px crop covering most of the painting)
        scale_choices = [
            A.SmallestMaxSize(max_size=image_size + 32),   # ~90% coverage per crop
            A.SmallestMaxSize(max_size=resize_size),        # ~70% coverage
            A.NoOp(),                                       # full 512px, ~44% coverage
        ]
        return A.Compose([
            A.OneOf(scale_choices, p=1.0),
            A.RandomCrop(height=image_size, width=image_size),
            # Geometry — appropriate for paintings
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.Perspective(scale=(0.0, 0.1), p=0.2),
            # Color — simulate scan variation, aging, lighting
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
            # Blur — simulate lower-res scans
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # Occasionally strip color to force shape/texture learning
            A.ToGray(p=0.05),
            normalize,
            to_tensor,
        ])
    else:
        # Val / test: deterministic center crop
        return A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
            normalize,
            to_tensor,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ArtDataset(Dataset):
    """Loads images from processed/{split}/{artist}/ directories.

    Args:
        root: path to data/processed/
        split: "train", "val", or "test"
        transform: albumentations Compose pipeline (optional; use get_transforms())
        image_size: passed to get_transforms if transform is None
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: A.Compose | None = None,
        image_size: int = 384,
    ):
        self.root = Path(root) / split
        self.split = split
        self.transform = transform or get_transforms(split, image_size)
        self.samples: list[tuple[Path, int]] = []

        for artist in ARTISTS:
            artist_dir = self.root / artist
            if not artist_dir.exists():
                raise FileNotFoundError(
                    f"Expected directory {artist_dir}. Run preprocess.py first."
                )
            label = LABEL_MAP[artist]
            for p in sorted(artist_dir.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((p, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    @property
    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {a: 0 for a in ARTISTS}
        for _, label in self.samples:
            counts[ARTISTS[label]] += 1
        return counts


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 384,
    transform: A.Compose | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for the given split.

    For the train split, uses WeightedRandomSampler to balance classes.
    For val/test, uses sequential sampling.
    """
    dataset = ArtDataset(root=root, split=split, transform=transform, image_size=image_size)

    if split == "train":
        sampler = _make_weighted_sampler(dataset.labels)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances class frequencies."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
    num_classes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp augmentation to a batch.

    Returns mixed images and soft labels (one-hot mixed).
    """
    if alpha <= 0:
        return images, torch.nn.functional.one_hot(labels, num_classes).float()

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[idx]
    labels_a = torch.nn.functional.one_hot(labels, num_classes).float()
    labels_b = torch.nn.functional.one_hot(labels[idx], num_classes).float()
    mixed_labels = lam * labels_a + (1 - lam) * labels_b

    return mixed_images, mixed_labels


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    for split in ("train", "val", "test"):
        try:
            dl = get_dataloader(root, split, batch_size=4, num_workers=0)
            imgs, lbls = next(iter(dl))
            print(f"{split}: batch shape {imgs.shape}, labels {lbls.tolist()}")
        except FileNotFoundError as e:
            print(f"Skipping {split}: {e}")
