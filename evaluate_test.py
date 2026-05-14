"""
Final evaluation on the held-out test set.
Run after training completes.

Usage:
    python evaluate_test.py --checkpoint checkpoints/resnet50/best.pth \
        --model-name resnet50 --data-dir data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.dataset import ARTISTS, get_dataloader
from src.inference.predict import load_model
from src.training.metrics import MetricsAccumulator, compute_metrics, format_metrics


def evaluate_test_set(checkpoint: str, model_name: str, data_dir: str, image_size: int = 224):
    model, device = load_model(checkpoint, model_name)
    model.eval()

    loader = get_dataloader(data_dir, "test", batch_size=16, num_workers=0,
                            image_size=image_size)

    accum = MetricsAccumulator()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            accum.update(
                labels=labels.numpy(),
                preds=preds.cpu().numpy(),
                probs=probs[:, 1].cpu().numpy(),
                loss=0.0,
                batch_size=images.size(0),
            )

    metrics = accum.compute()

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(format_metrics(metrics, prefix="Test"))
    print(f"\nDataset: {len(loader.dataset)} test images")
    print(f"  Turner:    {loader.dataset.class_counts['turner']} images")
    print(f"  Constable: {loader.dataset.class_counts['constable']} images")
    print(f"\nCheckpoint: {checkpoint}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    evaluate_test_set(args.checkpoint, args.model_name, args.data_dir, args.image_size)
