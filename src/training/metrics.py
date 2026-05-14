"""
Evaluation metrics for Turner / Constable classification.

Provides:
  - compute_metrics(labels, preds, probs) — accuracy, AUC-ROC, F1, confusion matrix
  - format_metrics(metrics_dict)          — pretty-print
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ARTISTS = ["turner", "constable"]


def compute_metrics(
    labels: list[int] | np.ndarray,
    preds: list[int] | np.ndarray,
    probs: list[float] | np.ndarray | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        labels: ground-truth integer labels
        preds:  predicted integer labels
        probs:  predicted probabilities for class 1 (Constable). Used for AUC.
                If None, AUC is not computed.

    Returns:
        Dict with keys: accuracy, auc, f1_turner, f1_constable, f1_macro, confusion_matrix
    """
    labels = np.array(labels)
    preds = np.array(preds)

    acc = accuracy_score(labels, preds)
    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    metrics = {
        "accuracy": float(acc),
        "f1_turner": float(f1_per_class[0]),
        "f1_constable": float(f1_per_class[1]),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
    }

    if probs is not None:
        probs = np.array(probs)
        try:
            auc = roc_auc_score(labels, probs)
            metrics["auc"] = float(auc)
        except ValueError:
            # Can happen if only one class is present in a small batch
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")

    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Return a human-readable string of metric values."""
    lines = []
    if prefix:
        lines.append(f"[{prefix}]")
    lines.append(f"  Accuracy:      {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    if not np.isnan(metrics["auc"]):
        lines.append(f"  AUC-ROC:       {metrics['auc']:.4f}")
    lines.append(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    lines.append(f"  F1 Turner:     {metrics['f1_turner']:.4f}")
    lines.append(f"  F1 Constable:  {metrics['f1_constable']:.4f}")
    cm = metrics["confusion_matrix"]
    lines.append("  Confusion matrix (rows=actual, cols=pred):")
    lines.append(f"               Turner  Constable")
    lines.append(f"    Turner:    {cm[0][0]:6d}  {cm[0][1]:9d}")
    lines.append(f"    Constable: {cm[1][0]:6d}  {cm[1][1]:9d}")
    return "\n".join(lines)


class MetricsAccumulator:
    """Accumulates predictions and labels over batches, then computes metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._labels: list[int] = []
        self._preds: list[int] = []
        self._probs: list[float] = []  # prob for class 1 (Constable)
        self._loss_sum: float = 0.0
        self._n: int = 0

    def update(
        self,
        labels: np.ndarray | list,
        preds: np.ndarray | list,
        probs: np.ndarray | list,
        loss: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        self._labels.extend(labels if isinstance(labels, list) else labels.tolist())
        self._preds.extend(preds if isinstance(preds, list) else preds.tolist())
        self._probs.extend(probs if isinstance(probs, list) else probs.tolist())
        self._loss_sum += loss * batch_size
        self._n += batch_size

    def compute(self) -> dict:
        metrics = compute_metrics(self._labels, self._preds, self._probs)
        metrics["loss"] = self._loss_sum / max(self._n, 1)
        return metrics
