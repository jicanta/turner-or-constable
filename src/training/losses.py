"""
Loss functions for art classification.

Provides:
  - get_class_weights(dataset)     — compute class weights from label distribution
  - LabelSmoothingCrossEntropy     — CrossEntropyLoss with label smoothing
  - FocalLoss                      — down-weights easy examples; useful when classes overlap
  - MixupLoss                      — wrapper that handles soft (MixUp) labels
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_class_weights(labels: list[int], num_classes: int = 2, device: str = "cpu") -> torch.Tensor:
    """Return inverse-frequency class weights as a tensor.

    Example: if Turner:Constable = 4.8:1, returns [1.0, 4.8] (normalized so min=1).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.min()  # normalize so smallest weight == 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Prevents the model from becoming overconfident, which is important
    when fine-grained boundaries between classes are ambiguous.

    Args:
        smoothing: label smoothing factor in [0, 1). 0.1 is a good default.
        weight: optional class weights tensor (shape [num_classes])
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw model outputs
            targets: (N,) integer class indices OR (N, C) soft labels (from MixUp)
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            # Hard labels → convert to soft labels
            smooth_val = self.smoothing / (num_classes - 1)
            one_hot = torch.full_like(log_probs, smooth_val)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            soft_targets = one_hot
        else:
            # Already soft labels (e.g. from MixUp)
            soft_targets = targets

        loss = -(soft_targets * log_probs)

        if self.weight is not None:
            # Apply class weights across the class dimension
            weight = self.weight.to(logits.device)
            loss = loss * weight.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

    Down-weights easy examples, focusing training on hard, misclassified samples.
    Particularly useful when classes are visually similar (Turner vs Constable).

    Args:
        gamma: focusing parameter. gamma=0 → standard CE. gamma=2 is the common choice.
        weight: optional class weights
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(
    loss_type: str = "label_smoothing",
    smoothing: float = 0.1,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Factory to select a loss function by name."""
    if loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma, weight=class_weights)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: label_smoothing, focal, cross_entropy")
