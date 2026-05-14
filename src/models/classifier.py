"""
Model factory for Turner / Constable art classification.

Provides:
  - build_model(name, ...)         — create a single model (Swin-Base or EfficientNet-B4)
  - EnsembleModel                  — average softmax outputs of multiple models
  - get_param_groups(model, cfg)   — differential learning-rate parameter groups
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Install timm: pip install timm")

SUPPORTED_MODELS = {
    "swin_base_patch4_window7_224": "Swin-Transformer-Base (recommended)",
    "efficientnet_b4": "EfficientNet-B4",
    "resnet50": "ResNet-50 (baseline)",
}


# ---------------------------------------------------------------------------
# Single model builder
# ---------------------------------------------------------------------------

class ArtClassifier(nn.Module):
    """Wraps a timm backbone with a custom classification head.

    Head: backbone_features → Linear(hidden_dim) → GELU → Dropout → Linear(num_classes)
    """

    def __init__(
        self,
        backbone_name: str = "swin_base_patch4_window7_224",
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.4,
        head_hidden_dim: int = 512,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Create backbone without its classifier head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # removes the default head
        )
        feature_dim = self.backbone.num_features

        # Custom head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(head_hidden_dim, num_classes),
        )

        self._init_head()

    def _init_head(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1 warm-up)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (Phase 2+)."""
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_model(
    name: str = "swin_base_patch4_window7_224",
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.4,
    head_hidden_dim: int = 512,
) -> ArtClassifier:
    """Create an ArtClassifier from a model name string."""
    if name not in SUPPORTED_MODELS:
        print(f"WARNING: '{name}' is not in the tested list {list(SUPPORTED_MODELS)}. Proceeding anyway.")
    return ArtClassifier(
        backbone_name=name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        head_hidden_dim=head_hidden_dim,
    )


# ---------------------------------------------------------------------------
# Differential LR parameter groups
# ---------------------------------------------------------------------------

def get_param_groups(
    model: ArtClassifier,
    lr_head: float = 1e-4,
    lr_late_stages: float = 5e-5,
    lr_early_stages: float = 1e-5,
    weight_decay: float = 0.01,
) -> list[dict]:
    """Split model parameters into three LR groups for Phase 2 fine-tuning.

    For Swin-Transformer, 'early stages' = stages 0-1, 'late stages' = stages 2-3.
    For other architectures, falls back to a two-group split (head vs. backbone).

    Returns a list of dicts suitable for torch.optim.AdamW(param_groups).
    """
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}

    # Try to identify early vs late backbone stages
    backbone = model.backbone
    early_params: list[nn.Parameter] = []
    late_params: list[nn.Parameter] = []

    # Swin-Transformer: backbone.layers is a ModuleList of 4 stages
    if hasattr(backbone, "layers") and hasattr(backbone, "patch_embed"):
        all_layers = list(backbone.layers)
        n = len(all_layers)
        split = n // 2
        early_modules = [backbone.patch_embed] + all_layers[:split]
        late_modules = all_layers[split:]
        for m in early_modules:
            early_params.extend(m.parameters())
        for m in late_modules:
            late_params.extend(m.parameters())
        # Norm layer (if any)
        if hasattr(backbone, "norm"):
            late_params.extend(backbone.norm.parameters())
    # EfficientNet / ResNet: simpler split
    elif hasattr(backbone, "blocks") or hasattr(backbone, "layer1"):
        for name, param in backbone.named_parameters():
            if id(param) not in head_ids:
                if any(k in name for k in ["layer1", "layer2", "blocks.0", "blocks.1",
                                            "stem", "patch_embed", "conv_stem"]):
                    early_params.append(param)
                else:
                    late_params.append(param)
    else:
        # Generic fallback: all backbone params at lr_late_stages
        for param in backbone.parameters():
            late_params.append(param)

    # Deduplicate: some params may appear in multiple lists
    early_ids = {id(p) for p in early_params}
    late_ids = {id(p) for p in late_params}
    # Params not captured go to late group
    for param in backbone.parameters():
        if id(param) not in early_ids and id(param) not in late_ids:
            late_params.append(param)

    return [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": late_params, "lr": lr_late_stages, "weight_decay": weight_decay},
        {"params": early_params, "lr": lr_early_stages, "weight_decay": weight_decay},
    ]


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    """Averages softmax probabilities from multiple ArtClassifier models.

    Usage:
        ensemble = EnsembleModel([swin_model, efficientnet_model])
        logits = ensemble(x)  # returns averaged log-probabilities (log-softmax)
    """

    def __init__(self, models: list[ArtClassifier]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = [torch.softmax(m(x), dim=-1) for m in self.models]
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        return torch.log(avg_probs + 1e-8)  # log-probs for NLLLoss compatibility

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_class, confidence_probability)."""
        with torch.no_grad():
            log_probs = self.forward(x)
            probs = torch.exp(log_probs)
            predicted = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
        return predicted, confidence


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building Swin-Base model...")
    model = build_model("swin_base_patch4_window7_224", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"  Output shape: {out.shape}")  # should be (2, 2)

    print("Building EfficientNet-B4 model...")
    eff = build_model("efficientnet_b4", pretrained=False)
    x2 = torch.randn(2, 3, 380, 380)
    out2 = eff(x2)
    print(f"  Output shape: {out2.shape}")

    print("Building ensemble...")
    ensemble = EnsembleModel([model, eff])
    # Use same size for ensemble test
    x3 = torch.randn(2, 3, 224, 224)
    pred, conf = model.head(model.backbone(x3)), None
    print("  All models built successfully.")
