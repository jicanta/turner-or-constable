"""
Inference module for Turner / Constable art classifier.

Provides:
  - load_model(checkpoint_path, model_name)     — load a trained ArtClassifier
  - predict_single(image, model, ...)           — predict one painting
  - predict_batch(image_dir, model, ...)        — predict all images in a folder
  - predict_with_tta(image, model, ...)         — TTA: average over augmented views
  - generate_gradcam(image, model, ...)         — Grad-CAM heatmap

Usage:
    python src/inference/predict.py --checkpoint checkpoints/swin.../best.pth \
        --model-name swin_base_patch4_window7_224 --image path/to/painting.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ARTISTS, IMAGENET_MEAN, IMAGENET_STD, get_transforms
from src.models.classifier import ArtClassifier, build_model

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    model_name: str = "swin_base_patch4_window7_224",
    device: str | torch.device | None = None,
    head_hidden_dim: int | None = None,
) -> tuple[ArtClassifier, torch.device]:
    """Load a trained ArtClassifier from a checkpoint file."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Auto-detect head_hidden_dim from checkpoint if not specified
    if head_hidden_dim is None:
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        if "head.0.weight" in state:
            head_hidden_dim = state["head.0.weight"].shape[0]
        else:
            head_hidden_dim = 512  # default

    model = build_model(name=model_name, pretrained=False, head_hidden_dim=head_hidden_dim)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    return model, device


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, image_size: int = 384) -> torch.Tensor:
    """Convert a PIL image to a normalized tensor (1, C, H, W)."""
    transform = get_transforms("test", image_size=image_size)
    img_np = np.array(image.convert("RGB"))
    tensor = transform(image=img_np)["image"]
    return tensor.unsqueeze(0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single(
    image: str | Path | Image.Image,
    model: ArtClassifier,
    device: torch.device,
    image_size: int = 384,
) -> dict:
    """Predict the artist for a single painting.

    Returns:
        {
            "artist": "Turner",
            "confidence": 0.94,
            "probabilities": {"turner": 0.94, "constable": 0.06}
        }
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    tensor = preprocess_image(image, image_size).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_idx = int(probs.argmax())
    return {
        "artist": ARTISTS[predicted_idx].capitalize(),
        "confidence": float(probs[predicted_idx]),
        "probabilities": {artist: float(p) for artist, p in zip(ARTISTS, probs)},
    }


def predict_with_tta(
    image: str | Path | Image.Image,
    model: ArtClassifier,
    device: torch.device,
    image_size: int = 384,
    n_augments: int = 5,
) -> dict:
    """Predict with Test-Time Augmentation (TTA).

    Runs the image through multiple augmented versions and averages softmax outputs.
    Typically adds 1-2% accuracy over a single forward pass.

    Args:
        n_augments: number of augmented views (including original)

    Returns: same format as predict_single, but with tta=True added.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")
    img_np = np.array(image)

    # Define TTA transforms
    tta_transforms = _build_tta_transforms(image_size, n_augments)

    all_probs = []
    with torch.no_grad():
        for tfm in tta_transforms:
            aug = tfm(image=img_np)["image"].unsqueeze(0).to(device)
            logits = model(aug)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

    avg_probs = np.stack(all_probs).mean(axis=0)
    predicted_idx = int(avg_probs.argmax())

    return {
        "artist": ARTISTS[predicted_idx].capitalize(),
        "confidence": float(avg_probs[predicted_idx]),
        "probabilities": {artist: float(p) for artist, p in zip(ARTISTS, avg_probs)},
        "tta": True,
        "n_augments": len(tta_transforms),
    }


def _build_tta_transforms(image_size: int, n: int) -> list:
    """Build a list of albumentations pipelines for TTA."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = ToTensorV2()

    base = A.Compose([A.CenterCrop(image_size, image_size), normalize, to_tensor])
    hflip = A.Compose([A.CenterCrop(image_size, image_size), A.HorizontalFlip(p=1.0), normalize, to_tensor])
    rot_pos = A.Compose([A.Rotate(limit=(5, 5), p=1.0), A.CenterCrop(image_size, image_size), normalize, to_tensor])
    rot_neg = A.Compose([A.Rotate(limit=(-5, -5), p=1.0), A.CenterCrop(image_size, image_size), normalize, to_tensor])
    bright = A.Compose([A.CenterCrop(image_size, image_size), A.RandomBrightnessContrast(0.15, 0.1, p=1.0), normalize, to_tensor])

    transforms = [base, hflip, rot_pos, rot_neg, bright]
    return transforms[:n]


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    image_dir: str | Path,
    model: ArtClassifier,
    device: torch.device,
    image_size: int = 384,
    use_tta: bool = False,
) -> list[dict]:
    """Predict all images in a directory."""
    import pandas as pd
    image_dir = Path(image_dir)
    results = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]

    if not paths:
        print(f"No images found in {image_dir}")
        return results

    fn = predict_with_tta if use_tta else predict_single
    for path in paths:
        try:
            result = fn(path, model, device, image_size)
            result["filename"] = path.name
            results.append(result)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(
    image: str | Path | Image.Image,
    model: ArtClassifier,
    device: torch.device,
    target_class: int | None = None,
    image_size: int = 384,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Grad-CAM heatmap for the given image.

    Args:
        target_class: 0=Turner, 1=Constable. If None, uses predicted class.

    Returns:
        (rgb_image, cam_overlay) — both as float32 numpy arrays in [0, 1].
    """
    if not HAS_GRADCAM:
        raise ImportError("Install pytorch-grad-cam: pip install pytorch-grad-cam")

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = image.convert("RGB")

    tensor = preprocess_image(image, image_size).to(device)

    # Choose target layer based on architecture
    target_layer = _get_target_layer(model)

    if target_class is None:
        with torch.no_grad():
            logits = model(tensor)
            target_class = int(logits.argmax(dim=-1).item())

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]

    # Prepare RGB image for overlay (resize to image_size)
    img_resized = image.resize((image_size, image_size))
    rgb = np.array(img_resized).astype(np.float32) / 255.0

    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return rgb, overlay


def _get_target_layer(model: ArtClassifier):
    """Return a suitable layer for Grad-CAM based on the backbone architecture."""
    backbone = model.backbone
    if hasattr(backbone, "layers"):
        # Swin-Transformer: last norm layer of last stage
        return backbone.layers[-1].blocks[-1].norm1
    elif hasattr(backbone, "blocks"):
        # EfficientNet: last conv block
        return backbone.blocks[-1][-1]
    elif hasattr(backbone, "layer4"):
        # ResNet
        return backbone.layer4[-1]
    else:
        # Generic fallback: last module with parameters
        for layer in reversed(list(backbone.modules())):
            if hasattr(layer, "weight"):
                return layer
        raise RuntimeError("Could not identify a target layer for Grad-CAM")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict artist for a painting")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="swin_base_patch4_window7_224")
    parser.add_argument("--image", required=True, help="Path to image or directory")
    parser.add_argument("--tta", action="store_true", help="Use Test-Time Augmentation")
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    model, device = load_model(args.checkpoint, args.model_name)
    print(f"Model loaded on {device}")

    image_path = Path(args.image)
    if image_path.is_dir():
        results = predict_batch(image_path, model, device, args.image_size, args.tta)
        for r in results:
            print(f"{r['filename']}: {r['artist']} ({r['confidence']*100:.1f}%)")
    else:
        fn = predict_with_tta if args.tta else predict_single
        result = fn(image_path, model, device, args.image_size)
        print(f"\nPrediction: {result['artist']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Probabilities:")
        for artist, prob in result["probabilities"].items():
            print(f"  {artist.capitalize():12s}: {prob*100:.1f}%")


if __name__ == "__main__":
    main()
