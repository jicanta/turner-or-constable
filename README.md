# Turner or Constable?

A binary image classifier that distinguishes paintings by **J.M.W. Turner** from those by **John Constable**. Both were early-19th-century British landscape painters working simultaneously — the model can't rely on subject matter and has to learn actual stylistic differences: Turner's atmospheric haze and luminous, almost dissolved light versus Constable's grounded palette and detailed naturalistic foliage.

The dataset is small (~317 images), the classes are imbalanced 2.4:1, and the visual boundary is genuinely ambiguous in many cases. That's the interesting part.

---

## Results

Trained on a ResNet50 backbone, CPU-only (Intel i5-1135G7):

| Metric | Score |
|---|---|
| Test accuracy | 68.75% |
| Test AUC-ROC | 0.889 |
| F1 — Turner | 0.727 |
| F1 — Constable | 0.634 |
| Best val AUC (training) | 0.941 |

Test set: 48 held-out images (34 Turner, 14 Constable). Constable recall was strong — 13 of 14 correct — while Turner proved harder due to his wider range across periods and subjects.

---

## Project Structure

```
turner-or-constable/
├── data/
│   ├── raw/                        # Downloaded originals (turner/ + constable/)
│   ├── processed/                  # Resized and split images (train/ val/ test/)
│   ├── metadata.csv                # Per-image split, artist, paths, pHash
│   └── metadata_raw.csv            # Pre-split download log
├── src/
│   ├── data/
│   │   ├── scrape_wikiart.py       # Direct WikiArt JSON API scraper
│   │   ├── download.py             # HuggingFace huggan/wikiart alternative
│   │   ├── preprocess.py           # Quality filter, dedup, resize, stratified split
│   │   └── dataset.py              # PyTorch Dataset + albumentations pipelines
│   ├── models/
│   │   └── classifier.py           # ArtClassifier, EnsembleModel, differential LR groups
│   ├── training/
│   │   ├── train.py                # 3-phase training loop
│   │   ├── losses.py               # Label smoothing CE + Focal loss
│   │   └── metrics.py              # AUC-ROC, F1, confusion matrix, MetricsAccumulator
│   └── inference/
│       └── predict.py              # Single/batch/TTA prediction + Grad-CAM
├── configs/
│   ├── cpu_resnet50.yaml           # CPU-optimized config (used for the training run)
│   ├── efficientnet.yaml           # EfficientNet-B4 config
│   └── swin.yaml                   # Swin-Transformer-Base config
├── checkpoints/                    # Saved model weights (best.pth + history.json)
├── app.py                          # Gradio demo
├── evaluate_test.py                # Final held-out test set evaluation
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Training was done on CPU; a GPU will speed up Phases 2–3 significantly.

---

## Pipeline

### 1. Get the data

Two options, depending on what you have access to:

**Option A — WikiArt scraper** (direct, no HuggingFace account needed):
```bash
python src/data/scrape_wikiart.py --max-per-artist 400 --output-dir data/raw
```
Fetches from WikiArt's public JSON API. Polite rate-limiting (150ms between requests) is built in.

**Option B — HuggingFace dataset** (slower first run, more complete):
```bash
python src/data/download.py --output-dir data/raw --max-per-artist 2000
```
Streams `huggan/wikiart` and filters for Turner and Constable entries. Requires the `datasets` package.

Both write raw images to `data/raw/{artist}/` and a metadata CSV to `data/metadata_raw.csv`.

### 2. Preprocess

Quality-filters (drops images under 224px), deduplicates using perceptual hashing, resizes to 512px on the shorter side, then does a stratified 70/15/15 train/val/test split:

```bash
python src/data/preprocess.py
```

Optional flags: `--target-size 512 --min-size 224 --phash-threshold 10 --seed 42`

### 3. Train

```bash
python src/training/train.py --config configs/cpu_resnet50.yaml
```

Training runs in three phases:

| Phase | Epochs | What trains | Learning rate |
|---|---|---|---|
| 1 — Warm-up | 8 | Classifier head only (backbone frozen) | 5e-4 |
| 2 — Fine-tune | 30 | Full network, differential LRs | head: 2e-4 / late stages: 5e-5 / early: 1e-5 |
| 3 — Polish | 10 | Same as Phase 2 | All × 0.1 |

Early stopping watches val AUC-ROC with patience=12. The best checkpoint is reloaded before Phase 3. Checkpoints go to `checkpoints/<model-name>/best.pth`.

To resume from a checkpoint:
```bash
python src/training/train.py --config configs/cpu_resnet50.yaml --resume checkpoints/resnet50/best.pth
```

### 4. Evaluate

```bash
python evaluate_test.py \
    --checkpoint checkpoints/resnet50/best.pth \
    --model-name resnet50 \
    --data-dir data/processed \
    --image-size 224
```

Prints accuracy, AUC-ROC, per-class F1, and the full confusion matrix.

### 5. Run the demo

```bash
python app.py --checkpoint checkpoints/resnet50/best.pth --model-name resnet50 --image-size 224
```

Opens a Gradio interface at `http://localhost:7860`. Upload any painting to get a prediction, confidence scores, and an optional Grad-CAM heatmap showing which regions drove the classification.

Add `--share` to get a public Gradio link.

---

## Model Architecture

Built on [timm](https://github.com/huggingface/pytorch-image-models). `ArtClassifier` wraps any timm backbone with a custom two-layer head:

```
backbone (ResNet50 / EfficientNet-B4 / Swin-Base)
    → Global average pool
    → Linear(features → head_hidden_dim)  [256 for ResNet50, 512 for others]
    → GELU + Dropout(0.4)
    → Linear(head_hidden_dim → 2)
```

Three backbones are configured out of the box:

- **ResNet50** (`cpu_resnet50.yaml`) — fast on CPU, reasonable baseline
- **EfficientNet-B4** (`efficientnet.yaml`) — better convolutional baseline
- **Swin-Transformer-Base** (`swin.yaml`) — best for fine-grained style classification; shifted-window attention captures both local brushstroke texture and global composition; needs a GPU to be practical

The differential LR split for fine-tuning (Phase 2) is architecture-aware: for Swin it uses `patch_embed` + first two stages as "early" and stages 2–3 + norm as "late". ResNet and EfficientNet fall back to a layer-prefix heuristic.

---

## Training Details

- **Optimizer:** AdamW, `weight_decay=0.05` (CPU config) / `0.01` (GPU configs)
- **Scheduler:** CosineAnnealingWarmRestarts (`T_0=10, T_mult=2`)
- **Gradient accumulation:** effective batch size 32 (batch 16 × accumulate 2 on CPU)
- **Regularization:** label smoothing 0.1, MixUp alpha=0.4, gradient clipping max_norm=1.0
- **Class imbalance:** WeightedRandomSampler + class-weighted loss (Turner is ~2.4× more common)
- **Augmentation (train):** random crop, horizontal flip, rotation ±10°, perspective distortion, color jitter, Gaussian blur, occasional grayscale — via [albumentations](https://albumentations.ai/)
- **Augmentation (val/test):** deterministic center crop only

---

## Inference

The `predict.py` module exposes four functions:

```python
from src.inference.predict import load_model, predict_single, predict_with_tta, predict_batch, generate_gradcam

model, device = load_model("checkpoints/resnet50/best.pth", "resnet50")

# Single image
result = predict_single("painting.jpg", model, device, image_size=224)
# → {"artist": "Turner", "confidence": 0.87, "probabilities": {"turner": 0.87, "constable": 0.13}}

# TTA — averages softmax over 5 augmented views
result = predict_with_tta("painting.jpg", model, device, image_size=224)

# Batch — runs over all images in a directory
results = predict_batch("some_dir/", model, device, image_size=224, use_tta=False)

# Grad-CAM heatmap
rgb, overlay = generate_gradcam("painting.jpg", model, device, target_class=0, image_size=224)
```

Or from the command line:
```bash
python src/inference/predict.py \
    --checkpoint checkpoints/resnet50/best.pth \
    --model-name resnet50 \
    --image path/to/painting.jpg \
    --image-size 224
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Training and inference |
| `timm` | Pretrained backbones |
| `albumentations` | Augmentation pipeline |
| `pytorch-grad-cam` | Grad-CAM heatmap visualization |
| `Pillow` + `imagehash` | Image loading and perceptual deduplication |
| `scikit-learn` | Stratified splits, AUC-ROC, F1 |
| `gradio` | Interactive demo |
| `datasets` | HuggingFace download path (optional) |
| `wandb` | Experiment tracking (disabled by default) |
| `requests` + `tqdm` | WikiArt scraping |
| `PyYAML` | Config parsing |
| `pandas` | Metadata CSVs |
