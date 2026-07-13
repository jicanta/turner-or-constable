# Turner or Constable?

A binary image classifier that distinguishes paintings by **J.M.W. Turner** from those by **John Constable**. Both were early-19th-century British landscape painters working simultaneously — the model can't rely on subject matter and has to learn actual stylistic differences: Turner's atmospheric haze and luminous, almost dissolved light versus Constable's grounded palette and detailed naturalistic foliage.

The dataset is 2,079 deduplicated paintings (1,089 Turner / 990 Constable) scraped from WikiArt and the full Wikimedia Commons category trees, with a leakage-audited train/val/test split, and the visual boundary is genuinely ambiguous in many cases. That's the interesting part.

---

## Results

Best model: a **DINOv2 ViT-S/14 feature probe** — frozen self-supervised backbone, classification head trained on cached features (see `src/training/probe.py`). Trained CPU-only (Intel i5-1135G7) in ~40 minutes, most of which is one-time feature extraction. All numbers below are on the leakage-audited split (see next section).

| Metric | Score |
|---|---|
| Test accuracy (flip-TTA) | **91.64%** |
| Test AUC-ROC | 0.975 |
| F1 — Turner | 0.920 |
| F1 — Constable | 0.912 |
| Best val AUC (training) | 0.983 |

Test set: 311 held-out images (163 Turner, 148 Constable). Confusion matrix (flip-TTA): 150/163 Turner and 135/148 Constable correct.

For scale: the previous release — the same kind of ResNet50 fine-tune on the original ~338-image dataset — scored 66.7% accuracy / 0.898 AUC on its 51-image test set. Most of the jump comes from ~6x more training data with near-balanced classes, plus the switch to frozen DINOv2 features, which can't overfit a dataset this size the way a fully fine-tuned network can.

### Leakage audit

Wikimedia Commons often hosts several *different photographs of the same painting* (a framed gallery shot and a direct reproduction, different color grading), and Constable in particular repainted the same composition multiple times. pHash deduplication does not catch these, so a naive random split lets near-duplicates straddle train/test and inflate test metrics.

`src/data/grouped_split.py` audits and fixes this: it computes a DINOv2 feature per image, unions images with cosine similarity ≥ 0.92 into groups (threshold picked by visually inspecting ranked cross-split pairs — above it, pairs are the same painting or near-identical variants; below it, distinct works), drops groups whose members carry both artists' labels (in practice: one Flickr photo of a National Gallery room that sat in both category trees), and re-splits 70/15/15 at group granularity so every near-duplicate cluster lands entirely in one split. The audit report is written to `data/leakage_audit.json`.

Measured effect: the probe scored **94.6%** accuracy / 0.986 AUC on the naive split vs. **91.6%** / 0.975 on the grouped split — i.e. cross-split near-duplicates were worth about 3 accuracy points. The number above is the honest one.

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
│   │   ├── scrape_commons.py       # Wikimedia Commons scraper (recursive category walk)
│   │   ├── download.py             # HuggingFace huggan/wikiart alternative
│   │   ├── preprocess.py           # Quality filter, dedup, resize, stratified split
│   │   ├── grouped_split.py        # Leakage audit + near-duplicate-safe re-split
│   │   └── dataset.py              # PyTorch Dataset + albumentations pipelines
│   ├── models/
│   │   └── classifier.py           # ArtClassifier, EnsembleModel, differential LR groups
│   ├── training/
│   │   ├── train.py                # 3-phase fine-tuning loop
│   │   ├── probe.py                # DINOv2 frozen-feature probe (best model)
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

Three options, depending on what you have access to:

**Option A — Wikimedia Commons scraper** (recommended; where most of the current dataset comes from):
```bash
python src/data/scrape_commons.py --max-per-artist 1200 --output-dir data/raw
```
Recursively walks the full `Paintings by …` category trees (by-museum / by-location / by-title subcategories hold most of the files), filters out engravings, prints, and cropped details by title, and downloads 1024px thumbnails with polite rate-limiting.

**Option B — WikiArt scraper** (direct, no HuggingFace account needed):
```bash
python src/data/scrape_wikiart.py --max-per-artist 400 --output-dir data/raw
```
Fetches from WikiArt's public JSON API. Polite rate-limiting (150ms between requests) is built in.

**Option C — HuggingFace dataset** (slower first run, more complete):
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

Then make the split leakage-safe (groups near-duplicate photos/versions of the same painting onto one side of the split — see the Leakage audit section):

```bash
python src/data/grouped_split.py --threshold 0.92 --dry-run   # inspect first
python src/data/grouped_split.py --threshold 0.92
```

### 3. Train

**Option A — DINOv2 feature probe** (recommended; best accuracy, CPU-friendly):
```bash
python src/training/probe.py --train-views 4 --image-size 336
```
Extracts frozen DINOv2 ViT-S/14 features once (cached to `checkpoints/dinov2_probe/features.npz`), sweeps head hyperparameters on val AUC, evaluates on test with flip-TTA, and exports a standard `ArtClassifier` checkpoint to `checkpoints/dinov2_probe/best.pth` — fully compatible with `evaluate_test.py`, `predict.py`, and `app.py`.

**Option B — full fine-tuning**:
```bash
python src/training/train.py --config configs/cpu_resnet50.yaml
```

Fine-tuning runs in three phases:

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
    --checkpoint checkpoints/dinov2_probe/best.pth \
    --model-name vit_small_patch14_reg4_dinov2.lvd142m \
    --data-dir data/processed \
    --image-size 336
```

Prints accuracy, AUC-ROC, per-class F1, and the full confusion matrix.

### 5. Run the demo

```bash
python app.py --checkpoint checkpoints/dinov2_probe/best.pth \
    --model-name vit_small_patch14_reg4_dinov2.lvd142m --image-size 336
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

Backbones in use:

- **DINOv2 ViT-S/14** (`probe.py`, default) — frozen self-supervised features + trained head; best accuracy and can't overfit a dataset this size. ViT backbones are created with `dynamic_img_size=True` so any input resolution works.
- **ResNet50** (`cpu_resnet50.yaml`) — fast fine-tuning baseline on CPU
- **EfficientNet-B4** (`efficientnet.yaml`) — better convolutional baseline
- **Swin-Transformer-Base** (`swin.yaml`) — fine-grained fine-tuning option; needs a GPU to be practical

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
