"""
Preprocess raw Turner/Constable images:
  1. Quality filter  — drop images smaller than min_size px on any side
  2. Perceptual dedup — remove near-duplicates using pHash (threshold configurable)
  3. Resize          — resize so shorter side == target_size, preserving aspect ratio
  4. Stratified split — 70 / 15 / 15 train / val / test
  5. Copy to data/processed/{split}/{artist}/
  6. Write data/metadata.csv

Usage:
    python src/data/preprocess.py
    python src/data/preprocess.py --raw-dir data/raw --processed-dir data/processed \
        --target-size 512 --min-size 224 --phash-threshold 10 --seed 42
"""

import argparse
import csv
import shutil
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ARTISTS = ["turner", "constable"]
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def load_raw_images(raw_dir: Path) -> list[dict]:
    """Collect all image paths from raw_dir/{artist}/ subdirectories."""
    records = []
    for artist in ARTISTS:
        artist_dir = raw_dir / artist
        if not artist_dir.exists():
            print(f"WARNING: {artist_dir} not found — skipping {artist}")
            continue
        for p in sorted(artist_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                records.append({"path": p, "artist": artist, "filename": p.name})
    return records


def quality_filter(records: list[dict], min_size: int) -> list[dict]:
    """Drop images smaller than min_size on any side or that cannot be opened."""
    kept, dropped = [], 0
    for rec in tqdm(records, desc="Quality filter"):
        try:
            with Image.open(rec["path"]) as img:
                w, h = img.size
            if w >= min_size and h >= min_size:
                rec["width"] = w
                rec["height"] = h
                kept.append(rec)
            else:
                dropped += 1
        except Exception:
            dropped += 1
    print(f"Quality filter: kept {len(kept)}, dropped {dropped}")
    return kept


def dedup(records: list[dict], threshold: int) -> list[dict]:
    """Remove near-duplicates using perceptual hash within each artist class."""
    kept = []
    for artist in ARTISTS:
        artist_records = [r for r in records if r["artist"] == artist]
        seen_hashes: list[imagehash.ImageHash] = []
        artist_kept = []
        for rec in tqdm(artist_records, desc=f"Dedup {artist}"):
            try:
                with Image.open(rec["path"]) as img:
                    h = imagehash.phash(img)
            except Exception:
                continue
            if all(abs(h - s) > threshold for s in seen_hashes):
                seen_hashes.append(h)
                rec["phash"] = str(h)
                artist_kept.append(rec)
        print(f"  {artist}: {len(artist_records)} → {len(artist_kept)} after dedup")
        kept.extend(artist_kept)
    return kept


def resize_image(src: Path, dst: Path, target_size: int) -> None:
    """Resize so shorter side == target_size, save as high-quality JPEG."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w <= h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=95)


def split_records(records: list[dict], seed: int) -> list[dict]:
    """Add a 'split' field (train/val/test) using stratified splitting."""
    df = pd.DataFrame(records)
    labels = df["artist"].tolist()

    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    # First split off test
    train_val_idx, test_idx = train_test_split(
        range(len(df)), test_size=test_ratio, stratify=labels, random_state=seed
    )
    # Then split train_val into train and val
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        stratify=[labels[i] for i in train_val_idx],
        random_state=seed,
    )

    split_map = {}
    for i in train_idx:
        split_map[i] = "train"
    for i in val_idx:
        split_map[i] = "val"
    for i in test_idx:
        split_map[i] = "test"

    for i, rec in enumerate(records):
        rec["split"] = split_map[i]
    return records


def copy_to_processed(records: list[dict], processed_dir: Path, target_size: int) -> list[dict]:
    """Resize and copy images to processed/{split}/{artist}/."""
    updated = []
    for rec in tqdm(records, desc="Resizing and copying"):
        dst_filename = rec["filename"]
        dst = processed_dir / rec["split"] / rec["artist"] / dst_filename
        resize_image(rec["path"], dst, target_size)
        updated.append({
            "filename": dst_filename,
            "artist": rec["artist"],
            "split": rec["split"],
            "source_path": str(rec["path"]),
            "processed_path": str(dst),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "phash": rec.get("phash", ""),
        })
    return updated


def save_metadata(records: list[dict], out_path: Path) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nMetadata saved to {out_path}")

    print("\n=== Split Summary ===")
    for split in SPLITS:
        split_df = df[df["split"] == split]
        for artist in ARTISTS:
            count = len(split_df[split_df["artist"] == artist])
            print(f"  {split:6s} / {artist:12s}: {count}")
    print(f"  TOTAL: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Turner/Constable images")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--metadata-out", default="data/metadata.csv")
    parser.add_argument("--target-size", type=int, default=512,
                        help="Shorter side target after resize (default: 512)")
    parser.add_argument("--min-size", type=int, default=224,
                        help="Minimum px on any side to keep an image (default: 224)")
    parser.add_argument("--phash-threshold", type=int, default=10,
                        help="pHash distance threshold for dedup (default: 10; lower = stricter)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading raw images...")
    records = load_raw_images(raw_dir)
    print(f"Found {len(records)} raw images")

    print("\nStep 2: Quality filter...")
    records = quality_filter(records, args.min_size)

    print("\nStep 3: Perceptual deduplication...")
    records = dedup(records, args.phash_threshold)

    print("\nStep 4: Stratified split...")
    records = split_records(records, args.seed)

    print("\nStep 5: Resize and copy to processed/...")
    final_records = copy_to_processed(records, processed_dir, args.target_size)

    print("\nStep 6: Saving metadata...")
    save_metadata(final_records, Path(args.metadata_out))

    print("\nDone. Next step: python src/training/train.py --config configs/cpu_resnet50.yaml")


if __name__ == "__main__":
    main()
