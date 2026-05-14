"""
Download Turner and Constable paintings from HuggingFace huggan/wikiart dataset.
Saves raw images to data/raw/{artist}/ and produces data/metadata_raw.csv.

Usage:
    python src/data/download.py
    python src/data/download.py --output-dir data/raw --max-per-artist 2000
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


# WikiArt artist labels as they appear in the huggan/wikiart dataset
ARTIST_LABELS = {
    "turner": ["joseph-mallord-william-turner", "j-m-w-turner", "turner"],
    "constable": ["john-constable", "constable"],
}

# Canonical folder names
ARTISTS = ["turner", "constable"]


def _find_artist_key(artist_field: str) -> str | None:
    """Map a raw WikiArt artist string to one of our canonical labels."""
    normalized = artist_field.lower().replace(" ", "-").replace("_", "-")
    for canonical, variants in ARTIST_LABELS.items():
        if any(v in normalized for v in variants):
            return canonical
    return None


def download_wikiart(output_dir: Path, max_per_artist: int) -> list[dict]:
    """Stream huggan/wikiart and save relevant images.

    Returns a list of metadata dicts with keys: filename, artist, source, width, height.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install the 'datasets' package: pip install datasets")

    print("Loading huggan/wikiart from HuggingFace (this may take a while on first run)...")
    ds = load_dataset("huggan/wikiart", split="train", trust_remote_code=True)

    # Inspect available columns to find the artist field
    sample = ds[0]
    print(f"Dataset columns: {list(sample.keys())}")

    # Typical columns: 'image', 'artist', 'style', 'genre'
    artist_col = "artist"
    if artist_col not in sample:
        candidate_cols = [c for c in sample.keys() if "artist" in c.lower()]
        if not candidate_cols:
            sys.exit(f"Cannot find artist column. Available: {list(sample.keys())}")
        artist_col = candidate_cols[0]
        print(f"Using column '{artist_col}' as artist identifier.")

    counts = {a: 0 for a in ARTISTS}
    metadata = []

    for idx, row in enumerate(tqdm(ds, desc="Scanning WikiArt")):
        artist_raw = str(row[artist_col])
        canonical = _find_artist_key(artist_raw)
        if canonical is None:
            continue
        if counts[canonical] >= max_per_artist:
            continue

        img = row["image"]
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                continue

        # Skip very small images (< 224px on any side)
        w, h = img.size
        if w < 224 or h < 224:
            continue

        artist_dir = output_dir / canonical
        artist_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{canonical}_{counts[canonical]:05d}.jpg"
        filepath = artist_dir / filename
        img.convert("RGB").save(filepath, "JPEG", quality=95)

        metadata.append({
            "filename": filename,
            "artist": canonical,
            "source": "huggan/wikiart",
            "width": w,
            "height": h,
            "original_index": idx,
            "original_artist_label": artist_raw,
        })
        counts[canonical] += 1

        # Early exit if both artists are saturated
        if all(counts[a] >= max_per_artist for a in ARTISTS):
            print("\nReached max_per_artist for all artists, stopping early.")
            break

    return metadata


def save_metadata(metadata: list[dict], output_dir: Path) -> None:
    csv_path = output_dir.parent / "metadata_raw.csv"
    if not metadata:
        print("WARNING: No images were downloaded. Check artist label matching.")
        return
    fieldnames = list(metadata[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Metadata saved to {csv_path}")


def print_summary(metadata: list[dict]) -> None:
    counts = {}
    for row in metadata:
        counts[row["artist"]] = counts.get(row["artist"], 0) + 1
    print("\n=== Download Summary ===")
    for artist, count in sorted(counts.items()):
        print(f"  {artist:12s}: {count} images")
    print(f"  {'TOTAL':12s}: {sum(counts.values())} images")


def main():
    parser = argparse.ArgumentParser(description="Download Turner/Constable images from WikiArt")
    parser.add_argument("--output-dir", default="data/raw", help="Directory to save raw images")
    parser.add_argument("--max-per-artist", type=int, default=2000,
                        help="Maximum images to download per artist (default: 2000)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = download_wikiart(output_dir, args.max_per_artist)
    save_metadata(metadata, output_dir)
    print_summary(metadata)


if __name__ == "__main__":
    main()
