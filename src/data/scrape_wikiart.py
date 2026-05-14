"""
Fast targeted scraper: downloads Turner and Constable paintings directly
from WikiArt's public JSON API — only fetches what we need.

Usage:
    python src/data/scrape_wikiart.py
    python src/data/scrape_wikiart.py --max-per-artist 300 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

ARTISTS = {
    "turner": {
        "slug": "william-turner",
        "display": "J.M.W. Turner",
    },
    "constable": {
        "slug": "john-constable",
        "display": "John Constable",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.wikiart.org/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_painting_list(artist_slug: str) -> list[dict]:
    """Fetch the full painting list for an artist from WikiArt's JSON API."""
    url = "https://www.wikiart.org/en/App/Painting/PaintingsByArtist"
    try:
        resp = SESSION.get(url, params={"artistUrl": artist_slug, "json": 2}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning: API fetch failed: {e}")
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("Paintings", data.get("paintings", []))
    return []


def get_image_url(painting: dict) -> str | None:
    """Extract the best available image URL from a painting dict."""
    raw = painting.get("image") or ""
    if not raw or not raw.startswith("http"):
        return None
    # Strip existing quality suffix and request !Large for a good balance
    base = raw.split("!")[0]
    return base + "!Large.jpg"


def download_image(url: str, dest: Path, min_size: int = 224) -> bool:
    """Download and validate a single image. Returns True on success."""
    try:
        resp = SESSION.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")
        w, h = img.size
        if w < min_size or h < min_size:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=92)
        return True
    except Exception:
        return False


def scrape_artist(
    canonical: str,
    artist_info: dict,
    output_dir: Path,
    max_images: int,
    min_size: int = 224,
) -> list[dict]:
    artist_dir = output_dir / canonical
    artist_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{artist_info['display']}] Fetching painting list...")
    paintings = get_painting_list(artist_info["slug"])
    print(f"  Found {len(paintings)} paintings in catalogue")

    metadata = []
    downloaded = 0

    for i, painting in enumerate(tqdm(paintings, desc=f"Downloading {canonical}")):
        if downloaded >= max_images:
            break

        url = get_image_url(painting)
        if not url:
            continue

        filename = f"{canonical}_{downloaded:05d}.jpg"
        dest = artist_dir / filename

        if dest.exists():
            downloaded += 1
            metadata.append({
                "filename": filename,
                "artist": canonical,
                "source": "wikiart.org",
                "title": painting.get("title", ""),
                "year": painting.get("year", ""),
                "url": url,
            })
            continue

        success = download_image(url, dest, min_size)
        if success:
            downloaded += 1
            metadata.append({
                "filename": filename,
                "artist": canonical,
                "source": "wikiart.org",
                "title": painting.get("title", ""),
                "year": painting.get("year", ""),
                "url": url,
            })
        time.sleep(0.15)  # be polite

    print(f"  Downloaded {downloaded} images for {artist_info['display']}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Scrape Turner/Constable from WikiArt")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-per-artist", type=int, default=400)
    parser.add_argument("--min-size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_metadata = []

    for canonical, info in ARTISTS.items():
        meta = scrape_artist(canonical, info, output_dir, args.max_per_artist, args.min_size)
        all_metadata.extend(meta)

    # Save metadata
    csv_path = output_dir.parent / "metadata_raw.csv"
    if all_metadata:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)

    print(f"\n=== Download Summary ===")
    for canonical in ARTISTS:
        count = sum(1 for m in all_metadata if m["artist"] == canonical)
        print(f"  {canonical:12s}: {count} images")
    print(f"  {'TOTAL':12s}: {len(all_metadata)} images")
    print(f"\nMetadata saved to {csv_path}")
    print("Next step: python src/data/preprocess.py")


if __name__ == "__main__":
    main()
