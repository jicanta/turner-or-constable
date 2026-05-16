"""
Scrape Turner and Constable paintings from Wikimedia Commons.

Categories used:
  - Paintings by Joseph Mallord William Turner
  - Paintings by John Constable

Usage:
    python src/data/scrape_commons.py
    python src/data/scrape_commons.py --max-per-artist 400 --output-dir data/raw
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
        "display": "J.M.W. Turner",
        "category": "Paintings by Joseph Mallord William Turner",
    },
    "constable": {
        "display": "John Constable",
        "category": "Paintings by John Constable",
    },
}

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://commons.wikimedia.org/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def get_category_files(category: str) -> list[str]:
    """Return all file titles in a Commons category (handles pagination)."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 500,
        "cmtype": "file",
        "format": "json",
    }
    while True:
        try:
            resp = SESSION.get(COMMONS_API, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning: category fetch failed: {e}")
            break

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont

    return titles


def get_image_urls(titles: list[str], batch_size: int = 50, thumb_width: int = 800) -> dict[str, str]:
    """Fetch thumbnail image URLs for a list of file titles (batched).

    Uses 800px thumbnails — large enough for 224px training crops,
    small enough to download quickly and stay within CDN rate limits.
    """
    url_map: dict[str, str] = {}

    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": thumb_width,
            "format": "json",
        }
        try:
            resp = SESSION.get(COMMONS_API, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning: imageinfo fetch failed: {e}")
            continue

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            infos = page.get("imageinfo", [])
            if infos:
                info = infos[0]
                # Use thumb URL (strip UTM tracking params that cause 403)
                raw_thumb = info.get("thumburl", "") or info.get("url", "")
                url = raw_thumb.split("?")[0] if raw_thumb else ""
                tw = info.get("thumbwidth") or info.get("width", 0)
                th = info.get("thumbheight") or info.get("height", 0)
                if url and tw >= 224 and th >= 224:
                    url_map[title] = url

        time.sleep(0.1)

    return url_map


def download_image(url: str, dest: Path, min_size: int = 224, retries: int = 3) -> bool:
    delay = 3.0
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", delay))
                time.sleep(wait)
                delay *= 2
                continue
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
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return False


def scrape_artist(
    canonical: str,
    artist_info: dict,
    output_dir: Path,
    max_images: int,
    min_size: int = 224,
    existing_count: int = 0,
) -> list[dict]:
    artist_dir = output_dir / canonical
    artist_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{artist_info['display']}] Fetching file list from Commons...")
    titles = get_category_files(artist_info["category"])
    print(f"  Found {len(titles)} files in category")

    if not titles:
        return []

    print(f"  Fetching image URLs...")
    url_map = get_image_urls(titles)
    print(f"  Got URLs for {len(url_map)} images")

    metadata = []
    downloaded = 0

    for title, url in tqdm(url_map.items(), desc=f"Downloading {canonical}"):
        if downloaded >= max_images:
            break

        # Use a commons_ prefix to distinguish from wikiart files
        safe_name = title.replace("File:", "").replace("/", "_")[:80]
        filename = f"commons_{canonical}_{downloaded:05d}.jpg"
        dest = artist_dir / filename

        if dest.exists():
            downloaded += 1
            metadata.append({
                "filename": filename,
                "artist": canonical,
                "source": "commons.wikimedia.org",
                "title": safe_name,
                "url": url,
            })
            continue

        success = download_image(url, dest, min_size)
        if success:
            downloaded += 1
            metadata.append({
                "filename": filename,
                "artist": canonical,
                "source": "commons.wikimedia.org",
                "title": safe_name,
                "url": url,
            })
        time.sleep(0.5)

    print(f"  Downloaded {downloaded} images for {artist_info['display']}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Scrape Turner/Constable from Wikimedia Commons")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-per-artist", type=int, default=300)
    parser.add_argument("--min-size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_metadata = []

    for canonical, info in ARTISTS.items():
        # Count existing files so we don't re-number from 0
        existing = list((output_dir / canonical).glob("commons_*.jpg")) if (output_dir / canonical).exists() else []
        meta = scrape_artist(
            canonical, info, output_dir,
            args.max_per_artist, args.min_size,
            existing_count=len(existing),
        )
        all_metadata.extend(meta)

    csv_path = output_dir.parent / "metadata_commons.csv"
    if all_metadata:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)

    print(f"\n=== Download Summary ===")
    for canonical in ARTISTS:
        count = sum(1 for m in all_metadata if m["artist"] == canonical)
        print(f"  {canonical:12s}: {count} images")
    print(f"  {'TOTAL':12s}: {len(all_metadata)}")
    print(f"\nMetadata saved to {csv_path}")


if __name__ == "__main__":
    main()
