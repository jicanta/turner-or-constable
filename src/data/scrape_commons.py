"""
Scrape Turner and Constable paintings from Wikimedia Commons.

Traverses the full category tree recursively (by museum / location / genre /
title subcategories hold most of the files — the top-level category alone has
only a few hundred), filters out engravings, prints, and crops, and downloads
1024px thumbnails.

Root categories:
  - Paintings by Joseph Mallord William Turner
  - Paintings by John Constable

Usage:
    python src/data/scrape_commons.py
    python src/data/scrape_commons.py --max-per-artist 1200 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

ARTISTS = {
    "turner": {
        "display": "J.M.W. Turner",
        "categories": [
            "Paintings by Joseph Mallord William Turner",
            "Watercolor paintings by Joseph Mallord William Turner",
        ],
    },
    "constable": {
        "display": "John Constable",
        "categories": [
            "Paintings by John Constable",
        ],
    },
}

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "turner-or-constable-research/2.0 (ignaciocantarella@gmail.com) python-requests",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Files whose titles match any of these are not (photos of) original paintings:
# engravings/prints after the artist, book plates, crops/details of other files.
BAD_TITLE_PATTERNS = re.compile(
    r"engrav|etch(ing|ed)|mezzotint|aquatint|lithograph|stipple|woodcut"
    r"|\bprint\b|\bplate\b|\bafter\s+(j\.?\s?m\.?\s?w\.?|joseph|william\s+turner|john\s+constable|turner|constable)\b"
    r"|\bcropped\b|\bcrop\b|\bdetail\b|\bframe[sd]?\b|\bx[- ]?ray\b|\binfrared\b"
    r"|photograph of|book|page\s+\d|title[- ]page|frontispiece",
    re.IGNORECASE,
)

# Subcategories that would drag in non-painting material.
BAD_CATEGORY_PATTERNS = re.compile(
    r"engrav|prints|drawings|sketchbook|etchings|mezzotints|reproductions"
    r"|liber studiorum|details of|signature|grave|memorial|statue|portrait photographs",
    re.IGNORECASE,
)

VALID_EXT = re.compile(r"\.(jpe?g|png|tiff?)$", re.IGNORECASE)


def api_get(params: dict, retries: int = 4) -> dict | None:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = SESSION.get(COMMONS_API, params={**params, "format": "json", "maxlag": 5}, timeout=30)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", delay)))
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and data["error"].get("code") == "maxlag":
                time.sleep(delay)
                delay *= 2
                continue
            return data
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    return None


def collect_files_recursive(root_categories: list[str], max_depth: int = 6) -> set[str]:
    """BFS over the category tree, returning the set of file titles found."""
    files: set[str] = set()
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(c, 0) for c in root_categories]

    pbar = tqdm(desc="Walking category tree", unit="cat")
    while queue:
        category, depth = queue.pop(0)
        if category in visited:
            continue
        visited.add(category)
        pbar.update(1)
        pbar.set_postfix(files=len(files), queued=len(queue))

        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 500,
            "cmtype": "file|subcat",
        }
        while True:
            data = api_get(params)
            if data is None:
                break
            for m in data.get("query", {}).get("categorymembers", []):
                title = m["title"]
                if title.startswith("Category:"):
                    sub = title[len("Category:"):]
                    if depth < max_depth and not BAD_CATEGORY_PATTERNS.search(sub):
                        queue.append((sub, depth + 1))
                elif title.startswith("File:"):
                    name = title[len("File:"):]
                    if VALID_EXT.search(name) and not BAD_TITLE_PATTERNS.search(name):
                        files.add(title)
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            params["cmcontinue"] = cont
        time.sleep(0.05)

    pbar.close()
    print(f"  Visited {len(visited)} categories, kept {len(files)} candidate files")
    return files


def get_image_urls(titles: list[str], batch_size: int = 50, thumb_width: int = 1024) -> dict[str, str]:
    """Fetch thumbnail URLs for file titles (batched). Skips tiny images."""
    url_map: dict[str, str] = {}
    for i in tqdm(range(0, len(titles), batch_size), desc="Fetching image URLs"):
        batch = titles[i : i + batch_size]
        data = api_get({
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size",
            "iiurlwidth": thumb_width,
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            infos = page.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
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
            resp = SESSION.get(url, timeout=45)
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", delay)))
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
    seed: int = 42,
) -> list[dict]:
    artist_dir = output_dir / canonical
    artist_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{artist_info['display']}] Walking category tree...")
    titles = sorted(collect_files_recursive(artist_info["categories"]))
    if not titles:
        return []

    # Deterministic shuffle so a cap doesn't bias toward alphabetical order
    random.Random(seed).shuffle(titles)

    url_map = get_image_urls(titles)
    print(f"  Got URLs for {len(url_map)} images")

    metadata = []
    downloaded = 0
    failed = 0

    for title, url in tqdm(url_map.items(), desc=f"Downloading {canonical}"):
        if downloaded >= max_images:
            break

        safe_name = title.replace("File:", "").replace("/", "_")
        # Stable content-independent name so re-runs resume cleanly
        digest = hashlib.md5(title.encode()).hexdigest()[:12]
        filename = f"commons_{canonical}_{digest}.jpg"
        dest = artist_dir / filename

        if dest.exists() or download_image(url, dest, min_size):
            downloaded += 1
            metadata.append({
                "filename": filename,
                "artist": canonical,
                "source": "commons.wikimedia.org",
                "title": safe_name[:120],
                "url": url,
            })
        else:
            failed += 1
        time.sleep(0.15)

    print(f"  Downloaded {downloaded} images for {artist_info['display']} ({failed} failed)")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Scrape Turner/Constable from Wikimedia Commons")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-per-artist", type=int, default=1200)
    parser.add_argument("--min-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_metadata = []

    for canonical, info in ARTISTS.items():
        meta = scrape_artist(canonical, info, output_dir, args.max_per_artist, args.min_size, args.seed)
        all_metadata.extend(meta)

    csv_path = output_dir.parent / "metadata_commons.csv"
    if all_metadata:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)

    print("\n=== Download Summary ===")
    for canonical in ARTISTS:
        count = sum(1 for m in all_metadata if m["artist"] == canonical)
        print(f"  {canonical:12s}: {count} images")
    print(f"  {'TOTAL':12s}: {len(all_metadata)}")
    print(f"\nMetadata saved to {csv_path}")


if __name__ == "__main__":
    main()
