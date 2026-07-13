"""
Leakage-safe re-split of data/processed using DINOv2 feature similarity.

pHash dedup (preprocess.py) catches identical files but misses different
*photographs of the same painting* — framed gallery shots vs. direct
reproductions, different color grading — and the artists' own repainted
versions of one composition. If those straddle train/test, test metrics are
inflated.

This script:
  1. Loads (or computes) a DINOv2 feature per processed image
  2. Union-finds images with cosine similarity >= --threshold into groups
  3. Drops groups containing both artists (ambiguous/mislabeled files)
  4. Re-splits 70/15/15 at GROUP granularity, stratified by artist,
     so every near-duplicate cluster lands entirely in one split
  5. Rewrites data/processed/{split}/{artist}/ and data/metadata.csv

Usage:
    python src/data/grouped_split.py --threshold 0.90 --dry-run
    python src/data/grouped_split.py --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import ARTISTS, ArtDataset, get_transforms

SPLITS = ["train", "val", "test"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
IMAGE_SIZE = 336


def compute_features(processed_dir: Path, cache_path: Path) -> tuple[list[Path], np.ndarray]:
    """One DINOv2 center-view feature per processed image, cached by filename."""
    paths = []
    for split in SPLITS:
        for artist in ARTISTS:
            paths.extend(sorted((processed_dir / split / artist).glob("*.jpg")))

    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {n: f for n, f in zip(data["names"], data["feats"])}

    missing = [p for p in paths if p.name not in cached]
    if missing:
        import timm
        from PIL import Image
        backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0,
                                     dynamic_img_size=True).eval()
        tf = get_transforms("val", IMAGE_SIZE)
        with torch.no_grad():
            for p in tqdm(missing, desc="Extracting features"):
                img = np.array(Image.open(p).convert("RGB"))
                x = tf(image=img)["image"].unsqueeze(0)
                cached[p.name] = backbone(x)[0].numpy()
        np.savez_compressed(cache_path,
                            names=np.array(list(cached.keys())),
                            feats=np.stack(list(cached.values())))

    feats = np.stack([cached[p.name] for p in paths])
    return paths, feats


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe grouped re-split")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="cosine similarity above which two images share a group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="report only, move nothing")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    paths, feats = compute_features(processed_dir, processed_dir / "dinov2_features_center.npz")
    artists = [p.parent.name for p in paths]
    n = len(paths)
    print(f"{n} images, threshold {args.threshold}")

    # --- Group by feature similarity ---
    normed = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sims = normed @ normed.T
    uf = UnionFind(n)
    ii, jj = np.where(np.triu(sims, k=1) >= args.threshold)
    for i, j in zip(ii, jj):
        uf.union(int(i), int(j))

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    group_list = sorted(groups.values(), key=len, reverse=True)

    sizes = [len(g) for g in group_list]
    multi = [g for g in group_list if len(g) > 1]
    print(f"{len(group_list)} groups; {len(multi)} with >1 image "
          f"(largest: {sizes[0]}, images in multi-groups: {sum(len(g) for g in multi)})")

    # --- Drop mixed-artist groups ---
    dropped: list[int] = []
    clean_groups: list[tuple[str, list[int]]] = []
    for g in group_list:
        arts = {artists[i] for i in g}
        if len(arts) > 1:
            dropped.extend(g)
            for i in g:
                print(f"  DROP (mixed labels): {paths[i]}")
        else:
            clean_groups.append((arts.pop(), g))
    print(f"Dropped {len(dropped)} images in mixed-artist groups")

    # --- Group-level stratified split ---
    rng = random.Random(args.seed)
    rng.shuffle(clean_groups)
    totals = {a: sum(len(g) for art, g in clean_groups if art == a) for a in ARTISTS}
    target = {(a, s): totals[a] * RATIOS[s] for a in ARTISTS for s in SPLITS}
    assigned: dict[tuple[str, str], int] = defaultdict(int)
    placement: dict[int, str] = {}

    # Largest groups first so they can't overshoot small splits at the end
    for artist, g in sorted(clean_groups, key=lambda x: -len(x[1])):
        split = max(SPLITS, key=lambda s: target[(artist, s)] - assigned[(artist, s)])
        for i in g:
            placement[i] = split
        assigned[(artist, split)] += len(g)

    print("\n=== New split ===")
    for s in SPLITS:
        row = {a: assigned[(a, s)] for a in ARTISTS}
        print(f"  {s:6s}: " + "  ".join(f"{a}={c}" for a, c in row.items()))

    # How many images actually move?
    moves = sum(1 for i, s in placement.items() if paths[i].parts[-3] != s)
    print(f"\n{moves} images change split, {len(dropped)} removed")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    # --- Apply: move files, update metadata ---
    for i in dropped:
        paths[i].unlink()
    for i, split in placement.items():
        src = paths[i]
        if src.parts[-3] == split:
            continue
        dst = processed_dir / split / artists[i] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

    df = pd.read_csv(args.metadata)
    name_to_split = {paths[i].name: s for i, s in placement.items()}
    df = df[df["filename"].isin(name_to_split)].copy()
    df["split"] = df["filename"].map(name_to_split)
    df["processed_path"] = df.apply(
        lambda r: str(processed_dir / r["split"] / r["artist"] / r["filename"]), axis=1)
    df.to_csv(args.metadata, index=False)

    report = {
        "threshold": args.threshold,
        "images": n,
        "groups": len(group_list),
        "multi_image_groups": len(multi),
        "largest_group": sizes[0],
        "dropped_mixed_label": [paths[i].name for i in dropped],
        "moved": moves,
        "split_counts": {s: {a: assigned[(a, s)] for a in ARTISTS} for s in SPLITS},
    }
    with open(processed_dir.parent / "leakage_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Done. Audit report: {processed_dir.parent / 'leakage_audit.json'}")


if __name__ == "__main__":
    main()
