"""Integrity checks for the landmarks-only How2Sign dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.landmark_schema import ARRAY_KEYS, TOTAL_LMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "how2sign_landmarks_hf")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.root / "metadata.parquet"
    if not meta_path.exists():
        raise SystemExit(f"metadata not found: {meta_path}")
    meta = pd.read_parquet(meta_path)
    if args.max_rows is not None:
        meta = meta.head(args.max_rows).copy()

    duplicate_count = meta.duplicated(subset=["split", "sentence_name"]).sum()
    if duplicate_count:
        raise SystemExit(f"duplicate split/sentence_name rows: {duplicate_count}")

    shard_cache = {}
    frame_counts = []
    coverage = []
    for row in meta.itertuples(index=False):
        shard = shard_cache.get(row.shard_file)
        if shard is None:
            path = args.root / row.shard_file
            if not path.exists():
                raise SystemExit(f"missing shard: {path}")
            shard = np.load(path)
            shard_cache[row.shard_file] = shard
        for key in ARRAY_KEYS:
            array_name = f"{row.sample_key}__{key}"
            if array_name not in shard:
                raise SystemExit(f"missing array {array_name} in {row.shard_file}")
        landmarks = shard[f"{row.sample_key}__landmarks_image"]
        mask = shard[f"{row.sample_key}__valid_mask"]
        timestamps = shard[f"{row.sample_key}__timestamps_ms"]
        if landmarks.ndim != 3 or landmarks.shape[1:] != (TOTAL_LMS, 3):
            raise SystemExit(f"bad landmarks shape for {row.sample_key}: {landmarks.shape}")
        if mask.shape != landmarks.shape[:2]:
            raise SystemExit(f"bad mask shape for {row.sample_key}: {mask.shape}")
        if timestamps.shape[0] != landmarks.shape[0]:
            raise SystemExit(f"bad timestamp count for {row.sample_key}")
        if landmarks.shape[0] == 0:
            raise SystemExit(f"empty sequence for {row.sample_key}")
        frame_counts.append(landmarks.shape[0])
        coverage.append(mask.mean())

    print("OK")
    print(f"  rows checked: {len(meta)}")
    print(f"  shards touched: {len(shard_cache)}")
    print(f"  avg frames: {np.mean(frame_counts):.1f}")
    print(f"  avg landmark coverage: {np.mean(coverage):.3f}")


if __name__ == "__main__":
    main()
