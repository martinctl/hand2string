"""Build the landmarks-only How2Sign dataset from local source videos.

Example:
    python scripts/build_how2sign_landmark_dataset.py \
        --root .. \
        --out data/how2sign_landmarks_hf \
        --limit-per-split 5 --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.how2sign_landmarks import build_landmark_dataset, rebuild_landmark_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT.parent,
        help="folder containing how2sign_realigned_*.csv and *_raw_videos folders",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "how2sign_landmarks_hf")
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--samples-per-shard", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="number of source videos to process in parallel",
    )
    parser.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="lite")
    parser.add_argument("--no-face", action="store_true", help="skip Face Landmarker for a much faster, hands+pose-only build")
    parser.add_argument(
        "--rebuild-metadata-only",
        action="store_true",
        help="reconstruct metadata.parquet from existing shards and exit",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="smoke-test limit applied independently to train/val/test",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    split_specs = [
        ("train", root / "how2sign_realigned_train.csv", root / "train_raw_videos" / "raw_videos"),
        ("val", root / "how2sign_realigned_val.csv", root / "val_raw_videos" / "raw_videos"),
        ("test", root / "how2sign_realigned_test.csv", root / "test_raw_videos" / "raw_videos"),
    ]
    for split, csv_path, videos_dir in split_specs:
        if not csv_path.exists():
            raise SystemExit(f"{split} CSV not found: {csv_path}")
        if not videos_dir.exists():
            raise SystemExit(f"{split} videos folder not found: {videos_dir}")

    if args.rebuild_metadata_only:
        meta = rebuild_landmark_metadata(
            split_specs,
            args.out,
            limit_per_split=args.limit_per_split,
        )
        print("\nMetadata rebuilt.")
        print(f"  rows: {len(meta)}")
        print(f"  output: {args.out}")
        return

    meta = build_landmark_dataset(
        split_specs,
        args.out,
        target_fps=args.target_fps,
        samples_per_shard=args.samples_per_shard,
        limit_per_split=args.limit_per_split,
        overwrite=args.overwrite,
        workers=args.workers,
        pose_model=args.pose_model,
        include_face=not args.no_face,
    )
    counts = meta["split"].value_counts().to_dict()
    print("\nDone.")
    print(f"  rows: {len(meta)}")
    print(f"  splits: {counts}")
    print(f"  avg frames: {meta['n_frames'].mean():.1f}")
    print(f"  output: {args.out}")


if __name__ == "__main__":
    main()
