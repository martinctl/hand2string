"""CLI: run MediaPipe over videos and cache landmark sequences.

For a How2Sign-style dataset root containing ``metadata.parquet`` and clip
paths, this writes:

    <output>/metadata.parquet
    <output>/landmarks/<split>/<sentence_id>.npz

Each ``.npz`` contains ``landmarks`` ``(T, 75, 3)``, ``mask`` ``(T, 75)``,
``sentence`` and identifying metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.mediapipe_extractor import extract_landmarks_and_mask

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
HF_CACHE_DATASET_DIR = "datasets--martinctl--how2sign-asl-clips"


def _safe_name(value: str) -> str:
    keep = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)
    if keep.strip("_"):
        return keep[:120]
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _latest_hf_snapshot(cache_root: Path) -> Path | None:
    """Return the newest local HF snapshot that looks like the clips dataset."""
    candidates = []
    snapshots = cache_root / HF_CACHE_DATASET_DIR / "snapshots"
    if not snapshots.exists():
        return None
    for path in snapshots.iterdir():
        meta = path / "metadata.parquet"
        clips = path / "clips"
        if meta.exists() and clips.exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_input_path(input_path: Path) -> Path:
    """Accept a direct dataset root, an HF cache root, or the old alias path."""
    if input_path.exists():
        if (input_path / "metadata.parquet").exists():
            return input_path
        snapshot = _latest_hf_snapshot(input_path)
        if snapshot is not None:
            return snapshot
        return input_path

    # Common case after ``scripts/download_data.py --root data``: the user
    # tries the README's old ``data/how2sign_hf`` path, but HF stored the
    # snapshot under ``data/datasets--.../snapshots/<hash>``.
    parent = input_path.parent
    if input_path.name in {"how2sign_hf", "how2sign", "how2sign-asl-clips"}:
        snapshot = _latest_hf_snapshot(parent)
        if snapshot is not None:
            print(f"{input_path} not found; using Hugging Face snapshot {snapshot}")
            return snapshot
    return input_path


def _load_rows(input_path: Path, split: str | None, limit: int | None) -> pd.DataFrame:
    input_path = _resolve_input_path(input_path)
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXT:
        return pd.DataFrame([
            {
                "sentence_id": input_path.stem,
                "sentence_name": input_path.stem,
                "sentence": input_path.stem,
                "split": split or "single",
                "clip_path": str(input_path.resolve()),
            }
        ])

    meta_path = input_path / "metadata.parquet"
    if meta_path.exists():
        rows = pd.read_parquet(meta_path)
        if split is not None and "split" in rows.columns:
            rows = rows[rows["split"] == split].reset_index(drop=True)
        if "clip_path" not in rows.columns:
            if "file_name" not in rows.columns:
                raise ValueError("metadata must contain either 'clip_path' or 'file_name'")
            rows["clip_path"] = rows["file_name"].apply(lambda p: str(input_path / p))
        return rows.head(limit).reset_index(drop=True) if limit else rows.reset_index(drop=True)

    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    rows = pd.DataFrame(
        [
            {
                "sentence_id": p.stem,
                "sentence_name": p.stem,
                "sentence": p.stem,
                "split": split or "train",
                "clip_path": str(p.resolve()),
            }
            for p in videos
        ]
    )
    return rows.head(limit).reset_index(drop=True) if limit else rows


def _write_one(row: pd.Series, out_root: Path, target_fps: float | None, overwrite: bool) -> dict:
    split = str(row.get("split", "train"))
    sample_id = str(row.get("sentence_id", row.get("sentence_name", Path(row["clip_path"]).stem)))
    rel = Path("landmarks") / split / f"{_safe_name(sample_id)}.npz"
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists() or overwrite:
        landmarks, mask, fps = extract_landmarks_and_mask(row["clip_path"], target_fps=target_fps)
        np.savez_compressed(
            dst,
            landmarks=landmarks.astype(np.float32),
            mask=mask.astype(np.float32),
            sentence=str(row.get("sentence", "")),
            sentence_id=sample_id,
            split=split,
            source_video=str(row["clip_path"]),
            fps=np.float32(fps),
        )
    else:
        with np.load(dst, allow_pickle=True) as blob:
            landmarks = blob["landmarks"]
            mask = blob["mask"] if "mask" in blob else np.isfinite(landmarks).all(axis=-1)
            fps = float(blob["fps"]) if "fps" in blob else float(target_fps or 0.0)

    out_row = row.to_dict()
    out_row["landmark_path"] = rel.as_posix()
    out_row["num_frames"] = int(landmarks.shape[0])
    out_row["num_landmarks"] = int(landmarks.shape[1]) if landmarks.ndim >= 2 else 0
    out_row["detected_fraction"] = float(mask.mean()) if mask.size else 0.0
    out_row["extraction_fps"] = float(fps)
    return out_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="dataset root or video path")
    parser.add_argument("--output", type=Path, required=True, help="cache root to write")
    parser.add_argument("--split", default="train", help="metadata split to extract; use 'all' for all")
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=None, help="extract only N clips")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.input, split, args.limit)
    if len(rows) == 0:
        raise SystemExit(f"no videos found in {args.input}")

    cached_rows = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="extract landmarks"):
        cached_rows.append(_write_one(row, args.output, args.target_fps, args.overwrite))

    meta = pd.DataFrame(cached_rows)
    meta_path = args.output / "metadata.parquet"
    if meta_path.exists() and not args.overwrite:
        existing = pd.read_parquet(meta_path)
        meta = (
            pd.concat([existing, meta], ignore_index=True)
            .drop_duplicates(subset=["sentence_id", "split"], keep="last")
            .reset_index(drop=True)
        )
    meta.to_parquet(meta_path, index=False)
    print(f"Done. cached {len(cached_rows)} clips -> {meta_path}")


if __name__ == "__main__":
    main()
