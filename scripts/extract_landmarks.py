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
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
HF_CACHE_DATASET_DIR = "datasets--martinctl--how2sign-asl-clips"


def _quiet_native_logs() -> None:
    """Reduce MediaPipe/absl/TFLite native logs that otherwise bury tqdm."""
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_LOG_LEVEL", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass


@contextmanager
def _suppress_worker_stderr(enabled: bool):
    if not enabled:
        yield
        return
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


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


def _split_missing_clip_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if "clip_path" not in rows.columns:
        return rows, []
    exists = rows["clip_path"].apply(lambda p: Path(p).exists())
    missing_rows = []
    for row in rows[~exists].to_dict(orient="records"):
        row["error"] = f"FileNotFoundError: missing clip_path {row.get('clip_path')}"
        missing_rows.append(row)
    return rows[exists].reset_index(drop=True), missing_rows


def _row_to_dict(row) -> dict:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def _write_one(
    row,
    out_root: Path,
    target_fps: float | None,
    overwrite: bool,
    quiet: bool = True,
) -> dict:
    _quiet_native_logs()
    from src.preprocessing.mediapipe_extractor import extract_landmarks_and_mask

    row = _row_to_dict(row)
    split = str(row.get("split", "train"))
    sample_id = str(row.get("sentence_id", row.get("sentence_name", Path(row["clip_path"]).stem)))
    rel = Path("landmarks") / split / f"{_safe_name(sample_id)}.npz"
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists() or overwrite:
        with _suppress_worker_stderr(quiet):
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

    out_row = dict(row)
    out_row["landmark_path"] = rel.as_posix()
    out_row["num_frames"] = int(landmarks.shape[0])
    out_row["num_landmarks"] = int(landmarks.shape[1]) if landmarks.ndim >= 2 else 0
    out_row["detected_fraction"] = float(mask.mean()) if mask.size else 0.0
    out_row["extraction_fps"] = float(fps)
    return out_row


def _write_one_from_args(args: tuple[dict, str, float | None, bool, bool]) -> dict:
    _quiet_native_logs()
    row, out_root, target_fps, overwrite, quiet = args
    row = _row_to_dict(row)
    try:
        return {
            "ok": True,
            "row": _write_one(row, Path(out_root), target_fps, overwrite, quiet),
        }
    except Exception as exc:
        return {
            "ok": False,
            "row": row,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    _quiet_native_logs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="dataset root or video path")
    parser.add_argument("--output", type=Path, required=True, help="cache root to write")
    parser.add_argument("--split", default="train", help="metadata split to extract; use 'all' for all")
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=None, help="extract only N clips")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="parallel video workers")
    parser.add_argument(
        "--verbose-mediapipe",
        action="store_true",
        help="show MediaPipe/TFLite native logs instead of keeping tqdm clean",
    )
    args = parser.parse_args()

    split = None if args.split == "all" else args.split
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.input, split, args.limit)
    if len(rows) == 0:
        raise SystemExit(f"no videos found in {args.input}")

    rows, failed_rows = _split_missing_clip_rows(rows)
    if len(rows) == 0:
        failures = pd.DataFrame(failed_rows)
        failures_path = args.output / "failures.csv"
        failures.to_csv(failures_path, index=False)
        raise SystemExit(f"all input rows are missing clips; wrote {failures_path}")
    if failed_rows:
        print(f"Skipping {len(failed_rows)} metadata rows with missing clip files.")

    cached_rows = []
    if args.workers <= 1:
        for _, row in tqdm(rows.iterrows(), total=len(rows), desc="extract landmarks"):
            row_dict = _row_to_dict(row)
            try:
                cached_rows.append(
                    _write_one(
                        row_dict,
                        args.output,
                        args.target_fps,
                        args.overwrite,
                        quiet=not args.verbose_mediapipe,
                    )
                )
            except Exception as exc:
                failed = dict(row_dict)
                failed["error"] = f"{type(exc).__name__}: {exc}"
                failed_rows.append(failed)
    else:
        jobs = [
            (row, str(args.output), args.target_fps, args.overwrite, not args.verbose_mediapipe)
            for row in rows.to_dict(orient="records")
        ]
        pool = ProcessPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(_write_one_from_args, job) for job in jobs]
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="extract landmarks"):
                result = future.result()
                if result["ok"]:
                    cached_rows.append(result["row"])
                else:
                    failed = dict(result["row"])
                    failed["error"] = result["error"]
                    failed_rows.append(failed)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            print("\nInterrupted. Completed clips remain cached; rerun to resume.")
            raise SystemExit(130)
        else:
            pool.shutdown()

    meta = pd.DataFrame(cached_rows)
    meta_path = args.output / "metadata.parquet"
    if meta_path.exists() and not args.overwrite:
        existing = pd.read_parquet(meta_path)
        meta = (
            pd.concat([existing, meta], ignore_index=True)
            .drop_duplicates(subset=["sentence_id", "split"], keep="last")
            .reset_index(drop=True)
        )
    if len(meta) > 0:
        meta.to_parquet(meta_path, index=False)

    if failed_rows:
        failures = pd.DataFrame(failed_rows)
        failures_path = args.output / "failures.csv"
        if failures_path.exists() and not args.overwrite:
            existing_failures = pd.read_csv(failures_path)
            failures = (
                pd.concat([existing_failures, failures], ignore_index=True)
                .drop_duplicates(subset=["sentence_id", "split"], keep="last")
                .reset_index(drop=True)
            )
        failures.to_csv(failures_path, index=False)
        print(f"Skipped {len(failed_rows)} failed clips -> {failures_path}")

    print(f"Done. cached {len(cached_rows)} clips -> {meta_path}")


if __name__ == "__main__":
    main()
