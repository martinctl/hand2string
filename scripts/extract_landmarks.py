"""Extract MediaPipe landmarks + rich derived features from How2Sign clips.

Downloads each video clip directly via HTTP (no HuggingFace API calls, so no
rate limiting), runs MediaPipe Holistic on every frame, and saves one
compressed .npz per clip.  The video file is deleted after extraction to keep
disk usage low.

Per-clip .npz contains:
  landmarks  (T, 75, 3)   raw output — pose [0:33], left-hand [33:54],
                           right-hand [54:75].  NaN = not detected.

  features   (T, 410)     enriched feature vector:
    [  0:225]  raw pose + hands (75×3), NaN→0
    [225:237]  6 BlazeFace keypoints × (x,y)         face / mouthing
    [237:300]  left  hand normalised shape (63)       handshape
    [300:363]  right hand normalised shape (63)
    [363:368]  left  finger extensions (5)            fingers up/down
    [368:373]  right finger extensions (5)
    [373:383]  left  fingertip pairwise distances (10) touching / pinching
    [383:393]  right fingertip pairwise distances (10)
    [393:396]  left  palm normal (3)                  palm facing direction
    [396:399]  right palm normal (3)
    [399:400]  left  wrist-to-nose distance (1)       signing location
    [400:401]  right wrist-to-nose distance (1)
    [401:404]  inter-wrist vector right−left (3)      bimanual coord
    [404:407]  left  wrist velocity Δ/frame (3)       movement
    [407:410]  right wrist velocity Δ/frame (3)

Output layout (= dataset.root in configs/transformer.yaml):
  <output>/
      metadata.parquet
      landmarks/train/<sentence_name>.npz
      landmarks/val/<sentence_name>.npz

Usage
-----
# Full dataset (run inside a SLURM job):
python scripts/extract_landmarks.py \\
    --output /scratch/izar/banuls/how2sign_landmarks

# Smoke-test with 20 clips:
python scripts/extract_landmarks.py \\
    --output /scratch/izar/banuls/how2sign_landmarks --limit 20

# Private repo:
python scripts/extract_landmarks.py \\
    --output /scratch/izar/banuls/how2sign_landmarks --token hf_...
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.holistic_tasks import HAND_LMS, POSE_LMS, Holistic

# ── HuggingFace direct-download (no API, no rate limiting) ───────────────────

DEFAULT_REPO_ID = "martinctl/how2sign-asl-clips"
_HF_BASE = "https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


def _hf_url(repo_id: str, path: str) -> str:
    return _HF_BASE.format(repo_id=repo_id, path=path)


def _http_get(url: str, token: str | None = None) -> bytes:
    """Download *url* and return raw bytes.  Raises on HTTP error."""
    headers: dict[str, str] = {"User-Agent": "hand2string-extractor/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _download_to_tmp(url: str, suffix: str, token: str | None = None) -> Path:
    """Download *url* into a NamedTemporaryFile and return its path."""
    data = _http_get(url, token)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


# ── Feature-vector constants ──────────────────────────────────────────────────

FEATURE_DIM = 410

_WRIST      = 0
_TIP_IDX    = [4, 8, 12, 16, 20]
_PIP_IDX    = [3, 6, 10, 14, 18]
_INDEX_MCP  = 5
_MIDDLE_MCP = 9
_PINKY_MCP  = 17
_NOSE_POSE  = 0
_FACE_KPS   = 6


# ── Per-hand derived features ─────────────────────────────────────────────────

def _ok(lms: np.ndarray | None) -> bool:
    return lms is not None and not np.isnan(lms).all()


def _norm_hand(lms: np.ndarray | None) -> np.ndarray:
    """21 landmarks relative to wrist, scaled by wrist-to-middle-MCP.
    Removes global position + scale → pure finger configuration.
    Returns (63,).
    """
    if not _ok(lms):
        return np.zeros(63, dtype=np.float32)
    h = np.nan_to_num(lms, nan=0.0)
    wrist = h[_WRIST]
    scale = np.linalg.norm(h[_MIDDLE_MCP] - wrist) + 1e-8
    return ((h - wrist) / scale).astype(np.float32).flatten()


def _finger_extensions(lms: np.ndarray | None) -> np.ndarray:
    """1 = finger extended / up, 0 = curled.
    Heuristic: tip farther from wrist than PIP joint.
    Returns (5,)  [thumb, index, middle, ring, pinky].
    """
    if not _ok(lms):
        return np.zeros(5, dtype=np.float32)
    h = np.nan_to_num(lms, nan=0.0)
    w = h[_WRIST]
    return (
        np.linalg.norm(h[_TIP_IDX] - w, axis=1) >
        np.linalg.norm(h[_PIP_IDX] - w, axis=1)
    ).astype(np.float32)


def _fingertip_pairs(lms: np.ndarray | None) -> np.ndarray:
    """C(5,2)=10 pairwise distances between fingertips.
    Small value → fingertips touching / pinching.
    Returns (10,).
    """
    if not _ok(lms):
        return np.zeros(10, dtype=np.float32)
    tips = np.nan_to_num(lms, nan=0.0)[_TIP_IDX]
    return np.array(
        [np.linalg.norm(tips[i] - tips[j])
         for i in range(5) for j in range(i + 1, 5)],
        dtype=np.float32,
    )


def _palm_normal(lms: np.ndarray | None) -> np.ndarray:
    """Unit vector normal to the palm plane.
    Sign encodes palm-toward-camera vs palm-away — a key ASL parameter.
    Returns (3,).
    """
    if not _ok(lms):
        return np.zeros(3, dtype=np.float32)
    h = np.nan_to_num(lms, nan=0.0)
    n = np.cross(h[_INDEX_MCP] - h[_WRIST], h[_PINKY_MCP] - h[_WRIST])
    return (n / (np.linalg.norm(n) + 1e-8)).astype(np.float32)


def _wrist_nose(hand: np.ndarray | None, pose: np.ndarray | None) -> float:
    """Distance from hand wrist to nose — encodes signing location."""
    if not _ok(hand) or pose is None or np.isnan(pose[_NOSE_POSE]).any():
        return 0.0
    return float(np.linalg.norm(
        np.nan_to_num(hand[_WRIST], nan=0.0) -
        np.nan_to_num(pose[_NOSE_POSE], nan=0.0)
    ))


def _face_kps(detections) -> np.ndarray:
    """6 BlazeFace keypoints (x, y) for the largest face.
    Captures mouth shape / mouthing — important for ASL disambiguation.
    Returns (12,).
    """
    if not detections:
        return np.zeros(_FACE_KPS * 2, dtype=np.float32)
    kps = detections[0].keypoints[:_FACE_KPS]
    return np.array([[kp.x, kp.y] for kp in kps], dtype=np.float32).flatten()


# ── Sequence enrichment ───────────────────────────────────────────────────────

def enrich(raw: np.ndarray, face_seq: list) -> np.ndarray:
    """Build (T, 410) enriched features from (T, 75, 3) raw landmarks."""
    T = len(raw)
    pose  = raw[:, :33,  :]
    left  = raw[:, 33:54, :]
    right = raw[:, 54:75, :]

    base   = np.nan_to_num(raw, nan=0.0).reshape(T, -1).astype(np.float32)
    face   = np.stack([_face_kps(face_seq[t])         for t in range(T)])
    lnorm  = np.stack([_norm_hand(left[t])             for t in range(T)])
    rnorm  = np.stack([_norm_hand(right[t])            for t in range(T)])
    lext   = np.stack([_finger_extensions(left[t])     for t in range(T)])
    rext   = np.stack([_finger_extensions(right[t])    for t in range(T)])
    ltip   = np.stack([_fingertip_pairs(left[t])       for t in range(T)])
    rtip   = np.stack([_fingertip_pairs(right[t])      for t in range(T)])
    lnrm   = np.stack([_palm_normal(left[t])           for t in range(T)])
    rnrm   = np.stack([_palm_normal(right[t])          for t in range(T)])
    lnose  = np.array([_wrist_nose(left[t],  pose[t])  for t in range(T)],
                      dtype=np.float32).reshape(T, 1)
    rnose  = np.array([_wrist_nose(right[t], pose[t])  for t in range(T)],
                      dtype=np.float32).reshape(T, 1)
    lw     = np.nan_to_num(left[:,  _WRIST, :], nan=0.0)
    rw     = np.nan_to_num(right[:, _WRIST, :], nan=0.0)
    inter  = (rw - lw).astype(np.float32)
    lvel   = np.diff(lw, axis=0, prepend=lw[:1]).astype(np.float32)
    rvel   = np.diff(rw, axis=0, prepend=rw[:1]).astype(np.float32)

    out = np.concatenate([
        base, face,              # 225 + 12
        lnorm, rnorm,            # 63  + 63
        lext,  rext,             # 5   + 5
        ltip,  rtip,             # 10  + 10
        lnrm,  rnrm,             # 3   + 3
        lnose, rnose,            # 1   + 1
        inter,                   # 3
        lvel,  rvel,             # 3   + 3
    ], axis=1)

    assert out.shape == (T, FEATURE_DIM), f"expected ({T}, {FEATURE_DIM}), got {out.shape}"
    return out.astype(np.float32)


# ── Per-clip extraction ───────────────────────────────────────────────────────

def _fill(arr: np.ndarray | None, n: int) -> np.ndarray:
    return arr if arr is not None else np.full((n, 3), np.nan, dtype=np.float32)


def extract_clip(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Run Holistic on every frame → (raw (T,75,3), features (T,410))."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames_raw: list[np.ndarray] = []
    frames_face: list = []
    try:
        with Holistic(fps=fps, include_face=True) as h:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lms = h.process(rgb)
                frames_raw.append(np.concatenate([
                    _fill(lms.pose,       POSE_LMS),
                    _fill(lms.left_hand,  HAND_LMS),
                    _fill(lms.right_hand, HAND_LMS),
                ], axis=0))
                frames_face.append(lms.face_detections or [])
    finally:
        cap.release()

    if not frames_raw:
        empty = np.zeros((0, POSE_LMS + HAND_LMS * 2, 3), dtype=np.float32)
        return empty, np.zeros((0, FEATURE_DIM), dtype=np.float32)

    raw = np.stack(frames_raw).astype(np.float32)
    return raw, enrich(raw, frames_face)


# ── Parallel worker (module-level so multiprocessing can pickle it) ───────────

def _worker(task: tuple) -> tuple[str, str | None]:
    """Download one clip, extract landmarks, save .npz, delete the video.

    Returns (sentence_id, None) on success or (sentence_id, error_msg) on failure.
    Each worker process creates its own Holistic instance — they are not
    thread-safe but are fully process-safe.
    """
    repo_id, file_name, sentence_id, npz_path_str, token = task
    npz_path = Path(npz_path_str)
    clip_tmp: Path | None = None
    try:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        clip_tmp = _download_to_tmp(_hf_url(repo_id, file_name), ".mp4", token)
        raw, features = extract_clip(clip_tmp)
        if len(raw) == 0:
            raise RuntimeError("no frames decoded")
        np.savez_compressed(npz_path, landmarks=raw, features=features)
        return sentence_id, None
    except Exception as exc:
        return sentence_id, str(exc)
    finally:
        if clip_tmp is not None and clip_tmp.exists():
            clip_tmp.unlink()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", required=True, type=Path,
                   help="Root output dir — set as dataset.root in the config")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HuggingFace dataset repo (default: {DEFAULT_REPO_ID})")
    p.add_argument("--token", default=None,
                   help="HuggingFace access token (for private repos)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N clips (smoke-test)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel extraction workers (default: 4)")
    p.add_argument("--skip-existing", dest="skip_existing",
                   action="store_true", default=True,
                   help="Skip clips whose .npz already exists (default: on)")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    repo_id = args.repo_id
    token   = args.token

    # ── 1. Download metadata.parquet (1 HTTP request, no rate limit) ──────────
    meta_dst = out_root / "metadata.parquet"
    if not meta_dst.exists():
        print("[extract] Downloading metadata.parquet …")
        raw_bytes = _http_get(_hf_url(repo_id, "metadata.parquet"), token)
        meta_dst.write_bytes(raw_bytes)
        print(f"[extract] metadata.parquet → {meta_dst}")

    df = pd.read_parquet(meta_dst)
    print(f"[extract] {len(df)} clips, splits: {sorted(df['split'].unique())}")

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"[extract] --limit {args.limit} → processing {len(df)} clips")

    # ── 2. Pre-fetch MediaPipe models once (avoids race conditions) ────────────
    print("[extract] Pre-fetching MediaPipe models …")
    from src.preprocessing.holistic_tasks import (
        FACE_DETECTOR_MODEL_URL, HAND_MODEL_URL, POSE_MODEL_URL, _ensure_model,
    )
    _ensure_model(POSE_MODEL_URL,          "pose_landmarker_lite.task")
    _ensure_model(HAND_MODEL_URL,          "hand_landmarker.task")
    _ensure_model(FACE_DETECTOR_MODEL_URL, "blaze_face_full_range.tflite")
    print("[extract] Models ready.\n")

    # ── 3. Build task list (skip already-done clips) ──────────────────────────
    tasks: list[tuple] = []
    n_skip = 0
    for row in df.itertuples(index=False):
        parts    = Path(row.file_name).parts
        npz_path = out_root / Path("landmarks", *parts[1:]).with_suffix(".npz")
        if args.skip_existing and npz_path.exists():
            n_skip += 1
            continue
        tasks.append((repo_id, row.file_name, row.sentence_id,
                      str(npz_path), token))

    print(f"[extract] {len(tasks)} clips to process  |  "
          f"{n_skip} already done  |  {args.workers} workers\n")

    # ── 4. Process in parallel ────────────────────────────────────────────────
    # Each worker independently downloads one clip, runs MediaPipe, saves the
    # .npz, then deletes the video.  Workers share no state so this is safe.
    n_ok = n_fail = 0
    failures: list[str] = []

    with Pool(processes=args.workers) as pool:
        for sid, err in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=len(tasks), desc="clips", unit="clip",
        ):
            if err:
                n_fail += 1
                failures.append(f"{sid}: {err}")
                tqdm.write(f"  FAIL {sid}: {err}")
            else:
                n_ok += 1

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(
        f"\n[extract] Done.\n"
        f"  processed : {n_ok}\n"
        f"  skipped   : {n_skip}\n"
        f"  failed    : {n_fail}\n"
        f"  output    : {out_root}"
    )
    if failures:
        log = out_root / "extraction_failures.txt"
        log.write_text("\n".join(failures), encoding="utf-8")
        print(f"  failures  : {log}")


if __name__ == "__main__":
    main()
