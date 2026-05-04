"""Extract hand landmarks from a folder of class-labeled images.

Expected folder layout:
    <root>/
        A/img1.jpg img2.jpg ...
        B/img1.jpg ...
        ...

Each image is passed through MediaPipe HandLandmarker (still-image mode).
Images where no hand is detected are skipped.

Output: a single .npz with
    X        : (N, 63) float32   -- 21 hand landmarks * 3 (x,y,z) flattened
    y        : (N,)    int64     -- class index
    classes  : (C,)    object    -- class label strings (sorted folder names)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.preprocessing.holistic_tasks import HAND_MODEL_URL, _ensure_model

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True,
                   help="folder with subdirs per class")
    p.add_argument("--out", type=Path, default=Path("data/asl_alphabet_landmarks.npz"))
    p.add_argument("--per-class-limit", type=int, default=None,
                   help="cap N images per class (for quick tests)")
    return p.parse_args()


def build_hand_landmarker() -> mp_vision.HandLandmarker:
    path = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
        )
    )


def extract_one(image_path: Path, hand_landmarker) -> np.ndarray | None:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = hand_landmarker.detect(mp_img)
    if not res.hand_landmarks:
        return None
    lms = res.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in lms[:21]], dtype=np.float32).flatten()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in args.root.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"No class subfolders found in {args.root}")
    print(f"Found {len(classes)} classes: {classes[:10]}{'...' if len(classes) > 10 else ''}")

    landmarker = build_hand_landmarker()

    X_list, y_list = [], []
    skipped = 0
    try:
        for ci, cls in enumerate(classes):
            paths = [p for p in (args.root / cls).iterdir()
                     if p.suffix.lower() in IMG_EXT]
            if args.per_class_limit:
                paths = paths[:args.per_class_limit]
            for p in tqdm(paths, desc=f"{cls} ({ci+1}/{len(classes)})", leave=False):
                vec = extract_one(p, landmarker)
                if vec is None:
                    skipped += 1
                    continue
                X_list.append(vec)
                y_list.append(ci)
    finally:
        landmarker.close()

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y=y, classes=np.array(classes, dtype=object))
    print(f"Done. {X.shape[0]} samples, {len(classes)} classes, "
          f"{skipped} skipped (no hand detected). -> {args.out}")


if __name__ == "__main__":
    main()
