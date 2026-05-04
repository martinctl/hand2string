"""MediaPipe landmark extraction using the modern Tasks API.

Returns 75 keypoints per frame: 33 body-pose + 21 left-hand + 21 right-hand.
The 11 face landmarks (nose, eyes, ears, mouth corners) are **already part
of the pose output** (indices 0–10) — the helper
:func:`src.preprocessing.holistic_tasks.face_subset` slices them out if a
caller only needs the face geometry. Dense face mesh / detector outputs are
optional via ``include_face`` on :class:`src.preprocessing.holistic_tasks.Holistic`
(see :mod:`holistic_tasks`).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.holistic_tasks import (
    HAND_LMS,
    POSE_LMS,
    Holistic,
)

TOTAL_LMS = POSE_LMS + HAND_LMS + HAND_LMS  # 33 + 21 + 21 = 75


def _flatten(arr: np.ndarray | None, n: int) -> np.ndarray:
    if arr is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return arr


def _flatten_with_mask(arr: np.ndarray | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    if arr is None:
        return (
            np.full((n, 3), np.nan, dtype=np.float32),
            np.zeros((n,), dtype=np.float32),
        )
    arr = arr.astype(np.float32, copy=False)
    mask = np.isfinite(arr).all(axis=-1).astype(np.float32)
    return arr, mask


def extract_from_video(path: str | Path) -> np.ndarray:
    """Run Holistic over every frame and return ``(T, 75, 3)`` landmarks.

    Coordinates are mediapipe's normalized image space (x, y in [0, 1],
    z relative to the hip / wrist). Missing detections are NaN.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames: list[np.ndarray] = []
    try:
        with Holistic(fps=fps) as h:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lms = h.process(rgb)
                pose = _flatten(lms.pose, POSE_LMS)
                left = _flatten(lms.left_hand, HAND_LMS)
                right = _flatten(lms.right_hand, HAND_LMS)
                frames.append(np.concatenate([pose, left, right], axis=0))
    finally:
        cap.release()

    if not frames:
        return np.zeros((0, TOTAL_LMS, 3), dtype=np.float32)
    return np.stack(frames, axis=0)


def extract_landmarks_and_mask(
    path: str | Path,
    *,
    target_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run Holistic and return ``(landmarks, mask, fps)`` for a video.

    ``landmarks`` has shape ``(T, 75, 3)`` and contains NaNs for missing
    detections. ``mask`` has shape ``(T, 75)`` with 1.0 for detected landmarks
    and 0.0 for missing ones. If ``target_fps`` is set, frames are uniformly
    subsampled while preserving temporal order.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    effective_fps = float(min(target_fps, source_fps) if target_fps else source_fps)

    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frame_idx = 0
    next_t = 0.0
    step = 1.0 / effective_fps if target_fps else 0.0

    try:
        with Holistic(fps=effective_fps) as h:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break

                t = frame_idx / source_fps
                frame_idx += 1
                if target_fps is not None and t + 1e-9 < next_t:
                    continue
                if target_fps is not None:
                    next_t += step

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lms = h.process(rgb)
                pose, pose_mask = _flatten_with_mask(lms.pose, POSE_LMS)
                left, left_mask = _flatten_with_mask(lms.left_hand, HAND_LMS)
                right, right_mask = _flatten_with_mask(lms.right_hand, HAND_LMS)
                frames.append(np.concatenate([pose, left, right], axis=0))
                masks.append(np.concatenate([pose_mask, left_mask, right_mask], axis=0))
    finally:
        cap.release()

    if not frames:
        return (
            np.zeros((0, TOTAL_LMS, 3), dtype=np.float32),
            np.zeros((0, TOTAL_LMS), dtype=np.float32),
            effective_fps,
        )
    return np.stack(frames, axis=0), np.stack(masks, axis=0), effective_fps


def extract_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Single-frame variant for live inference — same ``(75, 3)`` layout.

    Note: this spins up a fresh tracker per call, which is fine for one-off
    inference but wasteful in a loop — for live capture, instantiate
    :class:`Holistic` once and call its ``process`` directly.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with Holistic() as h:
        lms = h.process(rgb)
    return np.concatenate(
        [_flatten(lms.pose, POSE_LMS),
         _flatten(lms.left_hand, HAND_LMS),
         _flatten(lms.right_hand, HAND_LMS)],
        axis=0,
    )
