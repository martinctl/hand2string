"""Deterministic transforms from MediaPipe landmarks to model tensors."""
from __future__ import annotations

import numpy as np

POSE_LMS = 33
HAND_LMS = 21
TOTAL_LMS = POSE_LMS + HAND_LMS + HAND_LMS
LEFT_HAND_START = POSE_LMS
RIGHT_HAND_START = POSE_LMS + HAND_LMS

UPPER_POSE_INDICES = tuple(range(17))
FULL_INDICES = tuple(range(TOTAL_LMS))
UPPER_BODY_INDICES = (
    UPPER_POSE_INDICES
    + tuple(range(LEFT_HAND_START, LEFT_HAND_START + HAND_LMS))
    + tuple(range(RIGHT_HAND_START, RIGHT_HAND_START + HAND_LMS))
)


def landmark_indices(layout: str = "full") -> np.ndarray:
    """Return the fixed landmark indices used by a model layout."""
    if layout == "full":
        return np.asarray(FULL_INDICES, dtype=np.int64)
    if layout in {"upper", "upper_body"}:
        return np.asarray(UPPER_BODY_INDICES, dtype=np.int64)
    raise ValueError(f"unknown landmark layout: {layout!r}")


def make_mask(landmarks: np.ndarray) -> np.ndarray:
    """Build a ``(T, L)`` mask from finite landmark coordinates."""
    return np.isfinite(landmarks).all(axis=-1).astype(np.float32)


def normalize_landmarks(
    landmarks: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    layout: str = "full",
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Torso-center and scale landmarks frame-by-frame.

    Coordinates are centered on the shoulder midpoint when both shoulders are
    detected. Scale prefers shoulder width, then shoulder-to-hip size, then a
    per-frame valid-point radius fallback.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim != 3 or landmarks.shape[-1] != 3:
        raise ValueError(f"expected landmarks with shape (T, L, 3), got {landmarks.shape}")
    if mask is None:
        mask = make_mask(landmarks)
    else:
        mask = np.asarray(mask, dtype=np.float32)

    idx = landmark_indices(layout)
    lms = landmarks[:, idx, :].copy()
    out_mask = mask[:, idx].astype(np.float32, copy=True)
    out = np.zeros_like(lms, dtype=np.float32)

    for t in range(lms.shape[0]):
        full_frame = landmarks[t]
        full_mask = mask[t] > 0.5

        if full_mask[11] and full_mask[12]:
            center = (full_frame[11] + full_frame[12]) * 0.5
            scale = float(np.linalg.norm(full_frame[11] - full_frame[12]))
        elif full_mask[0]:
            center = full_frame[0]
            scale = 0.0
        elif full_mask.any():
            center = np.nanmean(full_frame[full_mask], axis=0)
            scale = 0.0
        else:
            center = np.zeros((3,), dtype=np.float32)
            scale = 1.0

        if scale <= eps and full_mask[11] and full_mask[12] and full_mask[23] and full_mask[24]:
            shoulders = (full_frame[11] + full_frame[12]) * 0.5
            hips = (full_frame[23] + full_frame[24]) * 0.5
            scale = float(np.linalg.norm(shoulders - hips))

        valid_selected = out_mask[t] > 0.5
        if scale <= eps and valid_selected.any():
            centered = lms[t, valid_selected] - center
            scale = float(np.linalg.norm(centered, axis=-1).max(initial=1.0))
        if scale <= eps:
            scale = 1.0

        if valid_selected.any():
            out[t, valid_selected] = (lms[t, valid_selected] - center) / scale

    return out.astype(np.float32), out_mask


def resample_sequence(values: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample ``values`` along the first axis."""
    values = np.asarray(values, dtype=np.float32)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if values.shape[0] == target_frames:
        return values.astype(np.float32, copy=False)
    if values.shape[0] == 0:
        return np.zeros((target_frames, *values.shape[1:]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values, target_frames, axis=0).astype(np.float32)

    src_x = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_x, src_x, flat[:, i])
    return out.reshape((target_frames, *values.shape[1:])).astype(np.float32)


def landmarks_to_features(
    landmarks: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    target_frames: int = 128,
    layout: str = "full",
) -> np.ndarray:
    """Convert a variable-length landmark clip to ``(T, L * 4)`` features."""
    norm, norm_mask = normalize_landmarks(landmarks, mask, layout=layout)
    norm = resample_sequence(norm, target_frames)
    norm_mask = resample_sequence(norm_mask, target_frames)
    features = np.concatenate([norm, norm_mask[..., None]], axis=-1)
    return features.reshape(target_frames, -1).astype(np.float32)
