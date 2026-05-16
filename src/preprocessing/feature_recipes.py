"""Deterministic feature recipes computed from landmark arrays on demand."""
from __future__ import annotations

import numpy as np

from src.preprocessing.landmark_schema import (
    ASL_FACE_LANDMARKS,
    FACE_BLOCK,
    LEFT_HAND_BLOCK,
    POSE_BLOCK,
    RIGHT_HAND_BLOCK,
)

FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FINGER_NAMES: tuple[str, ...] = tuple(FINGER_CHAINS)
HAND_BODY_DISTANCE_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "nose",
    "mouth_center",
    "torso_center",
)
TWO_HAND_FEATURE_NAMES: tuple[str, ...] = (
    "wrist_distance",
    "palm_distance",
    "thumb_tip_distance",
    "index_tip_distance",
    "middle_tip_distance",
    "ring_tip_distance",
    "pinky_tip_distance",
)
ARM_FEATURE_NAMES: tuple[str, ...] = (
    "left_elbow_angle_rad",
    "right_elbow_angle_rad",
    "left_forearm_len_over_shoulders",
    "right_forearm_len_over_shoulders",
)
FACE_CUE_NAMES: tuple[str, ...] = (
    "mouth_opening",
    "mouth_width",
    "eye_distance",
)
GEOMETRIC_FEATURE_NAMES: tuple[str, ...] = (
    *(f"left_{name}_curl" for name in FINGER_NAMES),
    *(f"right_{name}_curl" for name in FINGER_NAMES),
    "left_hand_openness",
    "right_hand_openness",
    *(f"left_wrist_to_{name}" for name in HAND_BODY_DISTANCE_NAMES),
    *(f"right_wrist_to_{name}" for name in HAND_BODY_DISTANCE_NAMES),
    *TWO_HAND_FEATURE_NAMES,
    *ARM_FEATURE_NAMES,
    *FACE_CUE_NAMES,
)

POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_MOUTH = 9
POSE_RIGHT_MOUTH = 10

FACE_LEFT_EYE = 33
FACE_RIGHT_EYE = 263
FACE_UPPER_LIP = 17
FACE_LOWER_LIP = 14
FACE_LEFT_MOUTH = 61
FACE_RIGHT_MOUTH = 291


def _safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab = a - b
    cb = c - b
    denom = _safe_norm(ab, keepdims=True) * _safe_norm(cb, keepdims=True)
    cos = np.sum(ab * cb, axis=-1, keepdims=True) / np.maximum(denom, 1e-6)
    return np.arccos(np.clip(cos[..., 0], -1.0, 1.0)).astype(np.float32)


def hand_landmarks(landmarks: np.ndarray, side: str) -> np.ndarray:
    block = LEFT_HAND_BLOCK if side == "left" else RIGHT_HAND_BLOCK
    return landmarks[:, block.start:block.end, :]


def pose_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[:, POSE_BLOCK.start:POSE_BLOCK.end, :]


def face_landmarks(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[:, FACE_BLOCK.start:FACE_BLOCK.end, :]


def finger_curl(landmarks: np.ndarray, side: str) -> np.ndarray:
    """Return five curl values per frame, roughly 0=open and 1=curled."""
    hand = hand_landmarks(landmarks, side)
    wrist = hand[:, 0]
    curls = []
    for chain in FINGER_CHAINS.values():
        mcp, pip, dip, tip = chain
        base_to_tip = _safe_norm(hand[:, tip] - hand[:, mcp])
        chain_len = (
            _safe_norm(hand[:, pip] - hand[:, mcp])
            + _safe_norm(hand[:, dip] - hand[:, pip])
            + _safe_norm(hand[:, tip] - hand[:, dip])
        )
        extension = base_to_tip / np.maximum(chain_len, 1e-6)
        wrist_tip = _safe_norm(hand[:, tip] - wrist)
        wrist_mcp = _safe_norm(hand[:, mcp] - wrist)
        folded = (wrist_tip < wrist_mcp).astype(np.float32)
        curls.append(np.clip((1.0 - extension) + 0.25 * folded, 0.0, 1.0))
    return np.stack(curls, axis=-1).astype(np.float32)


def hand_openness(landmarks: np.ndarray, side: str) -> np.ndarray:
    return (1.0 - finger_curl(landmarks, side).mean(axis=-1)).astype(np.float32)


def hand_body_distances(landmarks: np.ndarray, side: str) -> np.ndarray:
    """Distances from wrist to shoulders, nose, mouth center, and torso center."""
    hand = hand_landmarks(landmarks, side)
    pose = pose_landmarks(landmarks)
    wrist = hand[:, 0]
    mouth = 0.5 * (pose[:, POSE_LEFT_MOUTH] + pose[:, POSE_RIGHT_MOUTH])
    torso = 0.5 * (pose[:, POSE_LEFT_SHOULDER] + pose[:, POSE_RIGHT_SHOULDER])
    targets = np.stack([
        pose[:, POSE_LEFT_SHOULDER],
        pose[:, POSE_RIGHT_SHOULDER],
        pose[:, POSE_NOSE],
        mouth,
        torso,
    ], axis=1)
    return _safe_norm(targets - wrist[:, None, :], axis=-1).astype(np.float32)


def two_hand_features(landmarks: np.ndarray) -> np.ndarray:
    left = hand_landmarks(landmarks, "left")
    right = hand_landmarks(landmarks, "right")
    wrist_distance = _safe_norm(left[:, 0] - right[:, 0])
    palm_distance = _safe_norm(left[:, 9] - right[:, 9])
    fingertip_distances = np.stack([
        _safe_norm(left[:, tip] - right[:, tip])
        for tip in (4, 8, 12, 16, 20)
    ], axis=-1)
    return np.concatenate([
        wrist_distance[:, None],
        palm_distance[:, None],
        fingertip_distances,
    ], axis=-1).astype(np.float32)


def arm_angles(landmarks: np.ndarray) -> np.ndarray:
    pose = pose_landmarks(landmarks)
    left_elbow = _angle(pose[:, POSE_LEFT_SHOULDER], pose[:, POSE_LEFT_ELBOW], pose[:, POSE_LEFT_WRIST])
    right_elbow = _angle(pose[:, POSE_RIGHT_SHOULDER], pose[:, POSE_RIGHT_ELBOW], pose[:, POSE_RIGHT_WRIST])
    shoulder_line = pose[:, POSE_RIGHT_SHOULDER] - pose[:, POSE_LEFT_SHOULDER]
    left_forearm = pose[:, POSE_LEFT_WRIST] - pose[:, POSE_LEFT_ELBOW]
    right_forearm = pose[:, POSE_RIGHT_WRIST] - pose[:, POSE_RIGHT_ELBOW]
    left_forearm_len = _safe_norm(left_forearm)
    right_forearm_len = _safe_norm(right_forearm)
    shoulder_width = _safe_norm(shoulder_line)
    return np.stack([
        left_elbow,
        right_elbow,
        left_forearm_len / np.maximum(shoulder_width, 1e-6),
        right_forearm_len / np.maximum(shoulder_width, 1e-6),
    ], axis=-1).astype(np.float32)


def face_cues(landmarks: np.ndarray) -> np.ndarray:
    face = face_landmarks(landmarks)
    index_map = {idx: i for i, idx in enumerate(ASL_FACE_LANDMARKS)}
    upper_lip = face[:, index_map[FACE_UPPER_LIP]]
    lower_lip = face[:, index_map[FACE_LOWER_LIP]]
    left_mouth = face[:, index_map[FACE_LEFT_MOUTH]]
    right_mouth = face[:, index_map[FACE_RIGHT_MOUTH]]
    left_eye = face[:, index_map[FACE_LEFT_EYE]]
    right_eye = face[:, index_map[FACE_RIGHT_EYE]]
    return np.stack([
        _safe_norm(upper_lip - lower_lip),
        _safe_norm(left_mouth - right_mouth),
        _safe_norm(left_eye - right_eye),
    ], axis=-1).astype(np.float32)


def temporal_delta(values: np.ndarray, lag: int = 1) -> np.ndarray:
    if lag < 1:
        raise ValueError("lag must be >= 1")
    out = np.zeros_like(values, dtype=np.float32)
    out[lag:] = values[lag:] - values[:-lag]
    return out


def geometric_feature_block(landmarks: np.ndarray) -> np.ndarray:
    """Compact v1 feature block for quick experiments."""
    parts = [
        finger_curl(landmarks, "left"),
        finger_curl(landmarks, "right"),
        hand_openness(landmarks, "left")[:, None],
        hand_openness(landmarks, "right")[:, None],
        hand_body_distances(landmarks, "left"),
        hand_body_distances(landmarks, "right"),
        two_hand_features(landmarks),
        arm_angles(landmarks),
        face_cues(landmarks),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def geometric_feature_dict(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Named feature arrays for inspection dashboards and notebooks."""
    features = geometric_feature_block(landmarks)
    return {name: features[:, i] for i, name in enumerate(GEOMETRIC_FEATURE_NAMES)}
