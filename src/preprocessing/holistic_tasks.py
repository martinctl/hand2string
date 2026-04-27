"""MediaPipe Tasks-based Holistic helper.

The legacy ``mediapipe.solutions.holistic`` module was dropped in mediapipe
0.10.x. This wrapper drives PoseLandmarker + HandLandmarker from the
modern Tasks API and exposes a per-frame ``process(rgb)`` call plus the
connection tables we need to draw skeletons.

BlazePose already includes 11 face landmarks in the pose stream (indices 0–10).
For overlays, pass ``include_face=True`` to also run the **Face Detector**
task (BlazeFace full-range): bounding boxes plus six normalized keypoints
(eyes, nose tip, mouth, tragions) per face — lighter than Face Landmarker.

Models are auto-downloaded into ``$HAND2STRING_MODEL_DIR`` (default
``~/.cache/hand2string/mediapipe/``) on first use.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import detections as detections_lib

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
# Full-range BlazeFace: better for wide shots / small faces than short-range.
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)

POSE_LMS = 33
HAND_LMS = 21

# BlazeFace short output keypoints (fixed order); labels are often unset in API.
FACE_DETECTOR_KEYPOINT_LABELS: tuple[str, ...] = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth",
    "right_eye_tragion",
    "left_eye_tragion",
)

# BlazeFace still outputs 6 keypoints; overlays only draw these (no tragions).
FACE_DETECTOR_KEYPOINT_DRAW_INDICES: frozenset[int] = frozenset({0, 1, 2, 3})

# BlazePose 33-keypoint skeleton.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)

# 21-keypoint hand skeleton.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# The 11 face landmarks BlazePose detects (indices into POSE):
#   0  nose
#   1-3  left eye:  inner / center / outer
#   4-6  right eye: inner / center / outer
#   7-8  left ear, right ear
#   9-10 mouth corners (left, right)
FACE_FROM_POSE_INDICES: tuple[int, ...] = tuple(range(11))

# Face sub-skeleton drawn from the 11 pose face points: eye chains via the
# nose, ear-to-ear-eye links for face width, an approximate "jaw" outline
# from each ear down to the mouth corner, and the mouth line.
FACE_FROM_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (3, 2), (2, 1), (1, 0),    # left eye chain -> nose
    (0, 4), (4, 5), (5, 6),    # nose -> right eye chain
    (3, 7),                    # left outer eye -> left ear
    (6, 8),                    # right outer eye -> right ear
    (7, 9),                    # left ear -> left mouth corner (jaw)
    (8, 10),                   # right ear -> right mouth corner (jaw)
    (9, 10),                   # mouth line
    (1, 9), (4, 10),           # eye-inner -> mouth corner (cheek diagonal)
)


def _model_cache_dir() -> Path:
    override = os.environ.get("HAND2STRING_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "hand2string" / "mediapipe"


def _ensure_model(url: str, name: str) -> Path:
    cache = _model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        print(f"Downloading mediapipe model: {name}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


@dataclass
class FrameLandmarks:
    """Per-frame normalized landmarks. ``None`` entries mean "not detected".

    When :class:`Holistic` is constructed with ``include_face=True``,
    ``face_detections`` lists :class:`detections_lib.Detection` objects
    (pixel bbox + normalized keypoints in full-image space).
    """
    pose: np.ndarray | None       # (33, 3) x,y,z in image-normalized coords
    left_hand: np.ndarray | None  # (21, 3)
    right_hand: np.ndarray | None # (21, 3)
    face_detections: list[detections_lib.Detection] | None = None


def _to_array(landmarks, n: int) -> np.ndarray:
    return np.asarray(
        [[lm.x, lm.y, lm.z] for lm in landmarks[:n]],
        dtype=np.float32,
    )


def face_subset(pose: np.ndarray | None) -> np.ndarray | None:
    """Slice the 11 face landmarks out of a pose array."""
    if pose is None:
        return None
    return pose[list(FACE_FROM_POSE_INDICES)]


class Holistic:
    """Stateful per-video tracker. Use as ``with Holistic() as h: h.process(rgb)``."""

    def __init__(self, fps: float = 25.0, *, include_face: bool = False):
        pose_path = _ensure_model(POSE_MODEL_URL, "pose_landmarker_lite.task")
        hand_path = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")

        self._pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        self._hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )

        self._face_detector: mp_vision.FaceDetector | None = None
        if include_face:
            face_path = _ensure_model(
                FACE_DETECTOR_MODEL_URL, "blaze_face_full_range.tflite"
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(
                mp_vision.FaceDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(face_path)),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    min_detection_confidence=0.3,
                    min_suppression_threshold=0.3,
                )
            )

        self._dt_ms = max(1, int(round(1000.0 / fps)))
        self._t_ms = 0
        self._closed = False

    def __enter__(self) -> "Holistic":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._pose.close()
        self._hand.close()
        if self._face_detector is not None:
            self._face_detector.close()
        self._closed = True

    def process(self, rgb: np.ndarray) -> FrameLandmarks:
        """Run pose + hand inference on an RGB uint8 frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._t_ms
        self._t_ms += self._dt_ms

        pose_res = self._pose.detect_for_video(mp_image, ts)
        hand_res = self._hand.detect_for_video(mp_image, ts)

        face_list: list[detections_lib.Detection] | None = None
        if self._face_detector is not None:
            face_res = self._face_detector.detect_for_video(mp_image, ts)
            if face_res.detections:
                face_list = list(face_res.detections)

        pose_arr = (
            _to_array(pose_res.pose_landmarks[0], POSE_LMS)
            if pose_res.pose_landmarks
            else None
        )

        left = right = None
        for lm_list, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
            arr = _to_array(lm_list, HAND_LMS)
            label = hd[0].category_name if hd else ""
            if label == "Left" and left is None:
                left = arr
            elif label == "Right" and right is None:
                right = arr

        return FrameLandmarks(
            pose=pose_arr,
            left_hand=left,
            right_hand=right,
            face_detections=face_list,
        )
