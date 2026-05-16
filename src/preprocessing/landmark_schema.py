"""Shared schema for the preprocessed How2Sign landmark dataset."""
from __future__ import annotations

from dataclasses import dataclass

POSE_LMS = 33
HAND_LMS = 21

# ASL-relevant Face Landmarker indices: lips, eyebrows, eyes, and nose.
ASL_FACE_LANDMARKS: tuple[int, ...] = tuple(sorted({
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    70, 63, 105, 66, 107, 55, 65,
    300, 293, 334, 296, 336, 285, 295,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    1, 2, 4, 5, 6,
}))


@dataclass(frozen=True)
class LandmarkBlock:
    name: str
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


POSE_BLOCK = LandmarkBlock("pose", 0, POSE_LMS)
LEFT_HAND_BLOCK = LandmarkBlock("left_hand", POSE_BLOCK.end, POSE_BLOCK.end + HAND_LMS)
RIGHT_HAND_BLOCK = LandmarkBlock("right_hand", LEFT_HAND_BLOCK.end, LEFT_HAND_BLOCK.end + HAND_LMS)
FACE_BLOCK = LandmarkBlock("face_asl", RIGHT_HAND_BLOCK.end, RIGHT_HAND_BLOCK.end + len(ASL_FACE_LANDMARKS))

LANDMARK_BLOCKS: tuple[LandmarkBlock, ...] = (
    POSE_BLOCK,
    LEFT_HAND_BLOCK,
    RIGHT_HAND_BLOCK,
    FACE_BLOCK,
)
TOTAL_LMS = FACE_BLOCK.end

SCHEMA_VERSION = "how2sign-landmarks-v1"

ARRAY_KEYS: tuple[str, ...] = (
    "landmarks_image",
    "landmarks_world",
    "features_geometric",
    "valid_mask",
    "timestamps_ms",
    "handedness_scores",
)
