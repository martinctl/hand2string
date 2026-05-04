"""End-to-end demo: pull the dataset from HF, run MediaPipe Holistic on one
clip, and write an mp4 with the skeleton overlay + the English transcript
baked in as a caption.

This script depends on:
  - huggingface_hub  (to fetch the dataset)
  - pandas           (to read metadata)
  - mediapipe + cv2  (skeleton + drawing)

so a teammate can copy it standalone and run:

    python examples/visualize_one_sentence.py
    python examples/visualize_one_sentence.py --sentence-id --7E2sU6zP4_10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.holistic_tasks import (
    FACE_DETECTOR_KEYPOINT_DRAW_INDICES,
    FACE_FROM_POSE_INDICES,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    Holistic,
)

REPO_ID = "martinctl/how2sign-asl-clips"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sentence-id", default=None,
                        help="which sentence to visualize (default: first row)")
    parser.add_argument("--out", type=Path, default=Path("out"))
    return parser.parse_args()


def _draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray | None,
    connections: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    point_radius: int = 3,
    line_thickness: int = 2,
    *,
    skip_pose_face: bool = False,
) -> None:
    if landmarks is None:
        return
    h, w = frame_bgr.shape[:2]
    pts = np.column_stack([landmarks[:, 0] * w, landmarks[:, 1] * h]).astype(int)
    face_ix = set(FACE_FROM_POSE_INDICES) if skip_pose_face else set()
    for a, b in connections:
        if skip_pose_face and a in face_ix and b in face_ix:
            continue
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, tuple(pts[a]), tuple(pts[b]), color, line_thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        if skip_pose_face and i in face_ix:
            continue
        cv2.circle(frame_bgr, (int(x), int(y)), point_radius, color, -1, cv2.LINE_AA)


def _draw_face_detections(
    bgr: np.ndarray,
    detections,
    *,
    box_color: tuple[int, int, int] = (0, 255, 255),
    box_thickness: int = 2,
    point_radius: int = 4,
) -> None:
    """BlazeFace-style overlay: axis-aligned bbox + six keypoints per face."""
    if not detections:
        return
    h, w = bgr.shape[:2]
    kpt_colors = (
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 165, 255),
        (255, 255, 0),
    )
    for det in detections:
        bb = det.bounding_box
        x1, y1 = bb.origin_x, bb.origin_y
        x2, y2 = bb.origin_x + bb.width, bb.origin_y + bb.height
        cv2.rectangle(bgr, (x1, y1), (x2, y2), box_color, box_thickness, cv2.LINE_AA)
        if not det.keypoints:
            continue
        for i, kp in enumerate(det.keypoints):
            if i not in FACE_DETECTOR_KEYPOINT_DRAW_INDICES:
                continue
            if kp.x is None or kp.y is None:
                continue
            px = int(kp.x * w)
            py = int(kp.y * h)
            col = kpt_colors[i % len(kpt_colors)]
            cv2.circle(bgr, (px, py), point_radius, col, -1, cv2.LINE_AA)
            cv2.circle(bgr, (px, py), point_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_caption(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    bar_h = max(48, h // 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, w / 1280.0)
    thickness = max(1, int(scale * 2))
    margin = int(20 * scale)

    words = text.split()
    lines: list[str] = []
    current = ""
    max_text_w = w - 2 * margin
    for word in words:
        candidate = (current + " " + word).strip()
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw > max_text_w and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    lines = lines[-2:]  # show at most the last two wrapped lines

    line_h = int(28 * scale)
    y = h - bar_h + line_h
    for line in lines:
        cv2.putText(frame, line, (margin, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h + int(4 * scale)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.repo_id} from the Hub (cached after first call)...")
    local = Path(snapshot_download(args.repo_id, repo_type="dataset"))
    df = pd.read_parquet(local / "metadata.parquet")
    df = df[df["split"] == args.split].reset_index(drop=True)

    if args.sentence_id is None:
        row = df.iloc[0]
    else:
        match = df[df["sentence_id"] == args.sentence_id]
        if match.empty:
            raise SystemExit(f"sentence_id {args.sentence_id!r} not in split {args.split!r}")
        row = match.iloc[0]

    clip_path = local / row["file_name"]
    print(f"Visualizing {row['sentence_id']} ({row['duration']:.1f}s)")
    print(f"  transcript: {row['sentence']}")

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out / f"{row['sentence_id']}_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    pose_color = (0, 255, 0)        # green
    left_color = (0, 165, 255)      # orange (BGR)
    right_color = (255, 128, 0)     # cyan-ish (BGR)

    n = 0
    # Face Detector (BlazeFace full-range): bounding box + 6 keypoints per face.
    with Holistic(fps=fps, include_face=True) as h:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lms = h.process(rgb)

            _draw_skeleton(
                bgr, lms.pose, POSE_CONNECTIONS, pose_color, skip_pose_face=True
            )
            _draw_skeleton(bgr, lms.left_hand, HAND_CONNECTIONS, left_color)
            _draw_skeleton(bgr, lms.right_hand, HAND_CONNECTIONS, right_color)
            _draw_face_detections(bgr, lms.face_detections)

            draw_caption(bgr, row["sentence"])
            writer.write(bgr)
            n += 1

    cap.release()
    writer.release()

    print(f"Wrote {out_path}  ({n} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    main()
