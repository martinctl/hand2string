"""Live webcam demo: pose + hands + BlazeFace (box + 6 keypoints), same style as
``visualize_one_sentence.py`` but without Hugging Face or file I/O.

Run from the repo root (conda env ``hand2string``):

    python examples/live_webcam.py

Press **q** in the window to quit. Use ``--mirror`` for a selfie-style horizontal flip.

If **no window appears** (common over SSH, some IDE runs, or headless Linux), use::

    python examples/live_webcam.py --no-window --record out/live.mp4

Then open ``out/live.mp4``. On macOS, grant **Camera** (and if needed **Screen Recording**)
access to the app that launches Python (Terminal, iTerm, Cursor, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.holistic_tasks import (  # noqa: E402
    FACE_DETECTOR_KEYPOINT_DRAW_INDICES,
    FACE_FROM_POSE_INDICES,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    Holistic,
)

WIN_NAME = "hand2string live [q=quit]"


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


def _gui_prepare_window(name: str) -> None:
    """Create the window early and try to raise it (best-effort across platforms)."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass


def _headless_hint() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        print("Note: DISPLAY is unset; GUI windows usually need X11/Wayland or use --no-window.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Video clock for MediaPipe VIDEO mode (default: from camera, else 30)",
    )
    p.add_argument(
        "--mirror",
        action="store_true",
        help="Flip frame horizontally (selfie view); inference runs on the flipped image",
    )
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Skip imshow (SSH/headless); use --record to save mp4; stop with Ctrl+C",
    )
    p.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Save overlay video to this mp4 path (created if missing). With --no-window, "
        "defaults to out/live_webcam_overlay.mp4 when omitted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _headless_hint()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps is not None else (cam_fps if cam_fps and cam_fps > 1 else 30.0)

    show_gui = not args.no_window
    record_path = args.record
    if args.no_window and record_path is None:
        record_path = REPO_ROOT / "out" / "live_webcam_overlay.mp4"

    pose_color = (0, 255, 0)
    left_color = (0, 165, 255)
    right_color = (255, 128, 0)

    writer: cv2.VideoWriter | None = None
    window_ready = False

    print(f"MediaPipe clock {fps:.1f} fps.")
    if show_gui:
        print("GUI: focus the OpenCV window and press 'q' to quit.")
        if sys.platform == "darwin":
            print("macOS: if the window never appears, run from Terminal.app or grant permissions to that app.")
    else:
        print(f"Headless: writing to {record_path} - press Ctrl+C to stop.")

    try:
        with Holistic(fps=fps, include_face=True) as h:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    print("Frame grab failed; exiting.")
                    break
                if args.mirror:
                    bgr = cv2.flip(bgr, 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                lms = h.process(rgb)

                _draw_skeleton(
                    bgr, lms.pose, POSE_CONNECTIONS, pose_color, skip_pose_face=True
                )
                _draw_skeleton(bgr, lms.left_hand, HAND_CONNECTIONS, left_color)
                _draw_skeleton(bgr, lms.right_hand, HAND_CONNECTIONS, right_color)
                _draw_face_detections(bgr, lms.face_detections)

                cv2.putText(
                    bgr,
                    "q: quit",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if record_path is not None:
                    if writer is None:
                        record_path.parent.mkdir(parents=True, exist_ok=True)
                        h0, w0 = bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(record_path), fourcc, fps, (w0, h0)
                        )
                        if not writer.isOpened():
                            raise SystemExit(
                                f"Could not open VideoWriter for {record_path}"
                            )
                    writer.write(bgr)

                if show_gui:
                    if not window_ready:
                        _gui_prepare_window(WIN_NAME)
                        window_ready = True
                    try:
                        cv2.imshow(WIN_NAME, bgr)
                    except cv2.error as e:
                        print(f"cv2.imshow failed ({e}). Try: --no-window --record out/live.mp4")
                        try:
                            cv2.destroyWindow(WIN_NAME)
                        except cv2.error:
                            pass
                        window_ready = False
                        show_gui = False
                        if record_path is None:
                            record_path = REPO_ROOT / "out" / "live_webcam_overlay.mp4"
                            print(f"Saving to {record_path} instead; Ctrl+C to stop.")
                        continue
                    # ~60 Hz UI poll; waitKey(1) often fails to show a window on some macOS setups
                    if cv2.waitKey(16) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Wrote {record_path}")
        if window_ready:
            try:
                cv2.destroyWindow(WIN_NAME)
            except cv2.error:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
