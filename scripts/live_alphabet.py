"""Live ASL alphabet recognition: webcam -> MediaPipe hand -> MLP -> letter.

Run after training:
    python scripts/live_alphabet.py --ckpt runs/mlp_alphabet/best.pt
Press q to quit. Use --mirror for selfie view.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.models.mlp import MLP
from src.preprocessing.holistic_tasks import (
    HAND_CONNECTIONS, HAND_MODEL_URL, _ensure_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/mlp_alphabet/best.pt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--smooth", type=int, default=5,
                   help="majority vote over last N predictions for stability")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def normalize_hand(vec63: np.ndarray) -> np.ndarray:
    pts = vec63.reshape(21, 3).copy()
    pts -= pts[0:1, :]
    scale = np.linalg.norm(pts, axis=-1).max()
    return (pts / max(scale, 1e-6)).flatten().astype(np.float32)


def main() -> None:
    args = parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    blob = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    classes = blob["classes"]
    model = MLP(input_dim=blob["input_dim"], hidden_dim=blob["hidden_dim"],
                num_classes=len(classes)).to(args.device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    print(f"Loaded {args.ckpt}  classes={len(classes)}  device={args.device}")

    hand_path = _ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
    hand = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    history: deque = deque(maxlen=args.smooth)
    fps_buf: deque = deque(maxlen=30)
    dt_ms = max(1, int(round(1000.0 / args.fps)))
    t_ms = 0
    win = "hand2string live alphabet [q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            ok, bgr = cap.read()
            if not ok:
                break
            if args.mirror:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = bgr.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = hand.detect_for_video(mp_img, t_ms)
            t_ms += dt_ms

            label = "?"
            conf = 0.0
            if res.hand_landmarks:
                lms = res.hand_landmarks[0]
                vec = np.array([[p.x, p.y, p.z] for p in lms[:21]],
                               dtype=np.float32).flatten()
                # draw skeleton
                pts = np.column_stack([vec.reshape(21, 3)[:, 0] * w,
                                       vec.reshape(21, 3)[:, 1] * h]).astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(bgr, tuple(pts[a]), tuple(pts[b]),
                             (255, 128, 0), 2, cv2.LINE_AA)
                for x, y in pts:
                    cv2.circle(bgr, (int(x), int(y)), 3, (0, 255, 255), -1)

                with torch.no_grad():
                    x = torch.from_numpy(normalize_hand(vec))[None].to(args.device)
                    probs = torch.softmax(model(x), -1)[0].cpu().numpy()
                pred = int(probs.argmax())
                conf = float(probs[pred])
                history.append(pred)
                # majority vote on history for stability
                vals, counts = np.unique(np.array(history), return_counts=True)
                smoothed = int(vals[counts.argmax()])
                label = str(classes[smoothed])

            fps_buf.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            cv2.rectangle(bgr, (0, 0), (w, 110), (0, 0, 0), -1)
            cv2.putText(bgr, f"{label}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, f"conf={conf:.2f}  fps={np.mean(fps_buf):.1f}",
                        (180, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if res.hand_landmarks:
                top3 = np.argsort(-probs)[:3]
                top_str = "  ".join(f"{classes[i]}:{probs[i]:.2f}" for i in top3)
                cv2.putText(bgr, top_str, (15, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)

            cv2.imshow(win, bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hand.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
