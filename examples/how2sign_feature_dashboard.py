"""Generate a local HTML dashboard for inspecting How2Sign landmark features.

The dashboard uses an already-built landmarks dataset. It does not run
MediaPipe; it reads NPZ shard arrays, computes deterministic feature recipes,
and syncs them with the original source video.

Example:
    python examples/how2sign_feature_dashboard.py \
        --dataset data/how2sign_landmarks_smoke \
        --videos-root .. \
        --split train \
        --out outputs/demo/how2sign_feature_dashboard.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.feature_recipes import (
    FINGER_NAMES,
    arm_angles,
    face_cues,
    finger_curl,
    hand_body_distances,
    hand_openness,
    temporal_delta,
    two_hand_features,
)
from src.preprocessing.landmark_schema import (
    ASL_FACE_LANDMARKS,
    FACE_BLOCK,
    LEFT_HAND_BLOCK,
    POSE_BLOCK,
    RIGHT_HAND_BLOCK,
)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
FACE_CONNECTION_SOURCE = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (70, 63), (63, 105), (105, 66), (66, 107),
    (300, 293), (293, 334), (334, 296), (296, 336),
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    (362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "how2sign_landmarks_smoke")
    parser.add_argument("--videos-root", type=Path, default=REPO_ROOT.parent)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sentence-id", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "demo" / "how2sign_feature_dashboard.html")
    return parser.parse_args()


def _video_path(videos_root: Path, row: pd.Series) -> Path:
    return videos_root / f"{row['split']}_raw_videos" / "raw_videos" / f"{row['video_name']}.mp4"


def _load_arrays(dataset: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shard = np.load(dataset / row["shard_file"])
    prefix = row["sample_key"]
    landmarks = shard[f"{prefix}__landmarks_image"].astype(np.float32)
    mask = shard[f"{prefix}__valid_mask"].astype(bool)
    timestamps_ms = shard[f"{prefix}__timestamps_ms"].astype(np.float32)
    hand_scores = shard[f"{prefix}__handedness_scores"].astype(np.float32)
    return landmarks, mask, timestamps_ms, hand_scores


def _connections() -> list[dict]:
    conns = []
    conns.extend({"a": a, "b": b, "kind": "pose"} for a, b in POSE_CONNECTIONS)
    conns.extend(
        {"a": LEFT_HAND_BLOCK.start + a, "b": LEFT_HAND_BLOCK.start + b, "kind": "left"}
        for a, b in HAND_CONNECTIONS
    )
    conns.extend(
        {"a": RIGHT_HAND_BLOCK.start + a, "b": RIGHT_HAND_BLOCK.start + b, "kind": "right"}
        for a, b in HAND_CONNECTIONS
    )
    face_index = {idx: i for i, idx in enumerate(ASL_FACE_LANDMARKS)}
    conns.extend(
        {
            "a": FACE_BLOCK.start + face_index[a],
            "b": FACE_BLOCK.start + face_index[b],
            "kind": "face",
        }
        for a, b in FACE_CONNECTION_SOURCE
        if a in face_index and b in face_index
    )
    return conns


def _round_array(values: np.ndarray, digits: int = 4) -> list:
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(clean, digits).tolist()


def _relative_url(from_dir: Path, target: Path) -> str:
    rel = os.path.relpath(target.resolve(), from_dir.resolve())
    return quote(Path(rel).as_posix(), safe="/:._-")


def _features(landmarks: np.ndarray, timestamps_ms: np.ndarray) -> dict:
    left_curl = finger_curl(landmarks, "left")
    right_curl = finger_curl(landmarks, "right")
    left_ext = 1.0 - left_curl
    right_ext = 1.0 - right_curl
    left_d = hand_body_distances(landmarks, "left")
    right_d = hand_body_distances(landmarks, "right")
    two = two_hand_features(landmarks)
    arms = arm_angles(landmarks)
    face = face_cues(landmarks)
    left_wrist = landmarks[:, LEFT_HAND_BLOCK.start, :]
    right_wrist = landmarks[:, RIGHT_HAND_BLOCK.start, :]
    dt = np.diff(timestamps_ms / 1000.0, prepend=timestamps_ms[:1] / 1000.0)
    dt = np.maximum(dt, 1e-6)
    left_speed = np.linalg.norm(temporal_delta(left_wrist, 1), axis=-1) / dt
    right_speed = np.linalg.norm(temporal_delta(right_wrist, 1), axis=-1) / dt
    return {
        "fingerNames": list(FINGER_NAMES),
        "distanceNames": ["left shoulder", "right shoulder", "nose", "mouth", "torso"],
        "twoHandNames": ["wrists", "palms", "thumb tips", "index tips", "middle tips", "ring tips", "pinky tips"],
        "faceNames": ["mouth opening", "mouth width", "eye distance"],
        "leftExtension": _round_array(left_ext),
        "rightExtension": _round_array(right_ext),
        "leftCurl": _round_array(left_curl),
        "rightCurl": _round_array(right_curl),
        "leftOpenness": _round_array(hand_openness(landmarks, "left")),
        "rightOpenness": _round_array(hand_openness(landmarks, "right")),
        "leftDistances": _round_array(left_d),
        "rightDistances": _round_array(right_d),
        "twoHand": _round_array(two),
        "armDegrees": _round_array(np.rad2deg(arms[:, :2]), 1),
        "armRatios": _round_array(arms[:, 2:]),
        "face": _round_array(face),
        "leftSpeed": _round_array(left_speed),
        "rightSpeed": _round_array(right_speed),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How2Sign Landmark Feature Dashboard</title>
<style>
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f5f7f8;
  color: #172026;
}
body { margin: 0; }
.shell { max-width: 1240px; margin: 0 auto; padding: 24px; }
.topbar { display: flex; gap: 18px; align-items: flex-end; justify-content: space-between; margin-bottom: 16px; }
h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: 0; }
.sentence { margin: 0; color: #52616b; max-width: 760px; line-height: 1.45; }
.meta { display: grid; grid-template-columns: repeat(4, max-content); gap: 8px 16px; font-size: 13px; color: #52616b; }
.stage { display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(320px, .75fr); gap: 18px; align-items: start; }
.video-wrap { position: relative; background: #0b0f12; border-radius: 8px; overflow: hidden; box-shadow: 0 14px 36px rgba(14, 30, 37, .16); }
video { width: 100%; display: block; }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.panel, .card { background: #fff; border: 1px solid #d9e1e6; border-radius: 8px; box-shadow: 0 10px 24px rgba(14, 30, 37, .08); }
.panel { padding: 16px; }
.readout { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 14px; }
.metric { border: 1px solid #e2e8ec; border-radius: 8px; padding: 10px; background: #fbfcfd; }
.label { font-size: 12px; color: #64727c; margin-bottom: 4px; }
.value { font-variant-numeric: tabular-nums; font-size: 20px; font-weight: 700; }
.cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 16px; }
.card { padding: 14px; }
.card h2 { font-size: 16px; margin: 0 0 10px; }
.finger { display: grid; grid-template-columns: 68px 1fr 44px; align-items: center; gap: 10px; margin: 8px 0; }
.finger span, .tiny { font-size: 12px; color: #52616b; }
.bar { height: 10px; border-radius: 999px; background: #edf2f5; overflow: hidden; }
.fill { height: 100%; width: 0%; border-radius: inherit; background: linear-gradient(90deg, #31a37c, #e3a72f); }
.dist-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.dist { border-top: 1px solid #ecf0f2; padding-top: 8px; }
.dist b { display: block; font-size: 15px; font-variant-numeric: tabular-nums; }
.dist span { font-size: 12px; color: #64727c; }
.timeline { margin-top: 12px; }
input[type="range"] { width: 100%; accent-color: #247c68; }
.legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; font-size: 12px; color: #52616b; }
.dot { width: 9px; height: 9px; border-radius: 99px; display: inline-block; margin-right: 5px; }
.pose { background: #2b8fd9; } .left { background: #22a06b; } .right { background: #d45b7a; } .face { background: #e0a323; }
@media (max-width: 940px) { .stage, .cards { grid-template-columns: 1fr; } .meta { grid-template-columns: repeat(2, max-content); } }
</style>
</head>
<body>
<main class="shell">
  <div class="topbar">
    <div>
      <h1>How2Sign Landmark Feature Dashboard</h1>
      <p class="sentence" id="sentence"></p>
    </div>
    <div class="meta" id="meta"></div>
  </div>
  <section class="stage">
    <div>
      <div class="video-wrap">
        <video id="video" controls muted preload="metadata"></video>
        <canvas id="overlay"></canvas>
      </div>
      <div class="timeline">
        <input id="scrub" type="range" min="0" max="0" value="0" step="1">
        <div class="legend">
          <span><i class="dot pose"></i>pose</span>
          <span><i class="dot left"></i>left hand</span>
          <span><i class="dot right"></i>right hand</span>
          <span><i class="dot face"></i>face subset</span>
        </div>
      </div>
    </div>
    <aside class="panel">
      <div class="readout">
        <div class="metric"><div class="label">Frame</div><div class="value" id="frameNo">0</div></div>
        <div class="metric"><div class="label">Time</div><div class="value" id="timeNo">0.00s</div></div>
        <div class="metric"><div class="label">Left speed</div><div class="value" id="leftSpeed">0.00</div></div>
        <div class="metric"><div class="label">Right speed</div><div class="value" id="rightSpeed">0.00</div></div>
      </div>
      <div class="dist-grid" id="summary"></div>
    </aside>
  </section>
  <section class="cards">
    <div class="card"><h2>Left Hand Extension</h2><div id="leftFingers"></div></div>
    <div class="card"><h2>Right Hand Extension</h2><div id="rightFingers"></div></div>
    <div class="card"><h2>Hand To Body Distances</h2><div class="dist-grid" id="distances"></div></div>
    <div class="card"><h2>Two-Hand, Arms, Face</h2><div class="dist-grid" id="relations"></div></div>
  </section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data = JSON.parse(document.getElementById("payload").textContent);
const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const scrub = document.getElementById("scrub");
video.src = data.videoSrc;
document.getElementById("sentence").textContent = data.sentence;
document.getElementById("meta").innerHTML = [
  ["split", data.split], ["sentence", data.sentenceId],
  ["frames", data.timestamps.length], ["source fps", data.sourceFps.toFixed(2)]
].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
scrub.max = Math.max(0, data.timestamps.length - 1);

function nearestFrame() {
  const sourceMs = video.currentTime * 1000;
  let lo = 0, hi = data.timestamps.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (data.timestamps[mid] < sourceMs) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(data.timestamps[lo - 1] - sourceMs) < Math.abs(data.timestamps[lo] - sourceMs)) return lo - 1;
  return lo;
}
function color(kind) {
  return {pose:"#2b8fd9", left:"#22a06b", right:"#d45b7a", face:"#e0a323"}[kind] || "#fff";
}
function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
}
function drawLandmarks(i) {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2.2 * devicePixelRatio;
  const pts = data.landmarks[i];
  const mask = data.mask[i];
  for (const c of data.connections) {
    if (!mask[c.a] || !mask[c.b]) continue;
    const a = pts[c.a], b = pts[c.b];
    ctx.strokeStyle = color(c.kind);
    ctx.globalAlpha = .9;
    ctx.beginPath();
    ctx.moveTo(a[0] * canvas.width, a[1] * canvas.height);
    ctx.lineTo(b[0] * canvas.width, b[1] * canvas.height);
    ctx.stroke();
  }
  for (let p = 0; p < pts.length; p++) {
    if (!mask[p]) continue;
    let kind = p < 33 ? "pose" : p < 54 ? "left" : p < 75 ? "right" : "face";
    ctx.fillStyle = color(kind);
    ctx.globalAlpha = kind === "face" ? .72 : .92;
    ctx.beginPath();
    ctx.arc(pts[p][0] * canvas.width, pts[p][1] * canvas.height, (kind === "face" ? 2.2 : 3.2) * devicePixelRatio, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
function fingerRows(el, names, values) {
  el.innerHTML = names.map((name, idx) => {
    const v = Math.max(0, Math.min(1, values[idx] ?? 0));
    return `<div class="finger"><span>${name}</span><div class="bar"><div class="fill" style="width:${Math.round(v*100)}%"></div></div><b>${v.toFixed(2)}</b></div>`;
  }).join("");
}
function metric(label, value, suffix="") {
  return `<div class="dist"><b>${value}${suffix}</b><span>${label}</span></div>`;
}
function updateFeatures(i) {
  const f = data.features;
  document.getElementById("frameNo").textContent = i;
  document.getElementById("timeNo").textContent = Math.max(0, (data.timestamps[i] / 1000) - data.start).toFixed(2) + "s";
  document.getElementById("leftSpeed").textContent = f.leftSpeed[i].toFixed(2);
  document.getElementById("rightSpeed").textContent = f.rightSpeed[i].toFixed(2);
  fingerRows(document.getElementById("leftFingers"), f.fingerNames, f.leftExtension[i]);
  fingerRows(document.getElementById("rightFingers"), f.fingerNames, f.rightExtension[i]);
  document.getElementById("summary").innerHTML =
    metric("left openness", f.leftOpenness[i].toFixed(2)) +
    metric("right openness", f.rightOpenness[i].toFixed(2)) +
    metric("left elbow", f.armDegrees[i][0].toFixed(1), " deg") +
    metric("right elbow", f.armDegrees[i][1].toFixed(1), " deg");
  document.getElementById("distances").innerHTML = f.distanceNames.map((name, idx) =>
    metric(`L wrist to ${name}`, f.leftDistances[i][idx].toFixed(3)) +
    metric(`R wrist to ${name}`, f.rightDistances[i][idx].toFixed(3))
  ).join("");
  document.getElementById("relations").innerHTML =
    f.twoHandNames.map((name, idx) => metric(name, f.twoHand[i][idx].toFixed(3))).join("") +
    metric("left forearm / shoulders", f.armRatios[i][0].toFixed(3)) +
    metric("right forearm / shoulders", f.armRatios[i][1].toFixed(3)) +
    f.faceNames.map((name, idx) => metric(name, f.face[i][idx].toFixed(3))).join("");
}
function render(i = nearestFrame()) {
  i = Math.max(0, Math.min(data.timestamps.length - 1, i));
  scrub.value = i;
  drawLandmarks(i);
  updateFeatures(i);
}
video.addEventListener("loadedmetadata", () => {
  video.currentTime = data.start;
  render(0);
});
video.addEventListener("timeupdate", () => {
  if (video.currentTime > data.end) video.pause();
  render();
});
video.addEventListener("play", () => {
  const loop = () => { if (!video.paused && !video.ended) { render(); requestAnimationFrame(loop); } };
  requestAnimationFrame(loop);
});
window.addEventListener("resize", () => render());
scrub.addEventListener("input", () => {
  const i = Number(scrub.value);
  video.currentTime = Math.max(0, data.timestamps[i] / 1000);
  render(i);
});
render(0);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    meta = pd.read_parquet(args.dataset / "metadata.parquet")
    if args.sentence_id is not None:
        rows = meta[meta["sentence_id"] == args.sentence_id]
    elif args.split is not None:
        rows = meta[meta["split"] == args.split]
    else:
        rows = meta
    if rows.empty:
        raise SystemExit("no matching row in metadata")
    row = rows.reset_index(drop=True).iloc[args.index]
    video_path = _video_path(args.videos_root.resolve(), row)
    if not video_path.exists():
        raise SystemExit(f"source video not found: {video_path}")

    landmarks, mask, timestamps_ms, _hand_scores = _load_arrays(args.dataset, row)
    payload = {
        "videoSrc": _relative_url(args.out.parent, video_path),
        "split": row["split"],
        "sentenceId": row["sentence_id"],
        "sentence": row["sentence"],
        "start": float(row["start"]),
        "end": float(row["end"]),
        "sourceFps": float(row["source_fps"]),
        "timestamps": _round_array(timestamps_ms, 1),
        "landmarks": _round_array(landmarks, 5),
        "mask": mask.astype(int).tolist(),
        "connections": _connections(),
        "features": _features(landmarks, timestamps_ms),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload)),
        encoding="utf-8",
    )
    print(f"Dashboard written to: {args.out.resolve()}")
    print(f"Open: {args.out.resolve().as_uri()}")


if __name__ == "__main__":
    main()
