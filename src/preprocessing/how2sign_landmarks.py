"""Build a landmarks-only How2Sign dataset from local source videos."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from src.preprocessing.feature_recipes import GEOMETRIC_FEATURE_NAMES, geometric_feature_block
from src.preprocessing.landmark_schema import (
    ARRAY_KEYS,
    ASL_FACE_LANDMARKS,
    FACE_BLOCK,
    HAND_LMS,
    LANDMARK_BLOCKS,
    LEFT_HAND_BLOCK,
    POSE_BLOCK,
    POSE_LMS,
    RIGHT_HAND_BLOCK,
    SCHEMA_VERSION,
    TOTAL_LMS,
)

POSE_MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
}
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass
class ExtractedSegment:
    landmarks_image: np.ndarray
    landmarks_world: np.ndarray
    features_geometric: np.ndarray
    valid_mask: np.ndarray
    timestamps_ms: np.ndarray
    handedness_scores: np.ndarray
    source_fps: float
    frame_width: int
    frame_height: int


@dataclass
class ExtractedVideo:
    landmarks_image: np.ndarray
    landmarks_world: np.ndarray
    features_geometric: np.ndarray
    valid_mask: np.ndarray
    timestamps_ms: np.ndarray
    handedness_scores: np.ndarray
    source_fps: float
    frame_width: int
    frame_height: int


def model_cache_dir() -> Path:
    override = os.environ.get("HAND2STRING_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "hand2string" / "mediapipe"


def ensure_model(url: str, name: str) -> Path:
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        print(f"Downloading MediaPipe model: {name}")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _landmarks_to_array(landmarks, n: int) -> np.ndarray:
    return np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks[:n]], dtype=np.float32)


def _empty_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((TOTAL_LMS, 3), np.nan, dtype=np.float32)
    world = np.full((TOTAL_LMS, 3), np.nan, dtype=np.float32)
    mask = np.zeros((TOTAL_LMS,), dtype=np.bool_)
    handedness = np.full((2,), np.nan, dtype=np.float32)
    return image, world, mask, handedness


class SegmentLandmarkExtractor:
    """Stateful MediaPipe extractor for one source video at a time."""

    def __init__(self, target_fps: float = 25.0, *, pose_model: str = "lite", include_face: bool = True):
        self.target_fps = float(target_fps)
        self.include_face = bool(include_face)
        self._mp_timestamp_ms = 0
        self._mp_step_ms = max(1, int(round(1000.0 / self.target_fps)))
        if pose_model not in POSE_MODEL_URLS:
            raise ValueError(f"pose_model must be one of {sorted(POSE_MODEL_URLS)}")
        pose_path = ensure_model(POSE_MODEL_URLS[pose_model], f"pose_landmarker_{pose_model}.task")
        hand_path = ensure_model(HAND_MODEL_URL, "hand_landmarker.task")
        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.face: mp_vision.FaceLandmarker | None = None
        if self.include_face:
            face_path = ensure_model(FACE_MODEL_URL, "face_landmarker.task")
            self.face = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(face_path)),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

    def close(self) -> None:
        self.pose.close()
        self.hand.close()
        if self.face is not None:
            self.face.close()

    def __enter__(self) -> "SegmentLandmarkExtractor":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def reset_clock(self) -> None:
        self._mp_timestamp_ms = 0

    def _process_frame(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = self._mp_timestamp_ms
        self._mp_timestamp_ms += self._mp_step_ms
        image, world, mask, handedness_scores = _empty_frame()

        pose_res = self.pose.detect_for_video(mp_image, timestamp_ms)
        if pose_res.pose_landmarks:
            arr = _landmarks_to_array(pose_res.pose_landmarks[0], POSE_LMS)
            image[POSE_BLOCK.start:POSE_BLOCK.end] = arr
            mask[POSE_BLOCK.start:POSE_BLOCK.end] = True
        if getattr(pose_res, "pose_world_landmarks", None):
            arr = _landmarks_to_array(pose_res.pose_world_landmarks[0], POSE_LMS)
            world[POSE_BLOCK.start:POSE_BLOCK.end] = arr

        hand_res = self.hand.detect_for_video(mp_image, timestamp_ms)
        hand_world = getattr(hand_res, "hand_world_landmarks", None) or []
        for i, (lm_list, handedness) in enumerate(zip(hand_res.hand_landmarks, hand_res.handedness)):
            categories = list(handedness)
            label = categories[0].category_name if categories else ""
            score = float(categories[0].score) if categories else math.nan
            block = LEFT_HAND_BLOCK if label == "Left" else RIGHT_HAND_BLOCK if label == "Right" else None
            if block is None:
                continue
            image[block.start:block.end] = _landmarks_to_array(lm_list, HAND_LMS)
            mask[block.start:block.end] = True
            handedness_scores[0 if block is LEFT_HAND_BLOCK else 1] = score
            if i < len(hand_world):
                world[block.start:block.end] = _landmarks_to_array(hand_world[i], HAND_LMS)

        face_res = self.face.detect_for_video(mp_image, timestamp_ms) if self.face is not None else None
        if face_res is not None and face_res.face_landmarks:
            all_face = _landmarks_to_array(face_res.face_landmarks[0], len(face_res.face_landmarks[0]))
            face = all_face[list(ASL_FACE_LANDMARKS)]
            image[FACE_BLOCK.start:FACE_BLOCK.end] = face
            mask[FACE_BLOCK.start:FACE_BLOCK.end] = True

        return image, world, mask, handedness_scores

    def extract_segment(self, video_path: Path, start_s: float, end_s: float) -> ExtractedSegment:
        video = self.extract_video_range(video_path, start_s, end_s)
        segment = slice_video_segment(video, start_s, end_s)
        if segment.landmarks_image.shape[0] == 0:
            raise RuntimeError(f"no frames decoded for {video_path.name} [{start_s}, {end_s}]")
        return segment

    def extract_video_range(self, video_path: Path, start_s: float, end_s: float) -> ExtractedVideo:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.target_fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if end_s <= start_s:
                raise ValueError(f"invalid video timestamps: start={start_s}, end={end_s}")
            self.reset_clock()

            images: list[np.ndarray] = []
            worlds: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            handedness: list[np.ndarray] = []
            timestamps: list[float] = []

            target_step = 1.0 / self.target_fps
            next_sample_s = float(start_s)
            half_source_frame = 0.5 / max(source_fps, 1.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(start_s) * 1000.0))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms <= 0:
                    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    pos_s = frame_idx / max(source_fps, 1.0)
                else:
                    pos_s = pos_ms / 1000.0
                if pos_s >= float(end_s):
                    if images or pos_s > float(end_s) + half_source_frame:
                        break
                if pos_s + half_source_frame < next_sample_s:
                    continue
                ts_ms = int(round(pos_s * 1000.0))
                img, world, mask, hand_scores = self._process_frame(frame)
                images.append(img)
                worlds.append(world)
                masks.append(mask)
                handedness.append(hand_scores)
                timestamps.append(float(ts_ms))
                while next_sample_s <= pos_s + half_source_frame:
                    next_sample_s += target_step

            if not images:
                raise RuntimeError(f"no frames decoded for {video_path.name} [{start_s}, {end_s}]")

            landmarks_image = np.stack(images).astype(np.float32)
            landmarks_world = np.stack(worlds).astype(np.float32)
            features_geometric = geometric_feature_block(np.nan_to_num(landmarks_image, nan=0.0))
            return ExtractedVideo(
                landmarks_image=landmarks_image,
                landmarks_world=landmarks_world,
                features_geometric=features_geometric,
                valid_mask=np.stack(masks).astype(np.bool_),
                timestamps_ms=np.asarray(timestamps, dtype=np.float32),
                handedness_scores=np.stack(handedness).astype(np.float32),
                source_fps=source_fps,
                frame_width=width,
                frame_height=height,
            )
        finally:
            cap.release()


def slice_video_segment(video: ExtractedVideo, start_s: float, end_s: float) -> ExtractedSegment:
    start_ms = float(start_s) * 1000.0
    end_ms = float(end_s) * 1000.0
    keep = (video.timestamps_ms >= start_ms - 1e-3) & (video.timestamps_ms < end_ms - 1e-3)
    if not np.any(keep):
        midpoint = 0.5 * (start_ms + end_ms)
        in_neighborhood = (video.timestamps_ms >= start_ms - 25.0) & (video.timestamps_ms <= end_ms + 25.0)
        if np.any(in_neighborhood):
            candidates = np.flatnonzero(in_neighborhood)
            nearest = candidates[np.argmin(np.abs(video.timestamps_ms[candidates] - midpoint))]
            indices = np.asarray([nearest], dtype=np.int64)
        else:
            indices = np.zeros((0,), dtype=np.int64)
    else:
        indices = np.flatnonzero(keep)
    return ExtractedSegment(
        landmarks_image=video.landmarks_image[indices],
        landmarks_world=video.landmarks_world[indices],
        features_geometric=video.features_geometric[indices],
        valid_mask=video.valid_mask[indices],
        timestamps_ms=video.timestamps_ms[indices],
        handedness_scores=video.handedness_scores[indices],
        source_fps=video.source_fps,
        frame_width=video.frame_width,
        frame_height=video.frame_height,
    )


class ShardWriter:
    def __init__(
        self,
        out_dir: Path,
        samples_per_shard: int = 128,
        *,
        initial_rows: list[dict] | None = None,
        start_index: int = 0,
    ):
        self.out_dir = out_dir
        self.samples_per_shard = int(samples_per_shard)
        self.shards_dir = out_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self._pending: list[tuple[str, ExtractedSegment]] = []
        self._shard_index = int(start_index)
        self._rows: list[dict] = list(initial_rows or [])

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def add(self, sample_key: str, segment: ExtractedSegment, row: dict) -> None:
        self._pending.append((sample_key, segment))
        self._rows.append(row)
        if len(self._pending) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        shard_name = f"shard_{self._shard_index:05d}.npz"
        shard_path = self.shards_dir / shard_name
        payload = {}
        for sample_key, segment in self._pending:
            for array_key in ARRAY_KEYS:
                payload[f"{sample_key}__{array_key}"] = getattr(segment, array_key)
        np.savez_compressed(shard_path, **payload)
        for row in self._rows[-len(self._pending):]:
            row["shard_file"] = f"shards/{shard_name}"
        self._pending.clear()
        self._shard_index += 1


def _write_status(out_dir: Path, status: dict) -> None:
    status = dict(status)
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(out_dir / "status.json")


def _write_metadata(out_dir: Path, rows: list[dict]) -> pd.DataFrame:
    meta = pd.DataFrame(rows)
    if meta.empty:
        return meta
    meta = meta.drop_duplicates(subset=["split", "sentence_name"], keep="last").reset_index(drop=True)
    tmp = out_dir / "metadata.tmp.parquet"
    meta.to_parquet(tmp, index=False)
    target = out_dir / "metadata.parquet"
    for attempt in range(5):
        try:
            tmp.replace(target)
            break
        except PermissionError:
            if attempt == 4:
                # Windows can briefly keep parquet handles open; fall back to a
                # direct write rather than losing a completed run's manifest.
                meta.to_parquet(target, index=False)
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                time.sleep(0.5 * (attempt + 1))
    return meta


def _sample_key(row: dict) -> str:
    return f"{row['split']}_{row['SENTENCE_NAME']}".replace("/", "_")


def _metadata_row(row: dict, segment: ExtractedSegment) -> dict:
    return {
        "sentence_id": row["SENTENCE_ID"],
        "sentence_name": row["SENTENCE_NAME"],
        "video_id": row["VIDEO_ID"],
        "video_name": row["VIDEO_NAME"],
        "start": float(row["START_REALIGNED"]),
        "end": float(row["END_REALIGNED"]),
        "duration": float(row["END_REALIGNED"]) - float(row["START_REALIGNED"]),
        "sentence": row["SENTENCE"],
        "split": row["split"],
        "sample_key": _sample_key(row),
        "n_frames": int(segment.landmarks_image.shape[0]),
        "source_fps": float(segment.source_fps),
        "frame_width": int(segment.frame_width),
        "frame_height": int(segment.frame_height),
        "schema_version": SCHEMA_VERSION,
    }


def _next_shard_index(shards_dir: Path) -> int:
    max_index = -1
    for shard in shards_dir.glob("shard_*.npz"):
        stem = shard.stem
        try:
            max_index = max(max_index, int(stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1


def _video_properties(video_path: str | Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return math.nan, 0, 0
    try:
        return (
            float(cap.get(cv2.CAP_PROP_FPS) or math.nan),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def _reconstruct_completed_rows(out_dir: Path, jobs: pd.DataFrame) -> tuple[list[dict], set[str], int]:
    shards_dir = out_dir / "shards"
    next_index = _next_shard_index(shards_dir)
    meta_path = out_dir / "metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        existing_rows = meta.to_dict("records")
        completed = set(meta["sample_key"].astype(str))
        return existing_rows, completed, next_index

    if not shards_dir.exists():
        return [], set(), next_index

    jobs_by_key: dict[str, dict] = {}
    for row in jobs.to_dict("records"):
        jobs_by_key[_sample_key(row)] = row

    props_cache: dict[str, tuple[float, int, int]] = {}
    existing_rows: list[dict] = []
    completed: set[str] = set()
    for shard in sorted(shards_dir.glob("shard_*.npz")):
        shard_rel = f"shards/{shard.name}"
        try:
            with np.load(shard) as data:
                sample_keys = sorted(
                    key[: -len("__landmarks_image")]
                    for key in data.files
                    if key.endswith("__landmarks_image")
                )
                for sample_key in sample_keys:
                    if sample_key in completed:
                        continue
                    row = jobs_by_key.get(sample_key)
                    if row is None:
                        continue
                    timestamps = data.get(f"{sample_key}__timestamps_ms")
                    n_frames = int(timestamps.shape[0]) if timestamps is not None else 0
                    video_path = row["video_path"]
                    if video_path not in props_cache:
                        props_cache[video_path] = _video_properties(video_path)
                    source_fps, frame_width, frame_height = props_cache[video_path]
                    existing_rows.append({
                        "sentence_id": row["SENTENCE_ID"],
                        "sentence_name": row["SENTENCE_NAME"],
                        "video_id": row["VIDEO_ID"],
                        "video_name": row["VIDEO_NAME"],
                        "start": float(row["START_REALIGNED"]),
                        "end": float(row["END_REALIGNED"]),
                        "duration": float(row["END_REALIGNED"]) - float(row["START_REALIGNED"]),
                        "sentence": row["SENTENCE"],
                        "split": row["split"],
                        "sample_key": sample_key,
                        "n_frames": n_frames,
                        "source_fps": float(source_fps),
                        "frame_width": int(frame_width),
                        "frame_height": int(frame_height),
                        "schema_version": SCHEMA_VERSION,
                        "shard_file": shard_rel,
                    })
                    completed.add(sample_key)
        except Exception as exc:
            print(f"[warning] could not read existing shard {shard}: {exc}")

    if existing_rows:
        _write_metadata(out_dir, existing_rows)
    return existing_rows, completed, next_index


def _process_video_job(job: dict) -> dict:
    rows = job["rows"]
    video_path = Path(job["video_path"])
    start_s = min(float(r["START_REALIGNED"]) for r in rows)
    end_s = max(float(r["END_REALIGNED"]) for r in rows)
    segments = []
    failures = []
    try:
        with SegmentLandmarkExtractor(
            target_fps=float(job["target_fps"]),
            pose_model=str(job["pose_model"]),
            include_face=bool(job["include_face"]),
        ) as extractor:
            video = extractor.extract_video_range(video_path, start_s, end_s)
            for row in rows:
                try:
                    segment = slice_video_segment(
                        video,
                        float(row["START_REALIGNED"]),
                        float(row["END_REALIGNED"]),
                    )
                    if segment.landmarks_image.shape[0] == 0:
                        raise RuntimeError("segment has no sampled frames")
                    segments.append((_sample_key(row), segment, _metadata_row(row, segment)))
                except Exception as exc:
                    failures.append({
                        "split": row["split"],
                        "sentence_id": row["SENTENCE_ID"],
                        "sentence_name": row["SENTENCE_NAME"],
                        "video_name": row["VIDEO_NAME"],
                        "error": repr(exc),
                    })
    except Exception as exc:
        failures.extend({
            "split": row["split"],
            "sentence_id": row["SENTENCE_ID"],
            "sentence_name": row["SENTENCE_NAME"],
            "video_name": row["VIDEO_NAME"],
            "error": repr(exc),
        } for row in rows)
    return {
        "video_path": str(video_path),
        "segments": segments,
        "failures": failures,
    }


def load_split_csv(csv_path: Path, split: str, videos_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    required = {
        "VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
        "START_REALIGNED", "END_REALIGNED", "SENTENCE",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    available = {p.stem for p in videos_dir.glob("*.mp4")}
    df = df[df["VIDEO_NAME"].isin(available)].copy()
    df["split"] = split
    df["video_path"] = df["VIDEO_NAME"].map(lambda name: str(videos_dir / f"{name}.mp4"))
    return df.reset_index(drop=True)


def write_dataset_card(out_dir: Path, meta: pd.DataFrame, config: dict) -> None:
    failures_path = out_dir / "failures.parquet"
    if failures_path.exists():
        failures = pd.read_parquet(failures_path)
        failure_summary = (
            f"- Failed rows: {len(failures)} "
            f"({failures['split'].value_counts().to_dict()})\n"
            "- Failure details are stored in `failures.parquet`.\n"
        )
    else:
        failure_summary = "- Failed rows: 0\n"
    card = f"""---
license: cc-by-nc-4.0
task_categories:
  - translation
  - feature-extraction
language:
  - en
tags:
  - sign-language
  - asl
  - how2sign
  - mediapipe
---

# How2Sign ASL Landmarks

Sentence-level MediaPipe landmark cache built from local How2Sign front-view
videos and realigned sentence timestamps. This dataset stores arrays only, not
video clips.

## Contents

- Splits: {meta["split"].value_counts().to_dict()}
- Expected rows covered by `metadata.parquet` + `failures.parquet`: see `audit_report.json`
{failure_summary.rstrip()}
- Landmark schema: `{SCHEMA_VERSION}`
- Target FPS: {config["target_fps"]}
- Landmarks per frame: {TOTAL_LMS}
- Blocks: {", ".join(f"{b.name}[{b.start}:{b.end}]" for b in LANDMARK_BLOCKS)}
- Face subset indices: `{list(ASL_FACE_LANDMARKS)}`
- Geometric feature count: {len(GEOMETRIC_FEATURE_NAMES)}

Each metadata row points to one NPZ shard and `sample_key`. Arrays are named
`<sample_key>__landmarks_image`, `<sample_key>__landmarks_world`,
`<sample_key>__features_geometric`, `<sample_key>__valid_mask`,
`<sample_key>__timestamps_ms`, and `<sample_key>__handedness_scores`.

`features_geometric` is a compact recipe block computed from the landmark arrays
for training convenience. `feature_names.json` stores the column names.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def write_config(out_dir: Path, config: dict) -> None:
    (out_dir / "preprocessing_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_landmark_dataset(
    split_specs: Iterable[tuple[str, Path, Path]],
    out_dir: Path,
    *,
    target_fps: float = 25.0,
    samples_per_shard: int = 128,
    limit_per_split: int | None = None,
    overwrite: bool = False,
    workers: int = 1,
    pose_model: str = "lite",
    include_face: bool = True,
) -> pd.DataFrame:
    if out_dir.exists() and overwrite:
        for child in out_dir.glob("*"):
            if child.is_file():
                child.unlink()
            elif child.is_dir() and child.name == "shards":
                for shard in child.glob("*.npz"):
                    shard.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split_order, (split, csv_path, videos_dir) in enumerate(split_specs):
        split_df = load_split_csv(csv_path, split, videos_dir)
        if limit_per_split is not None:
            split_df = split_df.head(limit_per_split).copy()
        split_df["_split_order"] = split_order
        frames.append(split_df)
    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs.sort_values(
        ["_split_order", "VIDEO_NAME", "START_REALIGNED", "END_REALIGNED"],
        kind="stable",
    ).reset_index(drop=True)
    if jobs.empty:
        raise RuntimeError("no rows matched the provided CSV/video folders")

    config = {
        "schema_version": SCHEMA_VERSION,
        "target_fps": float(target_fps),
        "samples_per_shard": int(samples_per_shard),
        "workers": int(workers),
        "processing_unit": "source_video",
        "pose_model": pose_model,
        "include_face": bool(include_face),
        "landmark_blocks": [b.__dict__ for b in LANDMARK_BLOCKS],
        "asl_face_landmarks": list(ASL_FACE_LANDMARKS),
        "array_keys": list(ARRAY_KEYS),
        "geometric_feature_names": list(GEOMETRIC_FEATURE_NAMES),
    }

    existing_rows, completed_sample_keys, next_shard_index = _reconstruct_completed_rows(out_dir, jobs)
    completed_names = {
        (str(row["split"]), str(row["sentence_name"]))
        for row in existing_rows
        if "split" in row and "sentence_name" in row
    }
    if completed_names:
        print(f"Resuming: found {len(completed_names)} completed clips in existing shards/metadata.")
    writer = ShardWriter(
        out_dir,
        samples_per_shard=samples_per_shard,
        initial_rows=existing_rows,
        start_index=next_shard_index,
    )
    failures: list[dict] = []
    failures_path = out_dir / "failures.parquet"
    if failures_path.exists():
        failures_path.unlink()

    video_jobs = []
    for video_path, rows in jobs.groupby("video_path", sort=False):
        records = rows.drop(columns=[c for c in ["_split_order"] if c in rows.columns]).to_dict("records")
        records = [r for r in records if (str(r["split"]), str(r["SENTENCE_NAME"])) not in completed_names]
        if not records:
            continue
        video_jobs.append({
            "video_path": video_path,
            "rows": records,
            "target_fps": float(target_fps),
            "pose_model": pose_model,
            "include_face": bool(include_face),
        })

    _write_status(out_dir, {
        "state": "running",
        "processed_videos": 0,
        "total_videos": len(video_jobs),
        "processed_segments": len(writer.rows),
        "resumed_segments": len(completed_names),
        "failed_segments": 0,
        "shards_written": writer._shard_index,
    })

    workers = max(1, int(workers))
    processed_videos = 0

    def record_result(result: dict) -> None:
        nonlocal processed_videos
        failures.extend(result["failures"])
        for sample_key, segment, row in result["segments"]:
            writer.add(sample_key, segment, row)
        writer.flush()
        _write_metadata(out_dir, writer.rows)
        processed_videos += 1
        _write_status(out_dir, {
            "state": "running",
            "processed_videos": processed_videos,
            "total_videos": len(video_jobs),
            "processed_segments": len(writer.rows),
            "resumed_segments": len(completed_names),
            "failed_segments": len(failures),
            "shards_written": writer._shard_index,
            "last_video": result["video_path"],
        })

    try:
        if workers == 1:
            iterator = (_process_video_job(job) for job in video_jobs)
            for result in tqdm(iterator, total=len(video_jobs), desc="Extracting video landmarks"):
                record_result(result)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_process_video_job, job) for job in video_jobs]
                try:
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting video landmarks"):
                        record_result(fut.result())
                except KeyboardInterrupt:
                    for fut in futures:
                        fut.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
    except KeyboardInterrupt:
        writer.flush()
        meta = _write_metadata(out_dir, writer.rows)
        if failures:
            pd.DataFrame(failures).to_parquet(failures_path, index=False)
        elif failures_path.exists():
            failures_path.unlink()
        write_config(out_dir, config)
        (out_dir / "feature_names.json").write_text(
            json.dumps({"features_geometric": list(GEOMETRIC_FEATURE_NAMES)}, indent=2),
            encoding="utf-8",
        )
        _write_status(out_dir, {
            "state": "interrupted",
            "processed_videos": processed_videos,
            "total_videos": len(video_jobs),
            "processed_segments": len(meta),
            "resumed_segments": len(completed_names),
            "failed_segments": len(failures),
            "shards_written": writer._shard_index,
        })
        print("\nInterrupted. Flushed shards and metadata; rerun the same command without --overwrite to resume.")
        raise SystemExit(130)

    writer.flush()
    meta = _write_metadata(out_dir, writer.rows)
    if meta.empty:
        raise RuntimeError("no segments were successfully extracted")
    if failures:
        pd.DataFrame(failures).to_parquet(failures_path, index=False)
    elif failures_path.exists():
        failures_path.unlink()
    write_config(out_dir, config)
    (out_dir / "feature_names.json").write_text(
        json.dumps({"features_geometric": list(GEOMETRIC_FEATURE_NAMES)}, indent=2),
        encoding="utf-8",
    )
    write_dataset_card(out_dir, meta, config)
    _write_status(out_dir, {
        "state": "done",
        "processed_videos": len(video_jobs),
        "total_videos": len(video_jobs),
        "processed_segments": len(meta),
        "failed_segments": len(failures),
        "shards_written": writer._shard_index,
    })
    return meta


def rebuild_landmark_metadata(
    split_specs: Iterable[tuple[str, Path, Path]],
    out_dir: Path,
    *,
    limit_per_split: int | None = None,
) -> pd.DataFrame:
    frames = []
    for split_order, (split, csv_path, videos_dir) in enumerate(split_specs):
        split_df = load_split_csv(csv_path, split, videos_dir)
        if limit_per_split is not None:
            split_df = split_df.head(limit_per_split).copy()
        split_df["_split_order"] = split_order
        frames.append(split_df)
    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs.sort_values(
        ["_split_order", "VIDEO_NAME", "START_REALIGNED", "END_REALIGNED"],
        kind="stable",
    ).reset_index(drop=True)
    rows, completed, next_shard_index = _reconstruct_completed_rows(out_dir, jobs)
    meta = _write_metadata(out_dir, rows)
    _write_status(out_dir, {
        "state": "metadata_rebuilt",
        "processed_videos": 0,
        "total_videos": int(jobs["video_path"].nunique()),
        "processed_segments": len(meta),
        "resumed_segments": len(completed),
        "failed_segments": 0,
        "shards_written": next_shard_index,
    })
    return meta
