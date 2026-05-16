"""PyTorch datasets wrapping extracted landmark sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class How2SignPreprocessedLandmarkDataset(Dataset):
    """Read the NPZ-sharded landmarks-only How2Sign dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        *,
        rows: pd.DataFrame | None = None,
        use_world: bool = False,
        feature_source: str = "landmarks",
        flatten: bool = False,
        fill_value: float = 0.0,
    ):
        self.root = Path(root)
        if rows is None:
            meta_path = self.root / "metadata.parquet"
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            rows = pd.read_parquet(meta_path)
        if split is not None:
            rows = rows[rows["split"] == split]
        self.rows = rows.reset_index(drop=True)
        self.array_key = "landmarks_world" if use_world else "landmarks_image"
        if feature_source not in {"landmarks", "geometric", "landmarks+geometric"}:
            raise ValueError("feature_source must be landmarks, geometric, or landmarks+geometric")
        self.feature_source = feature_source
        self.flatten = bool(flatten)
        self.fill_value = float(fill_value)
        self._shard_cache: dict[str, np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_shard(self, shard_file: str) -> np.lib.npyio.NpzFile:
        shard = self._shard_cache.get(shard_file)
        if shard is None:
            shard = np.load(self.root / shard_file)
            self._shard_cache[shard_file] = shard
        return shard

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        shard = self._load_shard(row["shard_file"])
        prefix = row["sample_key"]
        landmarks = shard[f"{prefix}__{self.array_key}"].astype(np.float32)
        mask = shard[f"{prefix}__valid_mask"].astype(np.bool_)
        timestamps = shard[f"{prefix}__timestamps_ms"].astype(np.float32)
        hand_scores = shard[f"{prefix}__handedness_scores"].astype(np.float32)
        landmark_features = np.nan_to_num(landmarks, nan=self.fill_value)
        if self.flatten:
            landmark_features = landmark_features.reshape(landmark_features.shape[0], -1)
        if self.feature_source == "geometric":
            features = shard[f"{prefix}__features_geometric"].astype(np.float32)
        elif self.feature_source == "landmarks+geometric":
            geometric = shard[f"{prefix}__features_geometric"].astype(np.float32)
            flat_landmarks = landmark_features.reshape(landmark_features.shape[0], -1)
            features = np.concatenate([flat_landmarks, geometric], axis=-1)
        else:
            features = landmark_features
        return {
            "features": torch.from_numpy(features).float(),
            "landmarks": torch.from_numpy(landmark_features).float(),
            "features_geometric": torch.from_numpy(shard[f"{prefix}__features_geometric"].astype(np.float32)).float(),
            "valid_mask": torch.from_numpy(mask),
            "timestamps_ms": torch.from_numpy(timestamps),
            "handedness_scores": torch.from_numpy(hand_scores).float(),
            "sentence": str(row["sentence"]),
            "id": str(row["sentence_id"]),
            "index": idx,
            "row": row.to_dict(),
        }


def pad_landmark_batch(batch: list[dict]) -> dict:
    """Pad variable-length landmark sequences to the longest item in a batch."""
    max_len = max(item["features"].shape[0] for item in batch)
    feature_shape = batch[0]["features"].shape[1:]
    mask_shape = batch[0]["valid_mask"].shape[1:]
    features = torch.zeros((len(batch), max_len, *feature_shape), dtype=torch.float32)
    valid_mask = torch.zeros((len(batch), max_len, *mask_shape), dtype=torch.bool)
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        valid_mask[i, :n] = item["valid_mask"]
        padding_mask[i, :n] = False
    return {
        "features": features,
        "valid_mask": valid_mask,
        "padding_mask": padding_mask,
        "sentence": [item["sentence"] for item in batch],
        "id": [item["id"] for item in batch],
        "index": [item["index"] for item in batch],
    }


class How2SignLandmarkRetrievalDataset(How2SignPreprocessedLandmarkDataset):
    """Compatibility wrapper used by the retrieval training scripts."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        window_size: int | None = None,
        layout: str = "image",
        rows: pd.DataFrame | None = None,
        feature_source: str | None = None,
    ):
        if feature_source is None:
            feature_source = "geometric" if layout == "geometric" else "landmarks"
        super().__init__(
            root,
            split=split,
            rows=rows,
            use_world=(layout == "world"),
            feature_source=feature_source,
            flatten=True,
        )
        self.window_size = window_size

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self.window_size is None:
            return item
        x = item["features"]
        if x.shape[0] >= self.window_size:
            item["features"] = x[: self.window_size]
        else:
            padded = torch.zeros((self.window_size, x.shape[1]), dtype=x.dtype)
            padded[: x.shape[0]] = x
            item["features"] = padded
        return item


class LandmarkSequenceDataset(How2SignPreprocessedLandmarkDataset):
    def __init__(self, root: str, split: str, window_size: int):
        super().__init__(root, split=split, flatten=True)
        self.window_size = window_size
