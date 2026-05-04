"""PyTorch datasets wrapping extracted landmark sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.landmark_transforms import landmarks_to_features


def _read_metadata(root: Path) -> pd.DataFrame:
    parquet = root / "metadata.parquet"
    csv = root / "metadata.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"expected {parquet} or {csv}")


class LandmarkSequenceDataset(Dataset):
    """Dataset of cached landmark clips.

    The cache root must contain ``metadata.parquet`` or ``metadata.csv`` with a
    ``landmark_path`` column pointing to per-clip ``.npz`` files. Each file must
    contain ``landmarks`` and should contain ``mask`` and ``sentence``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        window_size: int = 128,
        *,
        layout: str = "full",
        rows: pd.DataFrame | None = None,
    ):
        self.root = Path(root)
        self.window_size = int(window_size)
        self.layout = layout
        meta = _read_metadata(self.root) if rows is None else rows.copy()
        if split is not None and "split" in meta.columns:
            meta = meta[meta["split"] == split].reset_index(drop=True)
        if "landmark_path" not in meta.columns:
            raise ValueError("metadata must contain a 'landmark_path' column")
        if "sentence" not in meta.columns:
            raise ValueError("metadata must contain a 'sentence' column")
        if len(meta) == 0:
            raise ValueError(f"no rows found for split={split!r} in {self.root}")
        self.metadata = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def _npz_path(self, row: pd.Series) -> Path:
        path = Path(row["landmark_path"])
        if path.is_absolute():
            return path
        return self.root / path

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        path = self._npz_path(row)
        with np.load(path, allow_pickle=True) as blob:
            landmarks = blob["landmarks"].astype(np.float32)
            mask = blob["mask"].astype(np.float32) if "mask" in blob else None
        features = landmarks_to_features(
            landmarks,
            mask,
            target_frames=self.window_size,
            layout=self.layout,
        )
        sentence = str(row["sentence"])
        sample_id = str(row.get("sentence_id", row.get("sentence_name", idx)))
        return {
            "features": torch.from_numpy(features),
            "sentence": sentence,
            "id": sample_id,
            "index": idx,
        }


class How2SignLandmarkRetrievalDataset(LandmarkSequenceDataset):
    """Named alias for the sentence-retrieval baseline."""
