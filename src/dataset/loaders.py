# PyTorch Datasets wrapping extracted landmark sequences.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandmarkSequenceDataset:
    def __init__(self, root: str, split: str, window_size: int):
        raise NotImplementedError


class How2SignLandmarkRetrievalDataset(Dataset):
    """Dataset of MediaPipe landmark sequences paired with English sentences.

    On-disk layout (produced by the landmark extraction pipeline):
        <root>/
            metadata.parquet
            landmarks/<split>/<sentence_name>.npz   # (T, 75, 3) float32

    Each .npz file stores a single key ``"landmarks"`` with shape ``(T, 75, 3)``:
      - [:33]   → 33 BlazePose body landmarks
      - [33:54] → 21 left-hand landmarks
      - [54:75] → 21 right-hand landmarks
    Missing detections are stored as NaN and replaced with 0.0 at load time.

    Args:
        root:        Path to the how2sign_landmarks directory.
        split:       Split name ("train" / "val" / "test").  Ignored when
                     *rows* is supplied (the caller already filtered).
        window_size: Number of frames per returned sample.  Sequences are
                     center-cropped when longer and zero-padded when shorter.
        layout:      Which landmarks to use.
                       "full"  – all 75 → 225 features / frame
                       "hands" – hands only (42 lms) → 126 features / frame
                       "pose"  – body pose only (33 lms) → 99 features / frame
        rows:        Pre-filtered metadata DataFrame.  When given, *split* is
                     ignored.
    """

    # Slice into the (T, 75, 3) array along the landmark axis.
    _LAYOUT_SLICES: dict[str, slice] = {
        "full": slice(0, 75),
        "hands": slice(33, 75),
        "pose": slice(0, 33),
    }

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        window_size: int = 128,
        layout: str = "full",
        rows: pd.DataFrame | None = None,
    ) -> None:
        self.root = Path(root)
        self.window_size = window_size

        if layout not in self._LAYOUT_SLICES:
            raise ValueError(
                f"layout must be one of {list(self._LAYOUT_SLICES)}, got {layout!r}"
            )
        self.layout = layout

        if rows is not None:
            self.rows = rows.reset_index(drop=True)
        else:
            meta = pd.read_parquet(self.root / "metadata.parquet")
            if split is not None:
                meta = meta[meta["split"] == split]
            self.rows = meta.reset_index(drop=True)

    # ── Path resolution ───────────────────────────────────────────────────────

    def _npz_path(self, row: pd.Series) -> Path:
        """Derive the .npz landmark path from a metadata row.

        Mirrors the ``file_name`` column but under ``landmarks/`` and with a
        ``.npz`` extension, e.g.:
            file_name  = "clips/train/foo.mp4"
            npz_path   = <root>/landmarks/train/foo.npz
        """
        parts = Path(row.file_name).parts  # ('clips', 'train', 'foo.mp4')
        # Replace the leading "clips" component with "landmarks" and swap extension.
        rel = Path("landmarks", *parts[1:]).with_suffix(".npz")
        path = self.root / rel
        if not path.exists():
            # Fallback: flat layout using the primary key.
            path = self.root / "landmarks" / f"{row.sentence_id}.npz"
        return path

    # ── Feature extraction ────────────────────────────────────────────────────

    def _load_seq(self, path: Path) -> np.ndarray:
        """Load and return a ``(T, 75, 3)`` float32 array."""
        if not path.exists():
            raise FileNotFoundError(
                f"Landmark file not found: {path}\n"
                "Run scripts/extract_landmarks.py to build the landmark cache."
            )
        with np.load(path, allow_pickle=False) as npz:
            key = "landmarks" if "landmarks" in npz else next(iter(npz.keys()))
            return npz[key].astype(np.float32)

    def _apply_layout(self, seq: np.ndarray) -> np.ndarray:
        """Slice landmarks and flatten: ``(T, 75, 3)`` → ``(T, N*3)``."""
        sl = self._LAYOUT_SLICES[self.layout]
        T = len(seq)
        return seq[:, sl, :].reshape(T, -1)

    def _window(self, seq: np.ndarray) -> np.ndarray:
        """Center-crop or zero-pad to exactly ``window_size`` frames."""
        T, D = seq.shape
        W = self.window_size
        if T == 0:
            return np.zeros((W, D), dtype=np.float32)
        if T >= W:
            start = (T - W) // 2
            return seq[start : start + W]
        pad = np.zeros((W - T, D), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        path = self._npz_path(row)

        seq = self._load_seq(path)          # (T, 75, 3)
        seq = np.nan_to_num(seq, nan=0.0)  # missing detections → 0
        seq = self._apply_layout(seq)       # (T, N*3)
        seq = self._window(seq)             # (W, N*3)

        return {
            "features": torch.from_numpy(seq),  # (W, N*3)
            "sentence": str(row.sentence),
            "id": str(row.sentence_id),
            "index": idx,
        }
