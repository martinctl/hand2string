"""Download / prepare datasets into ``root``.

Supported names: ``asl_alphabet``, ``wlasl``, ``how2sign``, ``how2sign_landmarks``.
"""
from __future__ import annotations

from pathlib import Path

from src.dataset.hub import DEFAULT_LANDMARK_REPO_ID, DEFAULT_REPO_ID, load_clips, load_landmarks


def download(name: str, root: str) -> Path:
    root_path = Path(root)
    if name == "how2sign":
        local_dir = root_path / "how2sign_clips"
        local, df = load_clips(repo_id=DEFAULT_REPO_ID, split=None, local_dir=local_dir)
        print(f"how2sign clips ready at: {local}")
        print(f"  rows: {len(df)}  splits: {sorted(df['split'].unique())}")
        return local

    if name == "how2sign_landmarks":
        local_dir = root_path / "how2sign_landmarks"
        local, df = load_landmarks(repo_id=DEFAULT_LANDMARK_REPO_ID, split=None, local_dir=local_dir)
        print(f"how2sign landmarks ready at: {local}")
        print(f"  rows: {len(df)}  splits: {sorted(df['split'].unique())}")
        print("  training config root: data/how2sign_landmarks")
        return local

    if name in {"asl_alphabet", "wlasl"}:
        raise NotImplementedError(f"{name!r} download not implemented yet")

    raise ValueError(f"unknown dataset name: {name!r}")
