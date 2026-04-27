"""Download / prepare raw datasets into ``root``.

Supported names: ``asl_alphabet``, ``wlasl``, ``how2sign``.
"""
from __future__ import annotations

from pathlib import Path

from src.dataset.hub import DEFAULT_REPO_ID, load_clips


def download(name: str, root: str) -> Path:
    if name == "how2sign":
        local, df = load_clips(repo_id=DEFAULT_REPO_ID, split=None, cache_dir=root)
        print(f"how2sign clips ready at: {local}")
        print(f"  rows: {len(df)}  splits: {sorted(df['split'].unique())}")
        return local

    if name in {"asl_alphabet", "wlasl"}:
        raise NotImplementedError(f"{name!r} download not implemented yet")

    raise ValueError(f"unknown dataset name: {name!r}")
