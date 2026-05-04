"""Download / prepare raw datasets into ``root``.

Supported names: ``asl_alphabet``, ``wlasl``, ``how2sign``.
"""
from __future__ import annotations

from pathlib import Path

from src.dataset.hub import DEFAULT_REPO_ID, load_clips


def _refresh_alias(alias: Path, target: Path) -> None:
    """Create a convenient local pointer to the HF snapshot when possible."""
    try:
        if alias.exists() or alias.is_symlink():
            return
        alias.symlink_to(target.resolve(), target_is_directory=True)
        print(f"  alias: {alias} -> {target}")
    except OSError:
        # Symlinks can be unavailable on some filesystems; the downloader still
        # returns and prints the real snapshot path, and extract_landmarks can
        # now resolve the HF cache layout directly.
        pass


def download(name: str, root: str) -> Path:
    if name == "how2sign":
        local, df = load_clips(repo_id=DEFAULT_REPO_ID, split=None, cache_dir=root)
        _refresh_alias(Path(root) / "how2sign_hf", local)
        print(f"how2sign clips ready at: {local}")
        print(f"  rows: {len(df)}  splits: {sorted(df['split'].unique())}")
        return local

    if name in {"asl_alphabet", "wlasl"}:
        raise NotImplementedError(f"{name!r} download not implemented yet")

    raise ValueError(f"unknown dataset name: {name!r}")
