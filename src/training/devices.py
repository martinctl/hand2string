"""Device selection helpers for training and evaluation."""
from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Return a torch device string from a config value."""
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested
