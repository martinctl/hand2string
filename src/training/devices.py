"""Device resolution utilities."""
from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> str:
    """Return a concrete device string.

    "auto" picks "cuda" when a GPU is available, otherwise "cpu".
    Any other string is returned unchanged (e.g. "cuda:1", "cpu").
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
