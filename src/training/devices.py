"""Device selection helpers for local experiments and cluster runs."""
from __future__ import annotations

import torch


def resolve_device(requested: str | None = "auto") -> str:
    """Resolve ``auto``/``cuda``/``mps``/``cpu`` to an available torch device."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested device=cuda, but CUDA is unavailable; falling back to CPU.")
        return "cpu"

    if requested == "mps":
        has_mps = getattr(torch.backends, "mps", None) is not None
        if not has_mps or not torch.backends.mps.is_available():
            print("Requested device=mps, but MPS is unavailable; falling back to CPU.")
            return "cpu"

    return requested
