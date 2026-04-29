"""Pull the How2Sign clips dataset from the Hugging Face Hub."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "martinctl/how2sign-asl-clips"


def load_clips(
    repo_id: str = DEFAULT_REPO_ID,
    split: str | None = "train",
    token: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Download (or fetch from cache) the dataset and return (root, metadata).

    The returned ``metadata`` dataframe has an absolute ``clip_path`` column
    so callers can open clips directly without re-resolving paths.
    """
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    )

    df = pd.read_parquet(local / "metadata.parquet")
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    df["clip_path"] = df["file_name"].apply(lambda p: str(local / p))
    return local, df
