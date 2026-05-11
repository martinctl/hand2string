"""Offline hard-negative mining for contrastive retrieval training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class HardNegativeMiner:
    """Find the hardest text negatives for each video in the training set.

    After refresh(), each sample index i maps to k_hard text indices whose
    embeddings are closest to video_i but belong to a different clip.  During
    training the caller uses get_hard_neg_sentences() to inject one hard
    negative per batch item, which is then concatenated to the standard
    in-batch negatives in the contrastive loss.

    Refresh is cheap (one forward pass over the dataset, no gradients) and
    should be called every few epochs after an initial warm-up.
    """

    def __init__(self, k_hard: int = 4):
        self.k_hard = int(k_hard)
        self._hard_neg_idx: np.ndarray | None = None  # (N, k_hard)

    @property
    def ready(self) -> bool:
        return self._hard_neg_idx is not None

    @torch.no_grad()
    def refresh(
        self,
        model: nn.Module,
        loader: DataLoader,
        encode_text_fn,
        device: str,
        all_sentences: list[str],
    ) -> None:
        """Recompute hard-negative indices from the current model embeddings.

        encode_text_fn: callable(List[str]) -> torch.Tensor (N, D) on CPU.
        """
        was_training = model.training
        model.eval()

        # Support DDP-wrapped models
        raw_model = model.module if hasattr(model, "module") else model

        video_embs: list[torch.Tensor] = []
        for batch in loader:
            x = batch["features"].to(device)
            video_embs.append(raw_model.encode_video(x).cpu())
        video_embs_t = torch.cat(video_embs, dim=0)  # (N, D)

        text_embs_t = encode_text_fn(all_sentences)  # (N, D), on CPU
        if not isinstance(text_embs_t, torch.Tensor):
            text_embs_t = torch.from_numpy(np.asarray(text_embs_t, dtype=np.float32))
        text_embs_t = text_embs_t.cpu()

        scores = video_embs_t @ text_embs_t.T  # (N, N)
        N = scores.shape[0]
        k = min(self.k_hard, N - 1)

        hard_neg_idx = np.empty((N, k), dtype=np.int64)
        for i in range(N):
            row = scores[i].clone()
            row[i] = float("-inf")  # exclude the ground-truth match
            hard_neg_idx[i] = row.topk(k).indices.numpy()
        self._hard_neg_idx = hard_neg_idx

        if was_training:
            model.train()

    def get_hard_neg_sentences(
        self,
        batch_indices,
        all_sentences: list[str],
    ) -> list[str]:
        """Return the single hardest negative sentence for each item in the batch."""
        if self._hard_neg_idx is None:
            raise RuntimeError("call refresh() before get_hard_neg_sentences()")
        return [all_sentences[int(self._hard_neg_idx[int(idx)][0])] for idx in batch_indices]
