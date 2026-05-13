"""Text feature encoders for the landmark retrieval pipeline.

Two concrete implementations are provided:
  - SentenceTransformerTextEncoder  (default; uses sentence-transformers)
  - TFIDFTextEncoder                (lightweight sklearn fallback)

Both expose the same interface used by train.py:
  encoder.fit(sentences)            → prepare the encoder (no-op for ST)
  encoder.output_dim                → embedding dimensionality (int)
  encoder.transform_tensor(s, dev)  → Tensor[N, D] on *device*
  encoder.save(out_dir)             → dict to merge into the checkpoint
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from src.training.devices import resolve_device


# ──────────────────────── Base class ─────────────────────────────────────────

class TextFeatureEncoder(ABC):
    """Common interface for frozen text-feature extractors."""

    @abstractmethod
    def fit(self, sentences: list[str]) -> None:
        """Fit the encoder on the training corpus (may be a no-op)."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the returned feature vectors."""

    @abstractmethod
    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        """Encode *sentences* and return a float32 Tensor of shape (N, D)."""

    @abstractmethod
    def save(self, out_dir: str | Path) -> dict:
        """Persist any learnt artefacts and return a dict for the checkpoint."""


# ──────────────────────── Sentence-transformer encoder ───────────────────────

class SentenceTransformerTextEncoder(TextFeatureEncoder):
    """Frozen sentence-transformer encoder (sentence-transformers library).

    The model is downloaded/cached on first use and kept in memory.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._enc_device = resolve_device(device)
        self.local_files_only = local_files_only
        self._model = None

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerTextEncoder requires the 'sentence-transformers' "
                "package.  Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(
            self.model_name,
            device=self._enc_device,
            local_files_only=self.local_files_only,
        )

    def fit(self, sentences: list[str]) -> None:
        pass  # sentence transformers are pre-trained; no fitting needed

    @property
    def output_dim(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        if self._model is None:
            self._load_model()
        emb = self._model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self._enc_device,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return emb.to(device)

    def save(self, out_dir: str | Path) -> dict:
        return {
            "text_encoder_type": "sentence_transformer",
            "text_model_name": self.model_name,
            "text_embedding_dim": self.output_dim,
        }


# ──────────────────────── TF-IDF encoder ─────────────────────────────────────

class TFIDFTextEncoder(TextFeatureEncoder):
    """TF-IDF bag-of-n-grams encoder (sklearn, no GPU required)."""

    def __init__(
        self,
        max_features: int = 20_000,
        max_ngram: int = 2,
        min_df: int = 1,
    ) -> None:
        self._max_features = max_features
        self._max_ngram = max_ngram
        self._min_df = min_df
        self._vectorizer = None

    def fit(self, sentences: list[str]) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "TFIDFTextEncoder requires scikit-learn.  "
                "Install with: pip install scikit-learn"
            ) from exc
        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=(1, self._max_ngram),
            min_df=self._min_df,
        )
        self._vectorizer.fit(sentences)

    @property
    def output_dim(self) -> int:
        if self._vectorizer is None:
            raise RuntimeError("TFIDFTextEncoder must be fit() before output_dim is known.")
        return len(self._vectorizer.vocabulary_)

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        if self._vectorizer is None:
            raise RuntimeError("TFIDFTextEncoder must be fit() before transform_tensor().")
        sparse = self._vectorizer.transform(sentences)
        dense = np.asarray(sparse.todense(), dtype=np.float32)
        return torch.from_numpy(dense).to(device)

    def save(self, out_dir: str | Path) -> dict:
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Saving TF-IDF requires joblib: pip install joblib") from exc
        if self._vectorizer is None:
            raise RuntimeError("Cannot save an unfit TFIDFTextEncoder.")
        path = Path(out_dir) / "tfidf_vectorizer.joblib"
        joblib.dump(self._vectorizer, path)
        return {
            "text_encoder_type": "tfidf",
            "text_vectorizer_path": str(path),
            "text_embedding_dim": self.output_dim,
        }


# ──────────────────────── Factory functions ───────────────────────────────────

def build_text_encoder_from_config(config: dict, _cfg) -> TextFeatureEncoder:
    """Construct a text encoder from a training YAML config."""
    encoder_type = str(_cfg(config, "text.encoder", "sentence_transformer"))
    if encoder_type == "sentence_transformer":
        return SentenceTransformerTextEncoder(
            model_name=str(_cfg(config, "text.sentence_model", "sentence-transformers/all-MiniLM-L6-v2")),
            batch_size=int(_cfg(config, "text.embedding_batch_size", 64)),
            device=str(_cfg(config, "text.embedding_device", "auto")),
            local_files_only=bool(_cfg(config, "text.local_files_only", False)),
        )
    # Fallback: TF-IDF
    return TFIDFTextEncoder(
        max_features=int(_cfg(config, "text.tfidf_max_features", 20_000)),
        max_ngram=int(_cfg(config, "text.tfidf_max_ngram", 2)),
        min_df=int(_cfg(config, "text.tfidf_min_df", 1)),
    )


def build_text_encoder_from_checkpoint(
    ckpt: dict,
    ckpt_path: str | Path | None = None,
) -> TextFeatureEncoder:
    """Reconstruct a text encoder from a saved checkpoint dict.

    *ckpt_path* is only needed for the TF-IDF path (to locate the vectorizer
    file when its absolute path is no longer valid).
    """
    encoder_type = ckpt.get("text_encoder_type", "sentence_transformer")
    if encoder_type == "sentence_transformer":
        return SentenceTransformerTextEncoder(
            model_name=str(ckpt.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        )

    # TF-IDF path
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Restoring TF-IDF encoder requires joblib: pip install joblib") from exc

    enc = TFIDFTextEncoder()
    vpath = ckpt.get("text_vectorizer_path")
    if vpath and Path(vpath).exists():
        enc._vectorizer = joblib.load(vpath)
    elif ckpt_path is not None:
        # Try next to the checkpoint file.
        candidate = Path(ckpt_path).parent / "tfidf_vectorizer.joblib"
        if candidate.exists():
            enc._vectorizer = joblib.load(candidate)
    if enc._vectorizer is None:
        raise FileNotFoundError(
            "Could not find the TF-IDF vectorizer file.  "
            "Re-run training to regenerate it."
        )
    return enc
