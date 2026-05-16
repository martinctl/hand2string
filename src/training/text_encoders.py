"""Text feature encoders for landmark-to-caption retrieval training."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureEncoder(Protocol):
    output_dim: int

    def fit(self, sentences: list[str]) -> None: ...

    def transform(self, sentences: list[str]) -> np.ndarray: ...

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor: ...

    def save(self, out_dir: Path) -> dict: ...


class TfidfFeatureEncoder:
    """Small, dependency-light text baseline."""

    def __init__(self, max_features: int = 20_000, max_ngram: int = 2, min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, max_ngram),
            min_df=min_df,
            dtype=np.float32,
        )
        self.output_dim = 0

    def fit(self, sentences: list[str]) -> None:
        self.vectorizer.fit(sentences)
        self.output_dim = len(self.vectorizer.get_feature_names_out())

    def transform(self, sentences: list[str]) -> np.ndarray:
        return self.vectorizer.transform(sentences).astype(np.float32).toarray()

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        return torch.from_numpy(self.transform(sentences)).to(device)

    def save(self, out_dir: Path) -> dict:
        import joblib

        path = out_dir / "text_encoder_tfidf.joblib"
        joblib.dump(self.vectorizer, path)
        return {"text_encoder_type": "tfidf", "text_encoder_path": str(path)}


class SentenceTransformerFeatureEncoder:
    """Frozen sentence-transformer feature extractor."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use this text encoder.") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def fit(self, sentences: list[str]) -> None:
        # Frozen encoder: no fitting required.
        _ = sentences

    def transform(self, sentences: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                sentences,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        return torch.from_numpy(self.transform(sentences)).to(device)

    def save(self, out_dir: Path) -> dict:
        return {"text_encoder_type": "sentence_transformer", "text_model_name": self.model_name}


def build_text_encoder_from_config(config: dict, cfg_getter) -> TextFeatureEncoder:
    encoder = str(cfg_getter(config, "text.encoder", "tfidf"))
    if encoder == "sentence_transformer":
        return SentenceTransformerFeatureEncoder(
            model_name=str(cfg_getter(config, "text.sentence_model", "sentence-transformers/all-MiniLM-L6-v2")),
            batch_size=int(cfg_getter(config, "text.embedding_batch_size", 64)),
            device=str(cfg_getter(config, "text.embedding_device", "auto")),
            local_files_only=bool(cfg_getter(config, "text.local_files_only", False)),
        )
    if encoder == "tfidf":
        return TfidfFeatureEncoder(
            max_features=int(cfg_getter(config, "text.tfidf_max_features", 20_000)),
            max_ngram=int(cfg_getter(config, "text.tfidf_max_ngram", 2)),
            min_df=int(cfg_getter(config, "text.tfidf_min_df", 1)),
        )
    raise ValueError(f"unknown text encoder: {encoder!r}")


def build_text_encoder_from_checkpoint(ckpt: dict, ckpt_path: Path | None = None) -> TextFeatureEncoder:
    kind = ckpt.get("text_encoder_type", "sentence_transformer")
    if kind == "tfidf":
        import joblib

        enc = TfidfFeatureEncoder()
        path = Path(ckpt["text_encoder_path"])
        if not path.exists() and ckpt_path is not None:
            path = ckpt_path.parent / path.name
        enc.vectorizer = joblib.load(path)
        enc.output_dim = len(enc.vectorizer.get_feature_names_out())
        return enc
    return SentenceTransformerFeatureEncoder(
        model_name=str(ckpt.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
    )
