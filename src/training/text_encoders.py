"""Frozen subtitle feature encoders for retrieval training."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.training.devices import resolve_device


class TextFeatureEncoder:
    """Small wrapper around TF-IDF or frozen sentence-transformer features."""

    def __init__(
        self,
        kind: str = "tfidf",
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_features: int = 20000,
        max_ngram: int = 2,
        min_df: int = 1,
        batch_size: int = 64,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        self.kind = kind
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        self.local_files_only = bool(local_files_only)
        self.vectorizer: TfidfVectorizer | None = None
        self.model = None
        self._cache: dict[str, np.ndarray] = {}

        if kind == "tfidf":
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, int(max_ngram)),
                max_features=int(max_features),
                min_df=int(min_df),
            )
        elif kind in {"sentence_transformer", "sentence-transformer", "minilm"}:
            self.kind = "sentence_transformer"
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "text.encoder='sentence_transformer' requires the "
                    "'sentence-transformers' package. Install/update the env "
                    "with: conda env update -f environment.yml --prune"
                ) from exc
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
        else:
            raise ValueError(f"unknown text encoder: {kind!r}")

    def fit(self, sentences: list[str]) -> None:
        sentences = [str(s) for s in sentences]
        if self.kind == "tfidf":
            assert self.vectorizer is not None
            self.vectorizer.fit(sentences)
        else:
            self.transform_numpy(sentences)

    @property
    def output_dim(self) -> int:
        if self.kind == "tfidf":
            assert self.vectorizer is not None
            return len(self.vectorizer.get_feature_names_out())
        if not self._cache:
            self.transform_numpy([""])
        return int(next(iter(self._cache.values())).shape[0])

    def transform_numpy(self, sentences: list[str]) -> np.ndarray:
        sentences = [str(s) for s in sentences]
        if self.kind == "tfidf":
            assert self.vectorizer is not None
            return self.vectorizer.transform(sentences).astype(np.float32).toarray()

        missing = [s for s in dict.fromkeys(sentences) if s not in self._cache]
        if missing:
            assert self.model is not None
            encoded = self.model.encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)
            for sentence, vec in zip(missing, encoded):
                self._cache[sentence] = vec
        return np.stack([self._cache[s] for s in sentences]).astype(np.float32)

    def transform_tensor(self, sentences: list[str], device: str) -> torch.Tensor:
        return torch.from_numpy(self.transform_numpy(sentences)).to(device)

    def save(self, out_dir: Path) -> dict:
        if self.kind == "tfidf":
            path = out_dir / "vectorizer.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            return {"text_encoder_type": "tfidf", "vectorizer_path": str(path)}
        return {
            "text_encoder_type": "sentence_transformer",
            "text_model_name": self.model_name,
            "text_embedding_dim": self.output_dim,
            "text_local_files_only": True,
        }


def build_text_encoder_from_config(config: dict, cfg_get) -> TextFeatureEncoder:
    return TextFeatureEncoder(
        kind=str(cfg_get(config, "text.encoder", "tfidf")),
        model_name=str(
            cfg_get(config, "text.sentence_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        max_features=int(cfg_get(config, "text.tfidf_max_features", 20000)),
        max_ngram=int(cfg_get(config, "text.tfidf_max_ngram", 2)),
        min_df=int(cfg_get(config, "text.tfidf_min_df", 1)),
        batch_size=int(cfg_get(config, "text.embedding_batch_size", 64)),
        device=str(cfg_get(config, "text.embedding_device", cfg_get(config, "training.device", "auto"))),
        local_files_only=bool(cfg_get(config, "text.local_files_only", False)),
    )


def build_text_encoder_from_checkpoint(ckpt: dict, ckpt_path: Path) -> TextFeatureEncoder:
    encoder_type = ckpt.get("text_encoder_type")
    if encoder_type is None:
        encoder_type = "tfidf" if "vectorizer_path" in ckpt else "sentence_transformer"

    if encoder_type == "tfidf":
        vec_path = Path(ckpt.get("vectorizer_path", ckpt_path.parent / "vectorizer.pkl"))
        if not vec_path.is_absolute() and not vec_path.exists():
            vec_path = ckpt_path.parent / vec_path.name
        if not vec_path.exists():
            raise FileNotFoundError(f"could not find vectorizer at {vec_path}")
        encoder = TextFeatureEncoder(kind="tfidf")
        with open(vec_path, "rb") as f:
            encoder.vectorizer = pickle.load(f)
        return encoder

    config = ckpt["config"]
    return TextFeatureEncoder(
        kind="sentence_transformer",
        model_name=str(ckpt.get(
            "text_model_name",
            config.get("text", {}).get("sentence_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )),
        batch_size=int(config.get("text", {}).get("embedding_batch_size", 64)),
        device=str(config.get("text", {}).get("embedding_device", config.get("training", {}).get("device", "auto"))),
        local_files_only=bool(ckpt.get("text_local_files_only", True)),
    )
