"""Two-tower retrieval model for landmark sequence to subtitle matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkVideoEncoder(nn.Module):
    """BiGRU encoder with mask-aware temporal pooling over frame features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.rnn(x)
        frame_has_signal = (x.abs().sum(dim=-1) > 0).float()
        denom = frame_has_signal.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (encoded * frame_has_signal.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(self.proj(pooled), dim=-1)


class TfidfTextEncoder(nn.Module):
    """Projection tower for frozen TF-IDF subtitle vectors."""

    def __init__(self, input_dim: int, embedding_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class LandmarkTextRetrievalModel(nn.Module):
    """Encode landmarks and sentence features into a shared embedding space."""

    def __init__(
        self,
        video_input_dim: int,
        text_input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.video = LandmarkVideoEncoder(
            input_dim=video_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.text = TfidfTextEncoder(
            input_dim=text_input_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        return self.video(x)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text(x)

    def logits(self, video_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        video_emb = self.encode_video(video_features)
        text_emb = self.encode_text(text_features)
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        return video_emb @ text_emb.T / temperature


def symmetric_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )
