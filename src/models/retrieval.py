"""Two-tower retrieval model for landmark sequence to subtitle matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────── Video encoders ──────────────────────────────────────

class LandmarkVideoEncoder(nn.Module):
    """BiGRU encoder with mask-aware temporal pooling over frame features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        **_,
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


class LandmarkTransformerEncoder(nn.Module):
    """Transformer encoder with CLS-token pooling and mask-aware key-padding.

    Chosen over BiGRU because self-attention captures non-local co-articulation
    patterns across the full sequence simultaneously (Sign Language Transformers,
    CVPR 2020; SMKD, ICCV 2021), whereas GRUs compress history into fixed-size
    hidden states.  Pre-LN (norm_first=True) stabilises training on the small
    How2Sign corpus.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        embedding_dim: int = 256,
        max_len: int = 512,
        **_,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        frame_has_signal = (x.abs().sum(dim=-1) > 0)  # (B, T)

        h = self.input_proj(x)  # (B, T, d_model)
        h = h + self.pos_embed[:, :T]

        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, d_model)

        # True = ignore this position in attention
        cls_attend = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_attend, ~frame_has_signal], dim=1)

        out = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return F.normalize(self.proj(out[:, 0]), dim=-1)  # CLS token


def build_video_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    embedding_dim: int = 256,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    t = encoder_type.lower().replace("retrieval_", "")
    if t in {"bigru", "gru"}:
        return LandmarkVideoEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
    if t == "transformer":
        return LandmarkTransformerEncoder(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=int(kwargs.get("nhead", 4)),
            num_layers=num_layers,
            dim_feedforward=int(kwargs.get("dim_feedforward", hidden_dim * 4)),
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
    raise ValueError(
        f"unknown video encoder type: {encoder_type!r}. "
        "Valid: 'bigru' / 'retrieval_bigru', 'transformer' / 'retrieval_transformer'."
    )


# ──────────────────────── Text encoders ───────────────────────────────────────

class TfidfTextEncoder(nn.Module):
    """Projection tower for frozen text feature vectors (TF-IDF or sentence-transformer)."""

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


class TrainableTextEncoder(nn.Module):
    """HuggingFace transformer wrapped as nn.Module for end-to-end fine-tuning.

    Takes List[str], tokenises on the fly, runs the transformer, mean-pools over
    token embeddings (weighted by attention mask), then projects to embedding_dim.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 256,
        dropout: float = 0.1,
        max_length: int = 128,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "TrainableTextEncoder requires 'transformers'. "
                "Install with: pip install transformers"
            ) from exc

        self.hf_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        hidden_size = self.hf_model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def _mean_pool(token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-6)

    def forward(self, sentences: list[str]) -> torch.Tensor:
        device = next(self.hf_model.parameters()).device
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = self.hf_model(**encoded)
        pooled = self._mean_pool(out.last_hidden_state, encoded["attention_mask"])
        return F.normalize(self.proj(pooled), dim=-1)

    def named_param_groups(self, base_lr: float, lr_scale: float = 0.05) -> list[dict]:
        """Param groups: transformer backbone at scaled LR, projection at full LR."""
        return [
            {"params": list(self.hf_model.parameters()), "lr": base_lr * lr_scale},
            {"params": list(self.proj.parameters()), "lr": base_lr},
        ]


# ──────────────────────── Two-tower model ─────────────────────────────────────

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
        encoder_type: str = "bigru",
        nhead: int = 4,
        dim_feedforward: int = 1024,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.video = build_video_encoder(
            encoder_type=encoder_type,
            input_dim=video_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.text = TfidfTextEncoder(
            input_dim=text_input_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        # Populated by attach_trainable_text() when fine-tuning is activated.
        self.trainable_text: TrainableTextEncoder | None = None
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def attach_trainable_text(self, model_name: str, dropout: float = 0.1) -> TrainableTextEncoder:
        """Replace the frozen text-projection path with a fine-tunable encoder."""
        embedding_dim = self.text.proj[-1].out_features
        self.trainable_text = TrainableTextEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        return self.trainable_text

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        return self.video(x)

    def encode_text(self, x) -> torch.Tensor:
        """x: pre-computed feature tensor (frozen path) or List[str] (fine-tune path)."""
        if self.trainable_text is not None and isinstance(x, list):
            return self.trainable_text(x)
        return self.text(x)

    def logits(self, video_features: torch.Tensor, text_input) -> torch.Tensor:
        video_emb = self.encode_video(video_features)
        text_emb = self.encode_text(text_input)
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        return video_emb @ text_emb.T / temperature


# ──────────────────────── Loss functions ──────────────────────────────────────

def symmetric_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def contrastive_loss_with_hard_negatives(
    video_emb: torch.Tensor,
    text_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor,
    log_temperature: nn.Parameter,
) -> torch.Tensor:
    """Symmetric contrastive loss augmented with one hard-negative text per video.

    video_emb:    (B, D) L2-normalised
    text_emb:     (B, D) matched text embeddings
    hard_neg_emb: (B, D) one hard-negative per video (high cosine sim, wrong label)

    Video-to-text logits cover (B matched + B hard-neg) candidates so the model
    must distinguish the correct match from its hardest impostor.
    """
    B = video_emb.shape[0]
    temperature = log_temperature.exp().clamp_min(1e-4)
    labels = torch.arange(B, device=video_emb.device)

    all_text = torch.cat([text_emb, hard_neg_emb], dim=0)  # (2B, D)
    logits_v2t = video_emb @ all_text.T / temperature       # (B, 2B)
    logits_t2v = text_emb @ video_emb.T / temperature       # (B, B)

    return 0.5 * (
        F.cross_entropy(logits_v2t, labels)
        + F.cross_entropy(logits_t2v, labels)
    )


# ──────────────────────── Checkpoint helpers ──────────────────────────────────

def build_model_from_checkpoint(ckpt: dict) -> LandmarkTextRetrievalModel:
    """Reconstruct a model from a saved checkpoint dict (new and legacy formats)."""
    config = ckpt["config"]

    def _cfg(path: str, default=None):
        cur = config
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    encoder_type = str(ckpt.get("encoder_type", _cfg("model.type", "retrieval_bigru")))
    model = LandmarkTextRetrievalModel(
        video_input_dim=int(ckpt["video_input_dim"]),
        text_input_dim=int(ckpt["text_input_dim"]),
        hidden_dim=int(_cfg("model.hidden_dim", 256)),
        num_layers=int(_cfg("model.num_layers", 2)),
        embedding_dim=int(_cfg("model.embedding_dim", 256)),
        dropout=float(_cfg("model.dropout", 0.2)),
        temperature=float(_cfg("model.temperature", 0.07)),
        encoder_type=encoder_type,
        nhead=int(_cfg("model.nhead", 4)),
        dim_feedforward=int(_cfg("model.dim_feedforward", 1024)),
    )
    if ckpt.get("has_trainable_text"):
        text_model_name = str(ckpt.get(
            "text_model_name",
            _cfg("text.sentence_model", "sentence-transformers/all-MiniLM-L6-v2"),
        ))
        enc = model.attach_trainable_text(
            text_model_name,
            dropout=float(_cfg("model.dropout", 0.1)),
        )
        enc.load_state_dict(ckpt["trainable_text_state"])
    return model
