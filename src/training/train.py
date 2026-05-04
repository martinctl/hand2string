"""Train landmark sequence models from a YAML config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.loaders import How2SignLandmarkRetrievalDataset
from src.models.retrieval import LandmarkTextRetrievalModel, symmetric_contrastive_loss
from src.training.devices import resolve_device
from src.training.text_encoders import TextFeatureEncoder, build_text_encoder_from_config


def _cfg(config: dict, path: str, default=None):
    cur = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _split_rows(meta: pd.DataFrame, split: str, val_split: str, val_frac: float, seed: int):
    if "split" in meta.columns and val_split in set(meta["split"]):
        train_rows = meta[meta["split"] == split].reset_index(drop=True)
        val_rows = meta[meta["split"] == val_split].reset_index(drop=True)
        return train_rows, val_rows

    rows = meta[meta["split"] == split].copy() if "split" in meta.columns else meta.copy()
    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(rows) < 2:
        raise ValueError("need at least two cached clips to create a validation split")
    n_val = max(1, int(round(len(rows) * val_frac)))
    n_val = min(n_val, len(rows) - 1)
    return rows.iloc[n_val:].reset_index(drop=True), rows.iloc[:n_val].reset_index(drop=True)


def _collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "sentence": [item["sentence"] for item in batch],
        "id": [item["id"] for item in batch],
    }


def _text_tensor(text_encoder: TextFeatureEncoder, sentences: list[str], device: str) -> torch.Tensor:
    return text_encoder.transform_tensor(sentences, device)


@torch.no_grad()
def _evaluate(
    model: LandmarkTextRetrievalModel,
    loader: DataLoader,
    text_encoder: TextFeatureEncoder,
    device: str,
) -> dict[str, float]:
    model.eval()
    video_embs, sentences = [], []
    for batch in loader:
        x = batch["features"].to(device)
        video_embs.append(model.encode_video(x).cpu())
        sentences.extend(batch["sentence"])

    video = torch.cat(video_embs, dim=0).to(device)
    text_features = _text_tensor(text_encoder, sentences, device)
    text = model.encode_text(text_features)
    scores = video @ text.T
    labels = torch.arange(scores.shape[0], device=device)
    top1 = (scores.argmax(dim=1) == labels).float().mean().item()
    k = min(5, scores.shape[1])
    topk = (scores.topk(k, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": topk}


def train(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = _cfg(config, "dataset.name", "how2sign")
    model_type = _cfg(config, "model.type", "retrieval_bigru")
    if dataset_name != "how2sign" or model_type not in {"retrieval", "retrieval_bigru"}:
        raise NotImplementedError(
            "this training entrypoint currently implements the How2Sign "
            "sentence-retrieval baseline only"
        )

    seed = int(_cfg(config, "training.seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = Path(_cfg(config, "dataset.root", "data/how2sign_landmarks"))
    meta_path = root / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} does not exist. Run scripts/extract_landmarks.py first."
        )
    meta = pd.read_parquet(meta_path)
    train_rows, val_rows = _split_rows(
        meta,
        split=str(_cfg(config, "dataset.split", "train")),
        val_split=str(_cfg(config, "dataset.val_split", "val")),
        val_frac=float(_cfg(config, "dataset.val_frac", 0.2)),
        seed=seed,
    )

    window_size = int(_cfg(config, "training.window_size", 128))
    layout = str(_cfg(config, "preprocessing.landmark_layout", "full"))
    train_ds = How2SignLandmarkRetrievalDataset(
        root, split=None, window_size=window_size, layout=layout, rows=train_rows
    )
    val_ds = How2SignLandmarkRetrievalDataset(
        root, split=None, window_size=window_size, layout=layout, rows=val_rows
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=int(_cfg(config, "training.batch_size", 32)),
        shuffle=True,
        num_workers=int(_cfg(config, "training.num_workers", 0)),
        collate_fn=_collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(_cfg(config, "training.batch_size", 32)),
        shuffle=False,
        num_workers=int(_cfg(config, "training.num_workers", 0)),
        collate_fn=_collate,
    )

    text_encoder = build_text_encoder_from_config(config, _cfg)
    text_encoder.fit(train_rows["sentence"].astype(str).tolist())

    sample = train_ds[0]["features"]
    text_dim = text_encoder.output_dim
    device = resolve_device(str(_cfg(config, "training.device", "auto")))

    model = LandmarkTextRetrievalModel(
        video_input_dim=sample.shape[-1],
        text_input_dim=text_dim,
        hidden_dim=int(_cfg(config, "model.hidden_dim", 256)),
        num_layers=int(_cfg(config, "model.num_layers", 2)),
        embedding_dim=int(_cfg(config, "model.embedding_dim", 256)),
        dropout=float(_cfg(config, "model.dropout", 0.2)),
        temperature=float(_cfg(config, "model.temperature", 0.07)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(_cfg(config, "training.lr", 1e-3)),
        weight_decay=float(_cfg(config, "training.weight_decay", 1e-4)),
    )

    out = Path(_cfg(config, "training.out_dir", "runs/how2sign_retrieval"))
    out.mkdir(parents=True, exist_ok=True)
    text_checkpoint = text_encoder.save(out)

    print(
        f"Training retrieval baseline: {len(train_ds)} train / {len(val_ds)} val | "
        f"video_dim={sample.shape[-1]} text_dim={text_dim} "
        f"text_encoder={text_encoder.kind} device={device}"
    )
    best_top1 = -1.0
    history = []
    for epoch in range(1, int(_cfg(config, "training.epochs", 20)) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_dl:
            x = batch["features"].to(device)
            txt = _text_tensor(text_encoder, batch["sentence"], device)
            logits = model.logits(x, txt)
            loss = symmetric_contrastive_loss(logits)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.shape[0]
            total_examples += x.shape[0]

        metrics = _evaluate(model, val_dl, text_encoder, device)
        train_loss = total_loss / max(total_examples, 1)
        history.append([epoch, train_loss, metrics["top1"], metrics["top5"]])
        print(
            f"epoch {epoch:03d} loss {train_loss:.4f} "
            f"val_top1 {metrics['top1']:.4f} val_top5 {metrics['top5']:.4f}"
        )

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            checkpoint = {
                    "model_state": model.state_dict(),
                    "config": config,
                    "video_input_dim": sample.shape[-1],
                    "text_input_dim": text_dim,
                    "layout": layout,
                    "window_size": window_size,
            }
            checkpoint.update(text_checkpoint)
            torch.save(checkpoint, out / "best.pt")

    np.savez(out / "history.npz", history=np.asarray(history, dtype=np.float32))
    print(f"Best val_top1: {best_top1:.4f} -> {out / 'best.pt'}")
