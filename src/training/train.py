"""Train landmark sequence models from a YAML config.

Supports single-GPU, multi-GPU (torchrun / DDP), hard negative mining, and
optional end-to-end text encoder fine-tuning.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.dataset.loaders import How2SignLandmarkRetrievalDataset
from src.models.retrieval import (
    LandmarkTextRetrievalModel,
    contrastive_loss_with_hard_negatives,
    symmetric_contrastive_loss,
)
from src.training.devices import resolve_device
from src.training.hard_negatives import HardNegativeMiner
from src.training.text_encoders import TextFeatureEncoder, build_text_encoder_from_config


# ──────────────────────── Utilities ───────────────────────────────────────────

def _cfg(config: dict, path: str, default=None):
    cur = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _split_rows(
    meta: pd.DataFrame,
    split: str,
    val_split: str,
    val_frac: float,
    seed: int,
    *,
    test_split: str = "test",
    test_frac: float = 0.1,
    group_by: str | None = "video_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test rows, creating grouped splits when needed."""
    if "split" in meta.columns and {split, val_split, test_split}.issubset(set(meta["split"])):
        return (
            meta[meta["split"] == split].reset_index(drop=True),
            meta[meta["split"] == val_split].reset_index(drop=True),
            meta[meta["split"] == test_split].reset_index(drop=True),
        )

    rows = meta[meta["split"] == split].copy() if "split" in meta.columns else meta.copy()
    if len(rows) < 3:
        raise ValueError("need at least three cached clips to create train/val/test splits")

    val_frac, test_frac = float(val_frac), float(test_frac)
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and sum to less than 1")

    if group_by and group_by in rows.columns and rows[group_by].notna().any():
        groups = rows[[group_by]].drop_duplicates().sample(frac=1.0, random_state=seed)
        n_g = len(groups)
        n_val = max(1, int(round(n_g * val_frac)))
        n_test = max(1, int(round(n_g * test_frac)))
        if n_val + n_test >= n_g:
            n_val = max(1, min(n_val, n_g - 2))
            n_test = max(1, min(n_test, n_g - n_val - 1))
        val_groups = set(groups.iloc[:n_val][group_by])
        test_groups = set(groups.iloc[n_val : n_val + n_test][group_by])
        return (
            rows[~rows[group_by].isin(val_groups | test_groups)].reset_index(drop=True),
            rows[rows[group_by].isin(val_groups)].reset_index(drop=True),
            rows[rows[group_by].isin(test_groups)].reset_index(drop=True),
        )

    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(rows) * val_frac)))
    n_test = max(1, int(round(len(rows) * test_frac)))
    if n_val + n_test >= len(rows):
        n_val = max(1, min(n_val, len(rows) - 2))
        n_test = max(1, min(n_test, len(rows) - n_val - 1))
    return (
        rows.iloc[n_val + n_test :].reset_index(drop=True),
        rows.iloc[:n_val].reset_index(drop=True),
        rows.iloc[n_val : n_val + n_test].reset_index(drop=True),
    )


def _collate(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "sentence": [item["sentence"] for item in batch],
        "id": [item["id"] for item in batch],
        "index": [item["index"] for item in batch],
    }


def _text_tensor(text_encoder: TextFeatureEncoder, sentences: list[str], device: str) -> torch.Tensor:
    return text_encoder.transform_tensor(sentences, device)


# ──────────────────────── DDP helpers ─────────────────────────────────────────

def _is_ddp() -> bool:
    return "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1


def _ddp_setup() -> tuple[int, int, int]:
    """Initialise process group; return (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def _ddp_cleanup() -> None:
    dist.destroy_process_group()


# ──────────────────────── Evaluation ──────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    model: LandmarkTextRetrievalModel,
    loader: DataLoader,
    encode_text_fn,
    device: str,
    use_trainable_text: bool,
) -> dict[str, float]:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    video_embs, sentences = [], []
    for batch in loader:
        x = batch["features"].to(device)
        video_embs.append(raw_model.encode_video(x).cpu())
        sentences.extend(batch["sentence"])

    video = torch.cat(video_embs).to(device)
    if use_trainable_text:
        text = raw_model.encode_text(sentences)
    else:
        text_feat = encode_text_fn(sentences)
        if not isinstance(text_feat, torch.Tensor):
            text_feat = torch.from_numpy(text_feat).to(device)
        else:
            text_feat = text_feat.to(device)
        text = raw_model.encode_text(text_feat)

    scores = video @ text.T
    labels = torch.arange(scores.shape[0], device=device)
    top1 = (scores.argmax(dim=1) == labels).float().mean().item()
    k = min(5, scores.shape[1])
    topk = (scores.topk(k, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()
>>>>>>> 8e902fb (feat: Added LandmarkTransformerEncoder, TrainableTextEncoder and HardNegativeMiner, updated the existing files so it is compatible)

    # MRR
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
    for i in range(scores.shape[0]):
        ranks[i] = (order[i] == i).nonzero(as_tuple=False)[0].item() + 1
    mrr = (1.0 / ranks.float()).mean().item()
    median_rank = float(ranks.float().median().item())

    raw_model.train()
    return {"top1": top1, "top5": topk, "mrr": mrr, "median_rank": median_rank}


# ──────────────────────── LR schedule ─────────────────────────────────────────

def _cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _set_lr(opt: torch.optim.Optimizer, lr: float, text_lr_scale: float | None = None) -> None:
    for pg in opt.param_groups:
        if text_lr_scale is not None and pg.get("is_text_backbone"):
            pg["lr"] = lr * text_lr_scale
        else:
            pg["lr"] = lr


# ──────────────────────── Wandb ───────────────────────────────────────────────

def _maybe_init_wandb(config: dict, rank: int) -> object | None:
    if rank != 0 or not _cfg(config, "logging.wandb", False):
        return None
    try:
        import wandb
        wandb.init(
            project=str(_cfg(config, "logging.wandb_project", "hand2string")),
            name=str(_cfg(config, "logging.wandb_run_name", "auto")) or None,
            config=config,
        )
        return wandb
    except ImportError:
        print("[warning] wandb not installed; skipping W&B logging.")
        return None


def _wandb_log(wb, metrics: dict, step: int) -> None:
    if wb is not None:
        wb.log(metrics, step=step)


# ──────────────────────── Main train function ─────────────────────────────────

def train(config_path: str) -> None:
<<<<<<< HEAD
    raise NotImplementedError
=======
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── DDP setup ────────────────────────────────────────────────────────────
    use_ddp = _is_ddp()
    if use_ddp:
        rank, local_rank, world_size = _ddp_setup()
        device = f"cuda:{local_rank}"
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = resolve_device(str(_cfg(config, "training.device", "auto")))

    is_main = rank == 0

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = int(_cfg(config, "training.seed", 0))
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # ── Data ─────────────────────────────────────────────────────────────────
    root = Path(_cfg(config, "dataset.root", "data/how2sign_landmarks"))
    meta_path = root / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run scripts/extract_landmarks.py first.")

    meta = pd.read_parquet(meta_path)
    train_rows, val_rows, test_rows = _split_rows(
        meta,
        split=str(_cfg(config, "dataset.split", "train")),
        val_split=str(_cfg(config, "dataset.val_split", "val")),
        val_frac=float(_cfg(config, "dataset.val_frac", 0.2)),
        seed=seed,
        test_split=str(_cfg(config, "dataset.test_split", "test")),
        test_frac=float(_cfg(config, "dataset.test_frac", 0.1)),
        group_by=_cfg(config, "dataset.group_split_by", "video_id"),
    )

    window_size = int(_cfg(config, "training.window_size", 128))
    layout = str(_cfg(config, "preprocessing.landmark_layout", "full"))
    batch_size = int(_cfg(config, "training.batch_size", 32))
    num_workers = int(_cfg(config, "training.num_workers", 4))

    train_ds = How2SignLandmarkRetrievalDataset(root, split=None, window_size=window_size, layout=layout, rows=train_rows)
    val_ds = How2SignLandmarkRetrievalDataset(root, split=None, window_size=window_size, layout=layout, rows=val_rows)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    # Non-distributed loader used for hard-negative refresh (avoids gather complexity)
    train_dl_plain = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

    # ── Text encoder (frozen baseline) ────────────────────────────────────────
    text_encoder = build_text_encoder_from_config(config, _cfg)
    all_train_sentences = train_rows["sentence"].astype(str).tolist()
    text_encoder.fit(all_train_sentences)

    # ── Model ─────────────────────────────────────────────────────────────────
    sample = train_ds[0]["features"]
    video_input_dim = sample.shape[-1]
    text_dim = text_encoder.output_dim
    model_type = str(_cfg(config, "model.type", "retrieval_transformer"))

    model = LandmarkTextRetrievalModel(
        video_input_dim=video_input_dim,
        text_input_dim=text_dim,
        hidden_dim=int(_cfg(config, "model.hidden_dim", 256)),
        num_layers=int(_cfg(config, "model.num_layers", 4)),
        embedding_dim=int(_cfg(config, "model.embedding_dim", 256)),
        dropout=float(_cfg(config, "model.dropout", 0.1)),
        temperature=float(_cfg(config, "model.temperature", 0.07)),
        encoder_type=model_type,
        nhead=int(_cfg(config, "model.nhead", 4)),
        dim_feedforward=int(_cfg(config, "model.dim_feedforward", 1024)),
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr = float(_cfg(config, "training.lr", 5e-4))
    min_lr = float(_cfg(config, "training.min_lr", 1e-5))
    weight_decay = float(_cfg(config, "training.weight_decay", 1e-4))
    total_epochs = int(_cfg(config, "training.epochs", 30))
    lr_warmup_epochs = int(_cfg(config, "training.lr_warmup_epochs", 3))
    scheduler_type = str(_cfg(config, "training.scheduler", "cosine"))

    raw_model = model.module if use_ddp else model
    opt = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Hard-negative miner ───────────────────────────────────────────────────
    hn_enabled = bool(_cfg(config, "training.hard_negatives.enabled", True))
    hn_warmup = int(_cfg(config, "training.hard_negatives.warmup_epochs", 3))
    hn_refresh_every = int(_cfg(config, "training.hard_negatives.refresh_every", 1))
    hn_k = int(_cfg(config, "training.hard_negatives.k_hard", 4))
    miner = HardNegativeMiner(k_hard=hn_k) if hn_enabled else None

    # ── Text fine-tuning ──────────────────────────────────────────────────────
    text_finetune = bool(_cfg(config, "text.finetune", True))
    text_ft_warmup = int(_cfg(config, "text.warmup_epochs", 5))
    text_lr_scale = float(_cfg(config, "text.lr_scale", 0.05))
    text_model_name = str(_cfg(config, "text.sentence_model", "sentence-transformers/all-MiniLM-L6-v2"))
    use_trainable_text = False

    # ── Output dir & logging ──────────────────────────────────────────────────
    out = Path(_cfg(config, "training.out_dir", "runs/how2sign_transformer"))
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        text_checkpoint = text_encoder.save(out)
        train_rows.assign(effective_split="train").to_parquet(out / "split_train.parquet", index=False)
        val_rows.assign(effective_split="val").to_parquet(out / "split_val.parquet", index=False)
        test_rows.assign(effective_split="test").to_parquet(out / "split_test.parquet", index=False)
        print(
            f"[rank0] Training {model_type}: {len(train_ds)} train / {len(val_ds)} val / "
            f"{len(test_rows)} test | video_dim={video_input_dim} text_dim={text_dim} "
            f"device={device} world_size={world_size}"
        )
    else:
        text_checkpoint = {}

    if use_ddp:
        dist.barrier()

    wb = _maybe_init_wandb(config, rank)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_top1 = -1.0
    history: list[list[float]] = []

    def _frozen_encode_text_fn(sentences: list[str]) -> torch.Tensor:
        return text_encoder.transform_tensor(sentences, "cpu")

    def _trainable_encode_text_fn(sentences: list[str]) -> torch.Tensor:
        with torch.no_grad():
            return raw_model.encode_text(sentences).cpu()

    for epoch in range(1, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Activate text fine-tuning ─────────────────────────────────────────
        if text_finetune and not use_trainable_text and epoch > text_ft_warmup:
            if is_main:
                print(f"[epoch {epoch}] unfreezing text encoder (lr_scale={text_lr_scale})")
            enc = raw_model.attach_trainable_text(text_model_name, dropout=float(_cfg(config, "model.dropout", 0.1)))
            enc.to(device)
            use_trainable_text = True

            # Rebuild optimizer with an extra low-lr param group for the backbone
            text_pg = enc.named_param_groups(base_lr=lr, lr_scale=text_lr_scale)
            for pg in text_pg:
                pg["is_text_backbone"] = True
                pg["weight_decay"] = weight_decay
            opt = torch.optim.AdamW(
                list(raw_model.video.parameters())
                + list(raw_model.text.parameters())
                + [raw_model.log_temperature],
                lr=lr,
                weight_decay=weight_decay,
            )
            for pg in text_pg:
                opt.add_param_group(pg)

        # ── LR schedule ───────────────────────────────────────────────────────
        if scheduler_type == "cosine":
            current_lr = _cosine_lr(epoch, total_epochs, lr_warmup_epochs, lr, min_lr)
            _set_lr(opt, current_lr, text_lr_scale if use_trainable_text else None)
        else:
            current_lr = lr

        # ── Hard-negative refresh ─────────────────────────────────────────────
        if miner is not None and epoch > hn_warmup and (epoch - hn_warmup - 1) % hn_refresh_every == 0:
            if is_main:
                print(f"[epoch {epoch}] refreshing hard negatives …")
            encode_fn = _trainable_encode_text_fn if use_trainable_text else _frozen_encode_text_fn
            miner.refresh(model, train_dl_plain, encode_fn, device, all_train_sentences)

        # ── Batch loop ────────────────────────────────────────────────────────
        raw_model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_dl:
            x = batch["features"].to(device)
            batch_indices = batch["index"]

            # Encode text (frozen or trainable path)
            if use_trainable_text:
                text_input = batch["sentence"]  # List[str]
                video_emb = raw_model.encode_video(x)
                text_emb = raw_model.encode_text(text_input)
            else:
                txt_feat = _text_tensor(text_encoder, batch["sentence"], device)
                video_emb = raw_model.encode_video(x)
                text_emb = raw_model.encode_text(txt_feat)

            # Loss: standard or hard-negative augmented
            if miner is not None and miner.ready:
                hard_sentences = miner.get_hard_neg_sentences(batch_indices, all_train_sentences)
                if use_trainable_text:
                    hard_emb = raw_model.encode_text(hard_sentences)
                else:
                    hard_feat = _text_tensor(text_encoder, hard_sentences, device)
                    hard_emb = raw_model.encode_text(hard_feat)
                loss = contrastive_loss_with_hard_negatives(
                    video_emb, text_emb, hard_emb, raw_model.log_temperature
                )
            else:
                temperature = raw_model.log_temperature.exp().clamp_min(1e-4)
                logits = video_emb @ text_emb.T / temperature
                loss = symmetric_contrastive_loss(logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * x.shape[0]
            total_examples += x.shape[0]

        # ── Validation (main rank only to avoid duplicate prints) ──────────────
        if use_ddp:
            dist.barrier()

        metrics = _evaluate(model, val_dl, _frozen_encode_text_fn, device, use_trainable_text)
        train_loss = total_loss / max(total_examples, 1)
        history.append([epoch, train_loss, metrics["top1"], metrics["top5"], metrics["mrr"]])

        if is_main:
            print(
                f"epoch {epoch:03d} | lr={current_lr:.2e} | loss={train_loss:.4f} | "
                f"val_top1={metrics['top1']:.4f} val_top5={metrics['top5']:.4f} "
                f"mrr={metrics['mrr']:.4f} median_rank={metrics['median_rank']:.1f}"
            )
            _wandb_log(wb, {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}, "lr": current_lr}, step=epoch)

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                ckpt = {
                    "model_state": raw_model.state_dict(),
                    "config": config,
                    "video_input_dim": video_input_dim,
                    "text_input_dim": text_dim,
                    "encoder_type": model_type,
                    "layout": layout,
                    "window_size": window_size,
                    "has_trainable_text": use_trainable_text,
                }
                if use_trainable_text:
                    ckpt["trainable_text_state"] = raw_model.trainable_text.state_dict()
                    ckpt["text_model_name"] = text_model_name
                ckpt.update(text_checkpoint)
                torch.save(ckpt, out / "best.pt")

        if use_ddp:
            dist.barrier()

    # ── Save history & cleanup ─────────────────────────────────────────────────
    if is_main:
        np.savez(out / "history.npz", history=np.asarray(history, dtype=np.float32))
        print(f"Best val_top1: {best_top1:.4f}  →  {out / 'best.pt'}")
        if wb is not None:
            wb.finish()

    if use_ddp:
        _ddp_cleanup()
