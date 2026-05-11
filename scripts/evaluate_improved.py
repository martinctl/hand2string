"""Evaluate a trained retrieval checkpoint (supports BiGRU and Transformer models).

Qualitative + quantitative retrieval metrics on val or test split.

Usage:
    python scripts/evaluate_improved.py --ckpt runs/how2sign_transformer/best.pt
    python scripts/evaluate_improved.py --ckpt runs/how2sign_transformer/best.pt --split test
    python scripts/evaluate_improved.py --ckpt runs/how2sign_transformer/best.pt \\
        --split test --out-csv results/test_retrieval.csv --out-dir results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.loaders import How2SignLandmarkRetrievalDataset
from src.models.retrieval import LandmarkTextRetrievalModel, build_model_from_checkpoint
from src.training.devices import resolve_device
from src.training.text_encoders import build_text_encoder_from_checkpoint
from src.training.train import _cfg, _collate, _split_rows


# ──────────────────────── CLI ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/how2sign_transformer/best.pt"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default=None, help="val (default) | test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=20, help="qualitative examples to print")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="directory for plots and JSON")
    parser.add_argument("--all-splits", action="store_true", help="evaluate both val and test")
    return parser.parse_args()


# ──────────────────────── Loading ─────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _rows_for_split(config: dict, root: Path, split: str | None) -> pd.DataFrame:
    """Resolve the metadata rows for a given split name."""
    meta = pd.read_parquet(root / "metadata.parquet")

    # Try the saved split parquet first (from training run)
    run_dir = Path(config.get("training", {}).get("out_dir", "runs/how2sign_transformer"))
    split_name = "val" if split in {None, "val", "validation"} else split
    split_file = run_dir / f"split_{split_name}.parquet"
    if split_file.exists():
        return pd.read_parquet(split_file).reset_index(drop=True)

    # Fallback: recompute the split from metadata
    _, val_rows, test_rows = _split_rows(
        meta,
        split=str(_cfg(config, "dataset.split", "train")),
        val_split=str(_cfg(config, "dataset.val_split", "val")),
        val_frac=float(_cfg(config, "dataset.val_frac", 0.2)),
        seed=int(_cfg(config, "training.seed", 0)),
        test_split=str(_cfg(config, "dataset.test_split", "test")),
        test_frac=float(_cfg(config, "dataset.test_frac", 0.1)),
        group_by=_cfg(config, "dataset.group_split_by", "video_id"),
    )
    return test_rows if split == "test" else val_rows


# ──────────────────────── Encoding ────────────────────────────────────────────

@torch.no_grad()
def _encode_all(
    model: LandmarkTextRetrievalModel,
    loader: DataLoader,
    text_encoder,
    device: str,
    use_trainable_text: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Return (video_emb, text_emb, sentences, ids) — all on CPU."""
    model.eval()
    video_embs, sentences, ids = [], [], []

    for batch in loader:
        x = batch["features"].to(device)
        video_embs.append(model.encode_video(x).cpu())
        sentences.extend(batch["sentence"])
        ids.extend(batch["id"])

    video_emb = torch.cat(video_embs)

    if use_trainable_text:
        # Fine-tuned encoder: pass sentences directly
        chunk = 64
        text_chunks = []
        for i in range(0, len(sentences), chunk):
            text_chunks.append(model.encode_text(sentences[i : i + chunk]).cpu())
        text_emb = torch.cat(text_chunks)
    else:
        text_feat = text_encoder.transform_tensor(sentences, device)
        text_emb = model.encode_text(text_feat).cpu()

    return video_emb, text_emb, sentences, ids


# ──────────────────────── Metrics ─────────────────────────────────────────────

def _rank_metrics(scores: torch.Tensor, top_k: int) -> tuple[dict[str, float], np.ndarray]:
    N = scores.shape[0]
    order = scores.argsort(dim=1, descending=True)
    ranks = np.empty(N, dtype=np.int64)
    for i in range(N):
        ranks[i] = int((order[i] == i).nonzero(as_tuple=False)[0].item()) + 1
    k5 = min(5, N)
    k10 = min(10, N)
    ktop = min(top_k, N)
    metrics = {
        "n": N,
        "random_top1": 1.0 / N,
        "random_top5": k5 / N,
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= k5).mean()),
        "top10": float((ranks <= k10).mean()),
        f"top{ktop}": float((ranks <= ktop).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
    }
    return metrics, ranks


# ──────────────────────── Plots ───────────────────────────────────────────────

def _plot_rank_hist(ranks: np.ndarray, split_name: str, metrics: dict, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    cap = min(int(np.max(ranks)), 100)
    bins = np.arange(1, cap + 2) - 0.5
    plt.hist(np.minimum(ranks, cap), bins=bins, color="#3B82F6", alpha=0.85)
    plt.axvline(metrics["median_rank"], color="#EF4444", linestyle="--", label=f"median={metrics['median_rank']:.1f}")
    plt.title(f"{split_name} rank distribution  (top1={metrics['top1']:.3f}  mrr={metrics['mrr']:.3f})")
    plt.xlabel("rank of correct subtitle (clipped at 100)")
    plt.ylabel("queries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_name}_rank_hist.png", dpi=160)
    plt.close()


def _plot_history(run_dir: Path, out_dir: Path) -> None:
    p = run_dir / "history.npz"
    if not p.exists():
        return
    history = np.load(p)["history"]
    if history.size == 0:
        return
    # history columns: epoch, loss, top1, top5, [mrr]
    epoch, loss = history[:, 0], history[:, 1]
    top1, top5 = history[:, 2], history[:, 3]
    mrr = history[:, 4] if history.shape[1] > 4 else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epoch, loss, color="#111827")
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].plot(epoch, top1, label="top1", color="#2563EB")
    axes[1].plot(epoch, top5, label="top5", color="#059669")
    if mrr is not None:
        axes[1].plot(epoch, mrr, label="mrr", color="#D97706", linestyle="--")
    axes[1].set(title="Validation retrieval", xlabel="epoch", ylabel="accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=160)
    plt.close()


# ──────────────────────── Per-split evaluation ────────────────────────────────

@torch.no_grad()
def evaluate_split(
    split_name: str,
    ckpt: dict,
    ckpt_path: Path,
    root: Path,
    device: str,
    batch_size: int,
    top_k: int,
    max_queries: int,
    out_dir: Path | None,
    out_csv: Path | None,
) -> dict[str, float]:
    config = ckpt["config"]
    rows = _rows_for_split(config, root, split_name)
    window_size = int(ckpt.get("window_size", _cfg(config, "training.window_size", 128)))
    layout = str(ckpt.get("layout", _cfg(config, "preprocessing.landmark_layout", "full")))
    num_workers = int(_cfg(config, "training.num_workers", 0))

    dataset = How2SignLandmarkRetrievalDataset(root, split=None, window_size=window_size, layout=layout, rows=rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate)

    use_trainable_text = bool(ckpt.get("has_trainable_text", False))
    text_encoder = None if use_trainable_text else build_text_encoder_from_checkpoint(ckpt, ckpt_path)

    model = build_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    video_emb, text_emb, sentences, ids = _encode_all(
        model, loader, text_encoder, device, use_trainable_text
    )
    scores = video_emb @ text_emb.T  # cosine sim (both L2-normalised)

    metrics, ranks = _rank_metrics(scores, top_k)
    k = min(top_k, scores.shape[1])
    top = scores.topk(k, dim=1)

    # ── Qualitative examples ──────────────────────────────────────────────────
    n_show = min(max_queries, len(dataset))
    print(f"\n{'='*90}")
    print(f"Split: {split_name}  |  {len(dataset)} queries  |  device={device}")
    print(f"top1={metrics['top1']:.4f}  top5={metrics['top5']:.4f}  "
          f"top10={metrics['top10']:.4f}  mrr={metrics['mrr']:.4f}  "
          f"median_rank={metrics['median_rank']:.1f}")
    print(f"random top1={metrics['random_top1']:.4f}  (improvement x{metrics['top1']/max(metrics['random_top1'],1e-9):.1f})")
    for i in range(n_show):
        print(f"\n{'─'*88}")
        print(f"query {i+1}/{len(dataset)} | id={ids[i]} | true_rank={ranks[i]}")
        print(f"GT: {sentences[i]}")
        for rank_pos, (j, score) in enumerate(zip(top.indices[i].tolist(), top.values[i].tolist()), start=1):
            mark = "★" if j == i else " "
            print(f"{mark} {rank_pos:02d}. score={score:+.4f}  {sentences[j]}")

    # ── CSV export ────────────────────────────────────────────────────────────
    if out_csv is not None or out_dir is not None:
        rows_out = []
        for i in range(scores.shape[0]):
            for rk, (j, sc) in enumerate(zip(top.indices[i].tolist(), top.values[i].tolist()), start=1):
                rows_out.append({
                    "split": split_name,
                    "query_index": i,
                    "query_id": ids[i],
                    "ground_truth": sentences[i],
                    "true_rank": int(ranks[i]),
                    "rank": rk,
                    "candidate_id": ids[j],
                    "candidate_sentence": sentences[j],
                    "score": float(sc),
                    "is_match": int(i == j),
                })
        df = pd.DataFrame(rows_out)
        target_csv = out_csv or (out_dir / f"{split_name}_retrieval_top{k}.csv")
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_csv, index=False)
        print(f"\nCSV  →  {target_csv}")

        ranks_df = pd.DataFrame({"split": split_name, "query_id": ids, "sentence": sentences, "true_rank": ranks})
        ranks_csv = out_dir / f"{split_name}_ranks.csv" if out_dir else target_csv.parent / f"{split_name}_ranks.csv"
        ranks_df.to_csv(ranks_csv, index=False)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_rank_hist(ranks, split_name, metrics, out_dir)

    return metrics


# ──────────────────────── Main ────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ckpt = _load_checkpoint(args.ckpt)
    config = ckpt["config"]
    root = args.data_root or Path(_cfg(config, "dataset.root", "data/how2sign_landmarks"))
    device = resolve_device(args.device)
    out_dir = args.out_dir or args.ckpt.parent / "analysis"

    encoder_type = ckpt.get("encoder_type", _cfg(config, "model.type", "unknown"))
    has_trainable = ckpt.get("has_trainable_text", False)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Encoder: {encoder_type}  |  fine-tuned text: {has_trainable}")

    splits = ["val", "test"] if args.all_splits else [args.split or "val"]
    all_metrics = {}
    for split_name in splits:
        all_metrics[split_name] = evaluate_split(
            split_name=split_name,
            ckpt=ckpt,
            ckpt_path=args.ckpt,
            root=root,
            device=device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            max_queries=args.max_queries,
            out_dir=out_dir,
            out_csv=args.out_csv,
        )

    # Plot training curves from the run dir
    _plot_history(args.ckpt.parent, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics JSON  →  {metrics_path}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
