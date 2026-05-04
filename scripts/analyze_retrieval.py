"""Generate metrics, CSVs, and plots for a retrieval experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_retrieval import _build_model, _encode_video, _load_checkpoint, _rows_for_eval
from src.dataset.loaders import How2SignLandmarkRetrievalDataset
from src.training.devices import resolve_device
from src.training.train import _cfg, _collate, _text_tensor
from src.training.text_encoders import build_text_encoder_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/how2sign_retrieval/best.pt"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _rank_metrics(scores: torch.Tensor) -> tuple[dict[str, float], np.ndarray]:
    labels = torch.arange(scores.shape[0])
    order = scores.argsort(dim=1, descending=True)
    ranks = np.empty(scores.shape[0], dtype=np.int64)
    for i in range(scores.shape[0]):
        ranks[i] = int((order[i] == i).nonzero(as_tuple=False)[0].item()) + 1
    metrics = {
        "n": int(scores.shape[0]),
        "random_top1": 1.0 / scores.shape[0],
        "random_top5": min(5, scores.shape[0]) / scores.shape[0],
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, scores.shape[0])).mean()),
        "top10": float((ranks <= min(10, scores.shape[0])).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
    }
    return metrics, ranks


@torch.no_grad()
def _evaluate_split(
    split_name: str,
    ckpt: dict,
    ckpt_path: Path,
    root: Path,
    device: str,
    batch_size: int,
    top_k: int,
    out_dir: Path,
) -> dict[str, float]:
    config = ckpt["config"]
    rows = _rows_for_eval(config, root, split_name)
    dataset = How2SignLandmarkRetrievalDataset(
        root,
        split=None,
        window_size=int(ckpt.get("window_size", _cfg(config, "training.window_size", 128))),
        layout=str(ckpt.get("layout", _cfg(config, "preprocessing.landmark_layout", "full"))),
        rows=rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(_cfg(config, "training.num_workers", 0)),
        collate_fn=_collate,
    )
    text_encoder = build_text_encoder_from_checkpoint(ckpt, ckpt_path)
    model = _build_model(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    video_emb, sentences, ids = _encode_video(model, loader, device)
    text_features = _text_tensor(text_encoder, sentences, device)
    text_emb = model.encode_text(text_features).cpu()
    scores = video_emb @ text_emb.T
    metrics, ranks = _rank_metrics(scores)

    k = min(top_k, scores.shape[1])
    top = scores.topk(k, dim=1)
    rows_out = []
    for i in range(scores.shape[0]):
        for rank, (j, score) in enumerate(zip(top.indices[i].tolist(), top.values[i].tolist()), start=1):
            rows_out.append({
                "split": split_name,
                "query_index": i,
                "query_id": ids[i],
                "ground_truth": sentences[i],
                "true_rank": int(ranks[i]),
                "rank": rank,
                "candidate_id": ids[j],
                "candidate_sentence": sentences[j],
                "score": float(score),
                "is_match": int(i == j),
            })
    pd.DataFrame(rows_out).to_csv(out_dir / f"{split_name}_retrieval_top{k}.csv", index=False)
    pd.DataFrame({
        "split": split_name,
        "query_id": ids,
        "sentence": sentences,
        "true_rank": ranks,
    }).to_csv(out_dir / f"{split_name}_ranks.csv", index=False)

    plt.figure(figsize=(7, 4))
    bins = np.arange(1, min(scores.shape[0], 100) + 2) - 0.5
    plt.hist(np.minimum(ranks, 100), bins=bins, color="#3B82F6", alpha=0.85)
    plt.axvline(metrics["median_rank"], color="#EF4444", linestyle="--", label="median")
    plt.title(f"{split_name} rank distribution")
    plt.xlabel("rank of correct subtitle (clipped at 100)")
    plt.ylabel("queries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{split_name}_rank_hist.png", dpi=160)
    plt.close()
    return metrics


def _plot_history(run_dir: Path, out_dir: Path) -> None:
    history_path = run_dir / "history.npz"
    if not history_path.exists():
        return
    history = np.load(history_path)["history"]
    if history.size == 0:
        return
    epoch, loss, top1, top5 = history.T
    plt.figure(figsize=(7, 4))
    plt.plot(epoch, loss, label="train loss", color="#111827")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epoch, top1, label="val top1", color="#2563EB")
    plt.plot(epoch, top5, label="val top5", color="#059669")
    plt.xlabel("epoch")
    plt.ylabel("retrieval accuracy")
    plt.title("Validation retrieval during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_retrieval.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ckpt = _load_checkpoint(args.ckpt)
    config = ckpt["config"]
    root = args.data_root or Path(_cfg(config, "dataset.root", "data/how2sign_landmarks"))
    out_dir = args.out_dir or args.ckpt.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    metrics = {}
    for split in ("val", "test"):
        metrics[split] = _evaluate_split(
            split, ckpt, args.ckpt, root, device, args.batch_size, args.top_k, out_dir
        )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    _plot_history(args.ckpt.parent, out_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
