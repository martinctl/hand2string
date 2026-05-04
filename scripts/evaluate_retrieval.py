"""Inspect a trained How2Sign sentence-retrieval checkpoint."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.loaders import How2SignLandmarkRetrievalDataset
from src.models.retrieval import LandmarkTextRetrievalModel
from src.training.devices import resolve_device
from src.training.train import _cfg, _collate, _split_rows, _text_tensor
from src.training.text_encoders import build_text_encoder_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/how2sign_retrieval/best.pt"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default=None, help="metadata split to query; defaults to val split")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=20, help="number of qualitative examples to print")
    parser.add_argument("--out-csv", type=Path, default=None, help="optional CSV of all query/rank rows")
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_model(ckpt: dict) -> LandmarkTextRetrievalModel:
    config = ckpt["config"]
    return LandmarkTextRetrievalModel(
        video_input_dim=int(ckpt["video_input_dim"]),
        text_input_dim=int(ckpt["text_input_dim"]),
        hidden_dim=int(_cfg(config, "model.hidden_dim", 256)),
        num_layers=int(_cfg(config, "model.num_layers", 2)),
        embedding_dim=int(_cfg(config, "model.embedding_dim", 256)),
        dropout=float(_cfg(config, "model.dropout", 0.2)),
        temperature=float(_cfg(config, "model.temperature", 0.07)),
    )


def _rows_for_eval(config: dict, root: Path, split: str | None) -> pd.DataFrame:
    meta = pd.read_parquet(root / "metadata.parquet")
    if split is not None:
        rows = meta[meta["split"] == split].reset_index(drop=True) if "split" in meta else meta
        if len(rows) == 0:
            raise ValueError(f"no rows found for split={split!r}")
        return rows

    _, val_rows = _split_rows(
        meta,
        split=str(_cfg(config, "dataset.split", "train")),
        val_split=str(_cfg(config, "dataset.val_split", "val")),
        val_frac=float(_cfg(config, "dataset.val_frac", 0.2)),
        seed=int(_cfg(config, "training.seed", 0)),
    )
    return val_rows


@torch.no_grad()
def _encode_video(model, loader, device: str) -> tuple[torch.Tensor, list[str], list[str]]:
    model.eval()
    embeddings = []
    sentences: list[str] = []
    ids: list[str] = []
    for batch in loader:
        x = batch["features"].to(device)
        embeddings.append(model.encode_video(x).cpu())
        sentences.extend(batch["sentence"])
        ids.extend(batch["id"])
    return torch.cat(embeddings, dim=0), sentences, ids


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt = _load_checkpoint(args.ckpt)
    config = ckpt["config"]
    root = args.data_root or Path(_cfg(config, "dataset.root", "data/how2sign_landmarks"))
    rows = _rows_for_eval(config, root, args.split)
    layout = str(ckpt.get("layout", _cfg(config, "preprocessing.landmark_layout", "full")))
    window_size = int(ckpt.get("window_size", _cfg(config, "training.window_size", 128)))

    dataset = How2SignLandmarkRetrievalDataset(
        root,
        split=None,
        window_size=window_size,
        layout=layout,
        rows=rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(_cfg(config, "training.num_workers", 0)),
        collate_fn=_collate,
    )

    text_encoder = build_text_encoder_from_checkpoint(ckpt, args.ckpt)
    device = resolve_device(args.device)
    model = _build_model(ckpt)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    video_emb, sentences, ids = _encode_video(model, loader, device)
    text_features = _text_tensor(text_encoder, sentences, device)
    text_emb = model.encode_text(text_features).cpu()
    scores = video_emb @ text_emb.T

    labels = torch.arange(scores.shape[0])
    top_k = min(args.top_k, scores.shape[1])
    ranked = scores.topk(top_k, dim=1)
    top1 = (ranked.indices[:, 0] == labels).float().mean().item()
    topk = (ranked.indices == labels[:, None]).any(dim=1).float().mean().item()
    ranks = []
    order = scores.argsort(dim=1, descending=True)
    for i in range(scores.shape[0]):
        ranks.append(int((order[i] == i).nonzero(as_tuple=False)[0].item()) + 1)

    print(
        f"Retrieval eval: {len(dataset)} queries | device={device} | "
        f"top1={top1:.4f} top{top_k}={topk:.4f} median_rank={float(np.median(ranks)):.1f}"
    )

    csv_rows = []
    n_show = min(args.max_queries, len(dataset))
    for i in range(n_show):
        print("\n" + "=" * 88)
        print(f"query {i + 1}/{len(dataset)} | id={ids[i]} | true_rank={ranks[i]}")
        print(f"GT: {sentences[i]}")
        for rank, (j, score) in enumerate(zip(ranked.indices[i].tolist(), ranked.values[i].tolist()), start=1):
            mark = "*" if j == i else " "
            print(f"{mark} {rank:02d}. score={score:+.4f} id={ids[j]} :: {sentences[j]}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        for i in range(scores.shape[0]):
            for rank, (j, score) in enumerate(zip(ranked.indices[i].tolist(), ranked.values[i].tolist()), start=1):
                csv_rows.append({
                    "query_index": i,
                    "query_id": ids[i],
                    "ground_truth": sentences[i],
                    "true_rank": ranks[i],
                    "rank": rank,
                    "candidate_id": ids[j],
                    "candidate_sentence": sentences[j],
                    "score": score,
                    "is_match": int(j == i),
                })
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote retrieval rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
