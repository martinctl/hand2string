"""Evaluate the trained MLP: per-class precision/recall/F1.

Usage:
    python scripts/eval_mlp.py --ckpt runs/mlp_alphabet/best.pt \
                               --data data/asl_alphabet_landmarks.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import random_split, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP


def normalize_hand(X: np.ndarray) -> np.ndarray:
    X = X.reshape(-1, 21, 3).copy()
    X -= X[:, 0:1, :]
    scale = np.linalg.norm(X, axis=-1).max(axis=-1, keepdims=True)
    return (X / np.maximum(scale, 1e-6)[..., None]).reshape(-1, 63)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/mlp_alphabet/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/asl_alphabet_landmarks.npz"))
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    blob = np.load(args.data, allow_pickle=True)
    X = normalize_hand(blob["X"])
    y, classes = blob["y"], list(blob["classes"])

    full = TensorDataset(torch.from_numpy(X).float(),
                         torch.from_numpy(y).long())
    n_val = int(len(full) * args.val_frac)
    _, val_ds = random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MLP(ckpt["input_dim"], ckpt["hidden_dim"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    xs = torch.stack([x for x, _ in val_ds])
    ys = torch.tensor([t for _, t in val_ds])
    with torch.no_grad():
        preds = model(xs).argmax(-1).numpy()

    print(classification_report(ys.numpy(), preds, target_names=classes, digits=3))

    cm = confusion_matrix(ys.numpy(), preds, labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    print("Top-5 most confused pairs (true -> predicted):")
    for idx in np.argsort(cm, axis=None)[::-1][:5]:
        i, j = np.unravel_index(idx, cm.shape)
        if cm[i, j] == 0:
            break
        print(f"  {classes[i]} -> {classes[j]}  : {cm[i, j]} times")


if __name__ == "__main__":
    main()
