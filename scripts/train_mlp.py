"""Train the MLP baseline on extracted landmark vectors.

Input: a .npz produced by scripts/extract_alphabet_landmarks.py with keys
    X (N, D), y (N,), classes (C,).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.mlp import MLP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/asl_alphabet_landmarks.npz"))
    p.add_argument("--out", type=Path, default=Path("runs/mlp_alphabet"))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def normalize_hand(X: np.ndarray) -> np.ndarray:
    """Translate so wrist (kp 0) is at origin, then scale by max distance.
    This makes the MLP invariant to hand position and image size.
    """
    X = X.reshape(-1, 21, 3).copy()
    X -= X[:, 0:1, :]
    scale = np.linalg.norm(X, axis=-1).max(axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)[..., None]
    X /= scale
    return X.reshape(-1, 63)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.data}")
    blob = np.load(args.data, allow_pickle=True)
    X, y, classes = blob["X"], blob["y"], blob["classes"]
    X = normalize_hand(X)
    num_classes = len(classes)
    print(f"  X: {X.shape}  y: {y.shape}  classes: {num_classes}")

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    full = TensorDataset(X_t, y_t)
    n_val = int(len(full) * args.val_frac)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    device = args.device
    model = MLP(input_dim=X.shape[1], hidden_dim=args.hidden_dim,
                num_classes=num_classes, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(-1) == yb).sum().item()
            tr_total += xb.size(0)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_total += xb.size(0)

        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total
        history.append((epoch, tr_loss / tr_total, tr_acc, val_acc))
        print(f"epoch {epoch:3d}  loss {tr_loss/tr_total:.4f}  "
              f"train_acc {tr_acc:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes.tolist(),
                "input_dim": X.shape[1],
                "hidden_dim": args.hidden_dim,
            }, args.out / "best.pt")

    np.savez(args.out / "history.npz",
             history=np.array(history, dtype=np.float32))
    print(f"Best val_acc: {best_val:.4f}  -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
