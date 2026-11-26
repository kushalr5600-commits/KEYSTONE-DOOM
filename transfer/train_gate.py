"""Train a compact learned intervention gate from wrapper trajectories.

The gate is a learned binary classifier over the official observation vector. It replaces the
wrapper's hand-written *decision* to intervene; the remaining wait/retry logic is a bounded
mechanical constraint, documented in DOOR_WRAPPER_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class GateNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="transfer/weights/gate_data.npz")
    ap.add_argument("--out", default="transfer/weights/gate.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(ROOT / args.data)
    X = torch.from_numpy(d["X"])
    Y = torch.from_numpy(d["Y"]).float()
    n, dim = X.shape
    model = GateNet(dim)
    pos = max(1, int(Y.sum()))
    pw = torch.tensor([(n - pos) / pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        total = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            logits = model(X[idx])
            loss = loss_fn(logits, Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"epoch {epoch+1}/{args.epochs} loss={total/n:.4f}", flush=True)

    with torch.no_grad():
        prob = torch.sigmoid(model(X))
        pred = prob > 0.5
        tp = int(((pred == 1) & (Y == 1)).sum())
        fp = int(((pred == 1) & (Y == 0)).sum())
        fn = int(((pred == 0) & (Y == 1)).sum())
    print(f"train precision={tp/max(1,tp+fp):.3f} recall={tp/max(1,tp+fn):.3f} "
          f"fp={fp} fn={fn}", flush=True)
    out = ROOT / args.out
    torch.save({"model": model.state_dict(), "dim": dim}, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
