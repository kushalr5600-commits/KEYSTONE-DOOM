"""Behavior cloning from training-only oracle demonstrations.

The teacher is a privileged oracle used only to generate (official observation, macro
action) pairs on verified training maps. The student sees exactly the deployed observation
interface and produces the same categorical action head. No privileged field is stored in
the dataset, and the teacher is never used at inference.

    python -m training.bc --demos experiments/demos.npz --epochs 8 \
        --out experiments/bc_init/keystone.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from keystone import KeyStoneConfig, KeyStoneNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="experiments/demos.npz")
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None, help="optional compatible checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    data = np.load(ROOT / args.demos)
    X = torch.from_numpy(data["X"])
    A = torch.from_numpy(data["A"]).long()
    cfg = KeyStoneConfig.load(ROOT / args.config)
    assert cfg.action_mode == "macro", "BC requires the macro action head"
    model = KeyStoneNet(cfg)
    if args.init:
        ck = torch.load(ROOT / args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print("initialized from", args.init)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    n = X.shape[0]
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        loss_sum = 0.0
        correct = 0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            x = X[idx]
            fb = __import__("keystone").FeatureBuilder(cfg).split(x)
            out, _ = model.step(fb, model.initial_state(len(idx)), torch.zeros(len(idx), dtype=torch.long))
            loss = F.cross_entropy(out["logits"], A[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((out["logits"].argmax(-1) == A[idx]).sum())
        print(f"epoch {epoch+1}/{args.epochs} loss={loss_sum/n:.4f} acc={correct/n:.3f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": json.loads(cfg.to_json()),
                "step": 0, "bc_from": str(args.demos)}, out)
    cfg.save(out.parent / "config.json")
    print("wrote", out)


if __name__ == "__main__":
    main()
