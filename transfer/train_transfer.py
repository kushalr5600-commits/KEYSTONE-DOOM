"""Behavior-clone the visual-transfer adapter on oracle demonstrations.

Starts from the protected baseline2 weights (copied read-only into transfer/weights/),
freezes the MIT-licensed pretrained conv stem, and trains only the new visual projection
plus the existing policy for a few epochs. Nothing in the main tree is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from keystone import FeatureBuilder, KeyStoneConfig
from transfer.cnn_adapter import TransferKeyStoneNet
from training.device import add_device_arg, describe, get_device

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="transfer/weights/demos_rgb.npz")
    ap.add_argument("--base", default="transfer/weights/baseline2.pt")
    ap.add_argument("--pretrained", default="transfer/weights/find_return_11.pth.tar")
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="transfer/weights/transfer_bc.pt")
    add_device_arg(ap)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {describe(device)}", flush=True)
    torch.set_num_threads(4)
    cfg = KeyStoneConfig.load(ROOT / args.config)
    fb = FeatureBuilder(cfg)
    data = np.load(ROOT / args.demos)
    X = torch.from_numpy(data["X"]).to(device, non_blocking=True)
    RGB = torch.from_numpy(data["RGB"]).to(device, non_blocking=True)  # (N,64,112,3)
    A = torch.from_numpy(data["A"]).long().to(device, non_blocking=True)

    model = TransferKeyStoneNet(cfg, ROOT / args.pretrained).to(device)
    base = torch.load(ROOT / args.base, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    for k, v in base["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v
    model.load_state_dict(sd)
    for p in model.pretrained_conv.parameters():
        p.requires_grad_(False)
    model.pretrained_conv.eval()

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(0)
    n = X.shape[0]
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        loss_sum, correct = 0.0, 0
        for start in range(0, n, 256):
            idx = torch.from_numpy(perm[start:start + 256]).to(device, non_blocking=True)
            parts = fb.split(X[idx])
            parts["rgb"] = RGB[idx]
            out, _ = model.step(parts, model.initial_state(len(idx), device=device),
                                torch.zeros(len(idx), dtype=torch.long, device=device))
            loss = F.cross_entropy(out["logits"], A[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((out["logits"].argmax(-1) == A[idx]).sum())
        dt = time.perf_counter() - t0
        print(f"epoch {epoch+1}/{args.epochs} loss={loss_sum/n:.4f} acc={correct/n:.3f} "
              f"samples/s={n/dt:.0f} seconds={dt:.1f}", flush=True)

    out = ROOT / args.out
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": json.loads(cfg.to_json()),
                "step": 0, "transfer_from": str(args.pretrained),
                "base_checkpoint": str(args.base)}, out)
    cfg.save(out.parent / "config.json")
    print("wrote", out)


if __name__ == "__main__":
    main()
