"""Sequence-level behavior cloning with a recurrent KEYSTONE student.

Open-source audit found no usable external door-interaction dataset, so we train on our own
oracle demonstrations in temporal order. The recurrent core lets the student remember that
it just pressed USE and avoid reversing a moving door — the reactive champion's verified
Tier-2 failure mode. Champion weights are migrated for all compatible layers; only the GRU
is newly initialised. Frozen champion is never modified.
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

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="transfer/weights/demos_seq.npz")
    ap.add_argument("--base", default="transfer/weights/champion_frozen.pt")
    ap.add_argument("--pretrained", default="transfer/weights/find_return_11.pth.tar")
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="transfer/weights/seq_candidate.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device(args.device)
    data = np.load(ROOT / args.demos)
    X, RGB = data["X"], data["RGB"]
    E, A, I = data["E"], data["A"], data["I"]

    cfg = KeyStoneConfig.load(ROOT / args.config)
    cfg.use_recurrence = True
    cfg.tags = {"variant": "seq_recurrent"}
    fb = FeatureBuilder(cfg)

    model = TransferKeyStoneNet(cfg, ROOT / args.pretrained).to(device)
    base = torch.load(ROOT / args.base, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    copied = 0
    for k, v in base["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v.to(device)
            copied += 1
    model.load_state_dict(sd)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print(f"device={device} copied={copied}/{len(sd)} tensors; "
          f"episodes={len(np.unique(I))} transitions={len(X)}", flush=True)

    episodes = []
    for ep in np.unique(I):
        idx = np.flatnonzero(I == ep)
        episodes.append((idx, len(idx)))

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        loss_sum, n_tok = 0.0, 0
        order = np.random.default_rng(epoch).permutation(len(episodes))
        for ei in order:
            idx, T = episodes[ei]
            for start in range(0, T, args.chunk):
                sub = idx[start:start + args.chunk]
                L = len(sub)
                feats = torch.from_numpy(X[sub]).to(device)                       # (L,F)
                parts = fb.split(feats)                                            # (L,...)
                parts = {k: v.unsqueeze(1).contiguous() for k, v in parts.items()} # (L,1,...)
                parts["rgb"] = torch.from_numpy(RGB[sub]).to(device).unsqueeze(1)  # (L,1,64,112,3)
                events = torch.from_numpy(E[sub]).long().to(device).unsqueeze(1)   # (L,1)
                actions = torch.from_numpy(A[sub]).long().to(device).unsqueeze(1)  # (L,1)
                state = model.initial_state(1, device=device)
                out, _ = model.forward_sequence(parts, state, events)
                loss = F.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]),
                                       actions.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                loss_sum += float(loss.detach()) * L
                n_tok += L
        print(f"epoch {epoch+1}/{args.epochs} loss={loss_sum/n_tok:.4f} "
              f"tokens/s={n_tok/(time.perf_counter()-t0):.0f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": json.loads(cfg.to_json()), "step": 0,
                "base": str(args.base), "demos": str(args.demos)}, out)
    cfg.save(out.parent / "config.json")
    print("wrote", out)


if __name__ == "__main__":
    main()
