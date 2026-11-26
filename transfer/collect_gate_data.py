"""Collect observation -> intervention labels from the verified door wrapper.

The wrapper is used only as a training-time teacher. The student gate is a learned network
trained on official observation features; no privileged signal is stored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from door_controller import DoorController
from keystone import FeatureBuilder, KeyStoneConfig, macro_action, legalize_action
from transfer.cnn_adapter import TransferKeyStoneNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="transfer/weights/gate_data.npz")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5000)
    args = ap.parse_args()

    torch.set_num_threads(1)
    ck = torch.load(ROOT / "transfer/weights/champion_frozen.pt", map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    model = TransferKeyStoneNet(cfg, ROOT / "transfer/weights/find_return_11.pth.tar")
    sd = model.state_dict()
    for k, v in ck["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v
    model.load_state_dict(sd); model.eval()
    fb = FeatureBuilder(cfg)

    positives = sorted((ROOT / "maps" / "train").glob("door_simple_*.wad"))
    negatives = sorted((ROOT / "maps" / "holdout").glob("eval_move_*.wad"))[:2] \
        + sorted((ROOT / "maps" / "holdout").glob("eval_nav_*.wad"))[:2] \
        + sorted((ROOT / "maps" / "holdout").glob("eval_combat_*.wad"))[:1] \
        + sorted((ROOT / "maps" / "holdout").glob("eval_key_*.wad"))[:1]

    X, Y = [], []
    for group, maps in (("pos", positives), ("neg", negatives)):
        for wad in maps:
            for ep in range(args.episodes):
                env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
                obs, _ = env.reset(seed=args.seed + ep * 17)
                state, ctrl = model.initial_state(1), DoorController()
                pe = np.zeros(16, np.float32); pa = np.zeros(9, np.int8)
                done = False
                while not done:
                    vec, ev, event = fb(obs, pe, pa)
                    parts = fb.split(torch.from_numpy(vec[None, :]))
                    parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
                    with torch.inference_mode():
                        out_s, state = model.step(parts, state, torch.tensor([event]))
                        idx = int(out_s["logits"][0].argmax())
                    chosen = ctrl.apply(idx, obs)
                    X.append(vec)
                    Y.append(1 if chosen != idx else 0)
                    a = legalize_action(macro_action(chosen), pa)
                    obs, _, term, trunc, _ = env.step(a)
                    pe, pa = ev, a
                    done = term or trunc
                env.close()
    out = ROOT / args.out
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.int64)
    np.savez_compressed(out, X=X, Y=Y)
    print(f"gate data: {len(X)} samples, positives={int(Y.sum())} -> {out}")


if __name__ == "__main__":
    main()
