"""Held-out evaluation of the isolated visual-transfer checkpoint.

Uses exactly the same maps, seeds, action legality and metrics as training/evaluate.py,
but instantiates TransferKeyStoneNet (additive pretrained conv features). Write results to a
transfer-local CSV so the main experiment ledger is untouched.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet

ROOT = Path(__file__).resolve().parents[1]
MAPS = [
    "maps/holdout/eval_move_0.wad:MAP01", "maps/holdout/eval_move_1.wad:MAP01",
    "maps/holdout/eval_nav_0.wad:MAP01", "maps/holdout/eval_nav_1.wad:MAP01",
    "maps/holdout/eval_combat_0.wad:MAP01", "maps/holdout/eval_key_1.wad:MAP01",
    "maps/holdout/eval_key_2.wad:MAP01", "maps/holdout/eval_key_3.wad:MAP01",
]
FIELDS = ["tag", "map", "episode", "seed", "exit", "keys", "kills", "damage",
          "deaths", "steps", "unique_sigs", "stuck_rate", "latency_p50_ms", "latency_p99_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="transfer/weights/transfer_bc.pt")
    ap.add_argument("--pretrained", default="transfer/weights/find_return_11.pth.tar")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tag", default="transfer_bc")
    ap.add_argument("--maps", nargs="*", default=None)
    args = ap.parse_args()
    maps = args.maps or MAPS

    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    model = TransferKeyStoneNet(cfg, ROOT / args.pretrained)
    sd = model.state_dict()
    for k, v in ck["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v
    model.load_state_dict(sd)
    model.eval()
    fb = FeatureBuilder(cfg)
    out = ROOT / "transfer" / "transfer_results.csv"
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for spec in maps:
            wad, _, mp = spec.partition(":")
            for ep in range(args.episodes):
                env = make_env(str(ROOT / wad), mp or "MAP01", episode_ticks=2100,
                               render_buffers=True)
                obs, _ = env.reset(seed=1234 + ep)
                state = model.initial_state(1)
                pe = np.zeros(16, np.float32)
                pa = np.zeros(9, np.int8)
                lat, sigs, rep = [], set(), 0
                steps, done = 0, False
                while not done:
                    vec, ev, event = fb(obs, pe, pa)
                    parts = fb.split(torch.from_numpy(vec[None, :]))
                    parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
                    t = time.perf_counter()
                    with torch.inference_mode():
                        out_s, state = model.step(parts, state, torch.tensor([event]))
                        logits = out_s["logits"][0].numpy()
                    lat.append((time.perf_counter() - t) * 1000)
                    chosen = macro_action(int(np.argmax(logits)))
                    a = legalize_action(chosen, pa)
                    rep += int(np.array_equal(a, pa))
                    sigs.add(bytes(np.clip((np.asarray(obs["rays"]["distance"]) * 16).astype(np.uint8), 0, 255)))
                    obs, _, term, trunc, _ = env.step(a)
                    pe, pa, steps = ev, a, steps + 1
                    done = term or trunc
                s = env.privileged_state()
                row = {"tag": args.tag, "map": spec, "episode": ep, "seed": 1234 + ep,
                       "exit": int(s["exit_reached"]), "keys": int(np.sum(np.asarray(s["keys"]) > 0.5)),
                       "kills": int(s["kills"]), "damage": float(s["damage_taken"]),
                       "deaths": int(s["dead"]), "steps": steps, "unique_sigs": len(sigs),
                       "stuck_rate": rep / max(1, steps),
                       "latency_p50_ms": float(np.percentile(lat, 50)),
                       "latency_p99_ms": float(np.percentile(lat, 99))}
                w.writerow(row)
                f.flush()
                print(row, flush=True)
                env.close()


if __name__ == "__main__":
    main()
