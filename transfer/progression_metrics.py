"""Phase-specific progression metrics on independently generated unseen key maps.

All event labels come from the observation-derived FeatureBuilder, not privileged state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet

ROOT = Path(__file__).resolve().parents[1]
MAPS = ["eval_key_1", "eval_key_2", "eval_key_3"]
FIELDS = ["map", "seed", "exit", "keys", "locked_frames", "use_presses",
          "interact_ok", "interact_fail", "useless_use", "steps", "stuck_rate", "damage", "deaths"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="transfer/weights/champion_frozen.pt")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="transfer/progression_metrics.csv")
    args = ap.parse_args()
    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    model = TransferKeyStoneNet(cfg, ROOT / "transfer/weights/find_return_11.pth.tar")
    sd = model.state_dict()
    for k, v in ck["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v
    model.load_state_dict(sd); model.eval()
    fb = FeatureBuilder(cfg)
    out = ROOT / args.out
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for name in MAPS:
            for ep in range(args.episodes):
                env = make_env(str(ROOT / "maps" / "holdout" / f"{name}.wad"), "MAP01",
                               episode_ticks=2100, render_buffers=True)
                obs, _ = env.reset(seed=1234 + ep)
                state = model.initial_state(1)
                pe = np.zeros(16, np.float32); pa = np.zeros(9, np.int8)
                c = Counter(); repeats = 0; steps = 0; done = False
                while not done:
                    vec, ev, event = fb(obs, pe, pa)
                    parts = fb.split(torch.from_numpy(vec[None, :]))
                    parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
                    with torch.inference_mode():
                        out_s, state = model.step(parts, state, torch.tensor([event]))
                        logits = out_s["logits"][0].numpy()
                    a = legalize_action(macro_action(int(np.argmax(logits))), pa)
                    repeats += int(np.array_equal(a, pa))
                    c[event] += 1
                    if a[7] and not pa[7]:
                        c["use"] += 1
                    obs, _, term, trunc, _ = env.step(a)
                    pe, pa, steps = ev, a, steps + 1
                    done = term or trunc
                s = env.privileged_state()
                row = {"map": name, "seed": 1234 + ep, "exit": int(s["exit_reached"]),
                       "keys": int(np.sum(np.asarray(s["keys"]) > 0.5)),
                       "locked_frames": c[1], "use_presses": c["use"],
                       "interact_ok": c[6], "interact_fail": c[7],
                       "useless_use": max(0, c["use"] - c[6]),
                       "steps": steps, "stuck_rate": repeats / max(1, steps),
                       "damage": float(s["damage_taken"]), "deaths": int(s["dead"])}
                w.writerow(row); f.flush()
                print(row, flush=True)
                env.close()


if __name__ == "__main__":
    main()
