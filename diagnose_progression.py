"""Training-only diagnostic trace of key-door→exit failure.

Uses privileged state ONLY for offline diagnosis; nothing here is in the deployed agent.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet

ROOT = __import__("pathlib").Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="maps/holdout/eval_key_1.wad:MAP01")
    ap.add_argument("--weights", default="transfer/weights/transfer_bc.pt")
    ap.add_argument("--pretrained", default="transfer/weights/find_return_11.pth.tar")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    model = TransferKeyStoneNet(cfg, ROOT / args.pretrained)
    sd = model.state_dict()
    for k, v in ck["model"].items():
        if k in sd and tuple(v.shape) == tuple(sd[k].shape):
            sd[k] = v
    model.load_state_dict(sd); model.eval()
    fb = FeatureBuilder(cfg)
    wad, _, mp = args.map.partition(":")

    for ep in range(args.episodes):
        env = make_env(str(ROOT / wad), mp or "MAP01", episode_ticks=2100, render_buffers=True)
        obs, _ = env.reset(seed=1234 + ep)
        state = model.initial_state(1)
        pe = np.zeros(16, np.float32); pa = np.zeros(9, np.int8)
        events, macros, hits, keys_seen = Counter(), Counter(), Counter(), []
        used = 0; goal_counts = Counter(); confs = []
        first_key = None; first_lock = None; first_use = None
        for step in range(525):
            vec, ev, event = fb(obs, pe, pa)
            parts = fb.split(torch.from_numpy(vec[None, :]))
            parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
            with torch.inference_mode():
                out, state = model.step(parts, state, torch.tensor([event]))
                logits = out["logits"][0].numpy()
                goal_counts[int(out["goal_logits"][0].argmax())] += 1
                confs.append(float(out["risk"][0].mean()))
            action = legalize_action(macro_action(int(np.argmax(logits))), pa)
            events[event] += 1; macros[int(np.argmax(logits))] += 1
            if action[7] and not pa[7]:
                used += 1
                if first_use is None: first_use = step
            ht = np.asarray(obs["rays"]["hit_type"])
            hits["locked" if np.any(ht == 3) else "door" if np.any(ht == 2) else "none"] += 1
            if np.any(ht == 3) and first_lock is None: first_lock = step
            priv = env.privileged_state()
            nkeys = int(np.sum(np.asarray(priv["keys"]) > 0.5))
            if nkeys and first_key is None: first_key = step
            keys_seen.append(nkeys)
            obs, _, term, trunc, _ = env.step(action)
            pe, pa = ev, action
            if term or trunc: break
        s = env.privileged_state()
        print(f"EP {ep}: exit={s['exit_reached']} dead={s['dead']} steps={step+1} "
              f"pos=({s['x']:.0f},{s['y']:.0f}) keys={int(np.sum(np.asarray(s['keys'])>0.5))} "
              f"first_key={first_key} first_lock_ray={first_lock} first_USE={first_use} USE_total={used}")
        print("  events", dict(events), "macros", dict(macros.most_common(6)), "rays", dict(hits))
        print("  goals", dict(goal_counts), "mean_conf", round(float(np.mean(confs)), 3))
        env.close()


if __name__ == "__main__":
    main()
