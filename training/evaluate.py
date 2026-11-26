"""Deterministic held-out evaluation for KEYSTONE-DOOM.

Reports objective metrics from the wrapper's scorer-side privileged state (never given to
the agent) plus inference latency. Results are appended to experiments/results.csv.

    python -m training.evaluate --weights submission/weights/keystone.pt \
        --maps maps/two_rooms.wad:MAP01 --episodes 6
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, KeyStoneNet, legalize_action, macro_action

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.csv"
FIELDS = [
    "tag", "weights", "map", "episode", "seed", "exit", "death", "timeout",
    "keys", "kills", "damage", "health_end", "steps", "ticks", "unique_sigs",
    "stuck_rate", "latency_mean_ms", "latency_p50_ms", "latency_p99_ms",
    "latency_max_ms", "memory_slots_used", "final_goal", "mean_confidence",
]


def run_episode(model, cfg, fb, wad, map_name, seed, ticks, device):
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    state = model.initial_state(1, device=device)
    prev_ev = np.zeros(16, np.float32)
    prev_act = np.zeros(9, np.int8)
    lat, sigs, repeated = [], set(), 0
    goals, confs, used = [], [], []
    steps = 0
    done = False
    t0 = time.perf_counter()
    while not done:
        vec, ev, event = fb(obs, prev_ev, prev_act)
        x = torch.from_numpy(vec[None, :]).to(device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            out, state = model.step(
                fb.split(x), state,
                torch.tensor([event], device=device, dtype=torch.long))
            logits = out["logits"][0].cpu().numpy()
            goals.append(int(out["goal_logits"][0].argmax().item()))
            confs.append(float(out["risk"][0].mean().item()))
            used.append(int((out["read_weights"][0] > 1e-2).sum().item()))
        lat.append((time.perf_counter() - t1) * 1000.0)
        if cfg.action_mode == "macro":
            chosen = macro_action(int(np.argmax(logits)))
        else:
            chosen = (logits > 0).astype(np.int8)
        action = legalize_action(chosen, prev_act)
        if np.array_equal(action, prev_act):
            repeated += 1
        sig = bytes(np.clip((obs["rays"]["distance"] * 16).astype(np.uint8), 0, 255))
        sigs.add(sig)
        obs, reward, term, trunc, info = env.step(action)
        prev_ev, prev_act = ev, action.copy()
        steps += 1
        done = term or trunc
    priv = env.privileged_state()
    lat = np.asarray(lat)
    row = {
        "exit": int(bool(priv["exit_reached"])),
        "death": int(bool(priv["dead"])),
        "timeout": int(bool(priv["timeout"])),
        "keys": int(np.sum(np.asarray(priv["keys"]) > 0.5)),
        "kills": int(priv["kills"]),
        "damage": float(priv["damage_taken"]),
        "health_end": float(priv["health"]),
        "steps": steps,
        "ticks": steps * 4,
        "unique_sigs": len(sigs),
        "stuck_rate": repeated / max(1, steps),
        "latency_mean_ms": float(lat.mean()),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p99_ms": float(np.percentile(lat, 99)),
        "latency_max_ms": float(lat.max()),
        "memory_slots_used": float(np.mean(used)),
        "final_goal": int(goals[-1]) if goals else -1,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "wall_s": time.perf_counter() - t0,
    }
    env.close()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="submission/weights/keystone.pt")
    ap.add_argument("--maps", nargs="+", default=[
        "maps/holdout/eval_move_0.wad:MAP01",
        "maps/holdout/eval_move_1.wad:MAP01",
        "maps/holdout/eval_nav_0.wad:MAP01",
        "maps/holdout/eval_nav_1.wad:MAP01",
        "maps/holdout/eval_combat_0.wad:MAP01",
        "maps/holdout/eval_key_1.wad:MAP01",
    ])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeyStoneNet(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    fb = FeatureBuilder(cfg)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["wall_s"])
        if new_file:
            writer.writeheader()
        for spec in args.maps:
            wad, _, mp = spec.partition(":")
            mp = mp or "MAP01"
            for i in range(args.episodes):
                seed = args.seed + i
                row = run_episode(model, cfg, fb, str(ROOT / wad), mp, seed, args.ticks, device)
                writer.writerow({
                    **{k: row.get(k, "") for k in FIELDS},
                    "tag": args.tag, "weights": args.weights, "map": spec,
                    "episode": i, "seed": seed, "wall_s": round(row["wall_s"], 2),
                })
                f.flush()
                print(json.dumps({k: row[k] for k in
                                  ("exit", "keys", "kills", "steps", "unique_sigs",
                                   "latency_p50_ms", "latency_p99_ms")}), flush=True)


if __name__ == "__main__":
    main()
