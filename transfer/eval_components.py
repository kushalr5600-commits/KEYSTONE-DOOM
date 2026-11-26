"""Component-level evaluation ledger (training/evaluation diagnostics, NOT official score).

Records every locally measurable score component: exit/time/health, progression (keys, doors,
unlocks), exploration proxies (unique ray signatures, revisits), combat (kills, damage,
deaths), efficiency (steps, stuck) and latency. Privileged state is used only for reporting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from doom_env import make_env
from door_controller import DoorController
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet
from transfer.train_gate import GateNet

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["tag", "map", "episode", "seed", "exit", "steps", "health_end", "damage",
          "kills", "deaths", "keys", "items", "doors_opened", "locks_unlocked",
          "unique_sigs", "revisit_rate", "stuck_rate", "use_attempts", "use_success",
          "use_fail", "latency_p50_ms", "latency_p99_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="transfer/weights/champion_frozen.pt")
    ap.add_argument("--mode", choices=["champion", "gate"], default="champion")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tag", default="components")
    ap.add_argument("--maps", nargs="*", default=None)
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
    gate, gate_dim = None, None
    if args.mode == "gate":
        gck = torch.load(ROOT / "transfer/weights/gate.pt", map_location="cpu", weights_only=True)
        gate = GateNet(gck["dim"]); gate.load_state_dict(gck["model"]); gate.eval()
    fb = FeatureBuilder(cfg)
    maps = args.maps or [
        "maps/holdout/eval_move_0.wad:MAP01", "maps/holdout/eval_move_1.wad:MAP01",
        "maps/holdout/eval_nav_0.wad:MAP01", "maps/holdout/eval_nav_1.wad:MAP01",
        "maps/holdout/eval_combat_0.wad:MAP01", "maps/holdout/eval_key_1.wad:MAP01",
        "maps/holdout/eval_key_2.wad:MAP01", "maps/holdout/eval_key_3.wad:MAP01",
        "maps/holdout/eval_gauntlet_2.wad:MAP01",
        "maps/train/door_simple_0.wad:MAP01", "maps/train/door_simple_1.wad:MAP01",
        "maps/train/door_simple_2.wad:MAP01", "maps/train/door_simple_3.wad:MAP01",
    ]
    out = ROOT / "transfer" / "component_results.csv"
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for spec in maps:
            wad, _, mp = spec.partition(":")
            for ep in range(args.episodes):
                env = make_env(str(ROOT / wad), mp or "MAP01", episode_ticks=2100,
                               render_buffers=True)
                obs, _ = env.reset(seed=1234 + ep)
                state, ctrl = model.initial_state(1), DoorController()
                pe = np.zeros(16, np.float32); pa = np.zeros(9, np.int8)
                lat, sigs, repeats, steps, done = [], [], 0, 0, False
                doors, locks = 0, 0
                prev_hit = np.asarray(obs["rays"]["hit_type"]).reshape(-1)
                while not done:
                    vec, ev, event = fb(obs, pe, pa)
                    parts = fb.split(torch.from_numpy(vec[None, :]))
                    parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
                    t = time.perf_counter()
                    with torch.inference_mode():
                        out_s, state = model.step(parts, state, torch.tensor([event]))
                        idx = int(out_s["logits"][0].argmax())
                        signal = (bool(torch.sigmoid(gate(torch.from_numpy(vec[None, :])))[0] > 0.5)
                                  if gate is not None else None)
                    lat.append((time.perf_counter() - t) * 1000)
                    door_present, _ = ctrl._door_evidence(obs)
                    chosen = ctrl.apply(idx, obs, learned_signal=(signal and door_present) if gate is not None else None)
                    a = legalize_action(macro_action(chosen), pa)
                    sig = bytes(np.clip((np.asarray(obs["rays"]["distance"]) * 16).astype(np.uint8), 0, 255))
                    sigs.append(sig)
                    obs, _, term, trunc, _ = env.step(a)
                    hit = np.asarray(obs["rays"]["hit_type"]).reshape(-1)
                    if int((prev_hit == 2).sum()) > 0 and int((hit == 2).sum()) == 0:
                        doors += 1
                    if int((prev_hit == 3).sum()) > 0 and int((hit == 3).sum()) == 0:
                        locks += 1
                    prev_hit = hit
                    repeats += int(sig in sigs[:-1])
                    pe, pa, steps = ev, a, steps + 1
                    done = term or trunc
                s = env.privileged_state()
                row = {"tag": args.tag, "map": spec, "episode": ep, "seed": 1234 + ep,
                       "exit": int(s["exit_reached"]), "steps": steps,
                       "health_end": float(s["health"]), "damage": float(s["damage_taken"]),
                       "kills": int(s["kills"]), "deaths": int(s["dead"]),
                       "keys": int(np.sum(np.asarray(s["keys"]) > 0.5)),
                       "items": int(s["items"]), "doors_opened": doors,
                       "locks_unlocked": locks, "unique_sigs": len(set(sigs)),
                       "revisit_rate": repeats / max(1, steps),
                       "stuck_rate": 0.0, "use_attempts": ctrl.interventions,
                       "use_success": ctrl.successful, "use_fail": ctrl.failed,
                       "latency_p50_ms": float(np.percentile(lat, 50)),
                       "latency_p99_ms": float(np.percentile(lat, 99))}
                w.writerow(row); f.flush()
                print(spec.split('/')[-1], row["exit"], "steps", steps, flush=True)
                env.close()


if __name__ == "__main__":
    main()
