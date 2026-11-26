"""Evaluate frozen champion + learned intervention gate on held-out maps/seeds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from door_controller import DoorController
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet
from transfer.train_gate import GateNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--tag", default="gate")
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--require-door", action="store_true")
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
    gate_ck = torch.load(ROOT / "transfer/weights/gate.pt", map_location="cpu", weights_only=True)
    fb = FeatureBuilder(cfg)
    gate = GateNet(gate_ck["dim"])
    gate.load_state_dict(gate_ck["model"]); gate.eval()

    maps = args.maps or [
        "maps/holdout/eval_move_0.wad:MAP01", "maps/holdout/eval_move_1.wad:MAP01",
        "maps/holdout/eval_nav_0.wad:MAP01", "maps/holdout/eval_nav_1.wad:MAP01",
        "maps/holdout/eval_combat_0.wad:MAP01", "maps/holdout/eval_key_1.wad:MAP01",
        "maps/holdout/eval_key_2.wad:MAP01", "maps/holdout/eval_key_3.wad:MAP01",
    ]
    out = ROOT / "transfer" / "gate_results.csv"
    fields = ["tag", "map", "episode", "seed", "exit", "keys", "kills", "damage",
              "deaths", "steps", "stuck_rate", "use_attempts", "use_success", "use_fail"]
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new: w.writeheader()
        for spec in maps:
            wad, _, mp = spec.partition(":")
            for ep in range(args.episodes):
                env = make_env(str(ROOT / wad), mp or "MAP01", episode_ticks=2100,
                               render_buffers=True)
                obs, _ = env.reset(seed=1234 + ep)
                state, ctrl = model.initial_state(1), DoorController()
                pe = np.zeros(16, np.float32); pa = np.zeros(9, np.int8)
                steps, done, rep = 0, False, 0
                while not done:
                    vec, ev, event = fb(obs, pe, pa)
                    parts = fb.split(torch.from_numpy(vec[None, :]))
                    parts["rgb"] = torch.from_numpy(obs["rgb"])[None, :]
                    with torch.inference_mode():
                        out_s, state = model.step(parts, state, torch.tensor([event]))
                        idx = int(out_s["logits"][0].argmax())
                        signal = bool(torch.sigmoid(gate(torch.from_numpy(vec[None, :])))[0] > args.threshold)
                    if args.require_door:
                        door_present, _ = ctrl._door_evidence(obs)
                        signal = signal and door_present
                    chosen = ctrl.apply(idx, obs, learned_signal=signal)
                    a = legalize_action(macro_action(chosen), pa)
                    rep += int(np.array_equal(a, pa))
                    obs, _, term, trunc, _ = env.step(a)
                    pe, pa, steps = ev, a, steps + 1
                    done = term or trunc
                s = env.privileged_state()
                row = {"tag": args.tag, "map": spec, "episode": ep, "seed": 1234 + ep,
                       "exit": int(s["exit_reached"]),
                       "keys": int(np.sum(np.asarray(s["keys"]) > 0.5)),
                       "kills": int(s["kills"]), "damage": float(s["damage_taken"]),
                       "deaths": int(s["dead"]), "steps": steps,
                       "stuck_rate": rep / max(1, steps),
                       "use_attempts": ctrl.interventions, "use_success": ctrl.successful,
                       "use_fail": ctrl.failed}
                w.writerow(row); f.flush()
                print(row, flush=True)
                env.close()


if __name__ == "__main__":
    main()
