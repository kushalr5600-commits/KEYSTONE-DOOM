"""Phase-1 diagnostic trace for a trained KEYSTONE checkpoint.

Records what the policy actually does (button rates, macro actions, turn efficiency,
USE outcomes, stuck/loop behaviour, death timing, distance-to-exit progress) and checks
the training-time reward/encoding pipeline. Uses privileged state ONLY for diagnosis,
never for the deployed agent.

    python -m training.diagnose --weights submission/weights/keystone.pt --episodes 8
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import math

import numpy as np
import torch

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, KeyStoneNet, legalize_action, macro_action
from training.exit_finder import find_exit

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="submission/weights/keystone.pt")
    ap.add_argument("--maps", nargs="+", default=[
        "maps/two_rooms.wad:MAP01",
        "tests/fixtures/509aca7cabb8cd7f.wad:MAP01",
        "tests/fixtures/5c688a052900e964.wad:MAP01",
    ])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.set_num_threads(1)
    ck = torch.load(ROOT / args.weights, map_location="cpu", weights_only=False)
    cfg = KeyStoneConfig(**ck.get("config", {}))
    print(f"checkpoint step={ck.get('step')} config novelty={cfg.reward_new_signature} "
          f"exit={cfg.reward_exit} potential={cfg.exit_potential} time={cfg.time_penalty}")
    print(f"shaping available? exit_potential={cfg.exit_potential > 0}")
    model = KeyStoneNet(cfg).eval()
    model.load_state_dict(ck["model"])
    fb = FeatureBuilder(cfg)

    buttons = Counter()
    macros = Counter()
    suppressed = Counter()
    use_events = Counter()
    episodes = []
    h_stds = []

    for spec in args.maps:
        wad, _, mp = spec.partition(":")
        mp = mp or "MAP01"
        ex = find_exit(ROOT / wad, mp)
        print(f"\nMAP {spec} exit={ex}")
        for ep in range(args.episodes):
            env = make_env(str(ROOT / wad), mp, episode_ticks=2100, render_buffers=False)
            obs, _ = env.reset(seed=args.seed + ep)
            state = model.initial_state(1)
            prev_ev = np.zeros(16, np.float32)
            prev_act = np.zeros(9, np.int8)
            ds, pos, hs, sigs = [], [], [], []
            logits_all = []
            death_step = None
            first_use_step = None
            steps = 0
            done = False
            while not done:
                vec, ev, event = fb(obs, prev_ev, prev_act)
                x = torch.from_numpy(vec[None, :])
                with torch.inference_mode():
                    out, state = model.step(fb.split(x), state, torch.tensor([event]))
                    logits_all.append(out["logits"][0].numpy().copy())
                if cfg.action_mode == "macro":
                    raw = macro_action(int(np.argmax(out["logits"][0].numpy())))
                else:
                    raw = (out["logits"][0].numpy() > 0).astype(np.int8)
                action = legalize_action(raw, prev_act)
                for i in range(9):
                    if int(action[i]):
                        buttons[i] += 1
                    if int(raw[i]) and not int(action[i]):
                        suppressed[i] += 1
                macros[int("".join(str(int(v)) for v in action), 2)] += 1
                if int(action[7]) and int(prev_act[7]) == 0 and first_use_step is None:
                    first_use_step = steps
                s_before = env.privileged_state()
                door_before = int(np.sum(np.asarray(obs["rays"]["hit_type"]) == 2))
                lock_before = int(np.sum(np.asarray(obs["rays"]["hit_type"]) == 3))
                obs, _, term, trunc, info = env.step(action)
                s_after = env.privileged_state()
                if int(action[7]) and int(prev_act[7]) == 0:
                    if int(np.sum(np.asarray(obs["rays"]["hit_type"]) == 2)) < door_before:
                        use_events["door_opened"] += 1
                    if int(np.sum(np.asarray(obs["rays"]["hit_type"]) == 3)) < lock_before:
                        use_events["lock_cleared"] += 1
                prev_ev, prev_act = ev, action.copy()
                sigs.append(bytes(np.clip((np.asarray(obs["rays"]["distance"]) * 16).astype(np.uint8), 0, 255)))
                d = math.hypot(s_after["x"] - ex[0], s_after["y"] - ex[1]) if ex else float("nan")
                ds.append(d)
                pos.append((s_after["x"], s_after["y"]))
                hs.append(state.h[0].detach().numpy().copy())
                if s_after["dead"] and death_step is None:
                    death_step = steps
                steps += 1
                done = term or trunc
            s = env.privileged_state()
            disp = sum(math.hypot(pos[i][0]-pos[i-1][0], pos[i][1]-pos[i-1][1])
                       for i in range(1, len(pos)))
            ep_rec = {
                "map": spec, "seed": args.seed + ep, "steps": steps,
                "exit": int(s["exit_reached"]), "dead": int(s["dead"]),
                "death_step": death_step, "d0": round(ds[0], 1) if ds else None,
                "dmin": round(min(ds), 1) if ds else None,
                "dend": round(ds[-1], 1) if ds else None,
                "displacement": round(disp, 1),
                "mean_speed": round(disp / max(1, steps), 2),
                "unique_ray_sigs": len(set(sigs)),
                "novelty_reward_est": round(cfg.reward_new_signature * len(set(sigs)), 3),
            }
            episodes.append(ep_rec)
            if hs:
                h_stds.append(float(np.std(np.stack(hs), axis=0).mean()))
            env.close()

    print("\n=== BUTTON COUNTS (name, presses, presses/step) ===")
    names = ["FWD", "BACK", "STRAFE_L", "STRAFE_R", "TURN_L", "TURN_R",
             "ATTACK", "USE", "NEXT_WEAPON"]
    total_steps = sum(e["steps"] for e in episodes)
    for i, n in enumerate(names):
        print(f"  {n:12s} {buttons[i]:6d}  {buttons[i]/max(1,total_steps):.3f}")
    print("=== SUPPRESSED BY LEGALIZATION ===")
    for i, n in enumerate(names):
        if suppressed[i]:
            print(f"  {n:12s} {suppressed[i]:6d}")
    print("=== TOP MACRO ACTIONS (binary -> count) ===")
    for code, cnt in macros.most_common(12):
        print(f"  {code:09b} {cnt:6d}  {[names[i] for i in range(9) if code & (1 << i)]}")
    print("=== USE OUTCOMES ===", dict(use_events))
    print("=== EPISODES ===")
    for e in episodes:
        print(" ", json.dumps(e))
    exits = sum(e["exit"] for e in episodes)
    deaths = sum(e["dead"] for e in episodes)
    dprog = np.nanmean([e["d0"] - e["dend"] for e in episodes if e["d0"] is not None])
    print(f"\nSUMMARY episodes={len(episodes)} exits={exits} deaths={deaths} "
          f"mean_distance_progress={dprog:.1f} hidden_activity={np.mean(h_stds):.3f}")


if __name__ == "__main__":
    main()
