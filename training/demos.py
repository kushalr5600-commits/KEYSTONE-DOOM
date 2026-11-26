"""Generate training-only demonstrations from a privileged oracle teacher.

The teacher may use map/position information because the official rules permit arbitrary
training-time scaffolding (docs/RULES.md §2, docs/CONTRACT.md §7). It is NOT used at
inference and is NOT part of the submission. Only (official observation features, macro
action) pairs are stored, so a student can never receive privileged inputs.

    python -m training.demos --out experiments/demos.npz --episodes 12
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from doom_env import make_env
from keystone import MACRO_ACTIONS, FeatureBuilder, KeyStoneConfig, legalize_action
from training.exit_finder import find_exit
from training.verify_maps import find_doors, find_keys

ROOT = Path(__file__).resolve().parents[1]


def nearest_macro(bits: np.ndarray) -> int:
    """Map an oracle 9-bit action to the closest legal macro by bit overlap."""
    b = np.asarray(bits, dtype=np.int8).ravel()
    best, score = 0, -1
    for i, m in enumerate(MACRO_ACTIONS):
        s = int(np.dot(b, m))
        if s > score:
            best, score = i, s
    return best


def oracle_action(obs, st, ex, keys, doors):
    target = ex
    if keys and not np.any(np.asarray(st["keys"]) > 0.5):
        target = keys[0]
    elif doors and ex is not None and st["x"] < max(d[0] for d in doors) + 20:
        target = min(doors, key=lambda p: (p[0]-st["x"])**2 + (p[1]-st["y"])**2)
    a = np.zeros(9, np.int8)
    if target is not None:
        desired = math.degrees(math.atan2(target[1]-st["y"], target[0]-st["x"]))
        err = (desired - st["angle"] + 180) % 360 - 180
        if abs(err) > 8:
            a[4 if err > 0 else 5] = 1
        else:
            a[0] = 1
    hit = np.asarray(obs["rays"]["hit_type"])
    if np.any(hit == 4):
        a[6] = 1  # attack visible enemies (observation-only decision)
    return a, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/demos.npz")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--seed", type=int, default=9000)
    args = ap.parse_args()

    cfg = KeyStoneConfig.load(ROOT / args.config)
    fb = FeatureBuilder(cfg)
    maps = sorted((ROOT / "maps" / "train").glob("move_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("nav_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("combat_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("key_[12].wad"))

    X, E, A = [], [], []
    episodes, successes = 0, 0
    for wad in maps:
        ex, keys, doors = find_exit(wad), find_keys(wad), find_doors(wad)
        for k in range(args.episodes):
            env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
            obs, _ = env.reset(seed=args.seed + episodes * 31 + k)
            prev_ev = np.zeros(16, np.float32)
            prev_act = np.zeros(9, np.int8)
            use_cd = 0
            for step in range(525):
                st = env.privileged_state()
                bits, target = oracle_action(obs, st, ex, keys, doors)
                hit = np.asarray(obs["rays"]["hit_type"])[:8]
                if use_cd > 0:
                    use_cd -= 1
                    bits[7] = 0
                elif np.any((hit == 2) | (hit == 3)) and target is not ex:
                    bits[7] = 1
                    use_cd = 40
                bits = legalize_action(bits, prev_act)
                vec, ev, event = fb(obs, prev_ev, prev_act)
                X.append(vec)
                E.append(event)
                A.append(nearest_macro(bits))
                obs, _, term, trunc, _ = env.step(bits)
                prev_ev, prev_act = ev, bits.copy()
                if term or trunc:
                    successes += int(bool(env.privileged_state()["exit_reached"]))
                    break
            episodes += 1
            env.close()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.asarray(X, np.float32), E=np.asarray(E, np.int64),
                        A=np.asarray(A, np.int64))
    print(f"demos: {episodes} episodes, {successes} exits, {len(X)} transitions -> {out}")


if __name__ == "__main__":
    main()
