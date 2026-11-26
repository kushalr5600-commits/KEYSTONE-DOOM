"""Generate transfer-specific demonstrations that include RGB frames.

Isolated copy of the training-only oracle demo loop; stores the official observation's RGB
buffer alongside the feature vector so the pretrained visual encoder can be trained. The
teacher uses privileged map data only for labels; the student input remains official.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from doom_env import make_env
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action
from training.demos import nearest_macro, oracle_action
from training.exit_finder import find_exit
from training.verify_maps import find_doors, find_keys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="transfer/weights/demos_rgb.npz")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--patterns", nargs="*", default=None,
                    help="optional WAD glob patterns, e.g. key_[12]; default all curriculum maps")
    args = ap.parse_args()

    cfg = KeyStoneConfig.load(ROOT / args.config)
    fb = FeatureBuilder(cfg)
    patterns = args.patterns or ["move_*", "nav_*", "combat_*", "key_[12]"]
    maps = []
    for pat in patterns:
        maps.extend(sorted((ROOT / "maps" / "train").glob(f"{pat}.wad")))
    maps = list(dict.fromkeys(maps))
    print("demo maps:", [p.name for p in maps], flush=True)
    X, RGB, E, A = [], [], [], []
    episodes, exits = 0, 0
    for wad in maps:
        ex, keys, doors = find_exit(wad), find_keys(wad), find_doors(wad)
        for k in range(args.episodes):
            env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=True)
            obs, _ = env.reset(seed=9000 + episodes * 31 + k)
            prev_ev = np.zeros(16, np.float32)
            prev_act = np.zeros(9, np.int8)
            cd = 0
            for step in range(525):
                st = env.privileged_state()
                bits, target = oracle_action(obs, st, ex, keys, doors)
                hit = np.asarray(obs["rays"]["hit_type"])[:8]
                if cd > 0:
                    cd -= 1
                    bits[7] = 0
                elif np.any((hit == 2) | (hit == 3)) and target is not ex:
                    bits[7] = 1
                    cd = 40
                bits = legalize_action(bits, prev_act)
                vec, ev, event = fb(obs, prev_ev, prev_act)
                X.append(vec)
                RGB.append(cv2.resize(obs["rgb"], (112, 64), interpolation=cv2.INTER_AREA))
                E.append(event)
                A.append(nearest_macro(bits))
                obs, _, term, trunc, _ = env.step(bits)
                prev_ev, prev_act = ev, bits.copy()
                if term or trunc:
                    exits += int(bool(env.privileged_state()["exit_reached"]))
                    break
            episodes += 1
            env.close()
    out = ROOT / args.out
    np.savez_compressed(out, X=np.asarray(X, np.float32), RGB=np.asarray(RGB, np.uint8),
                        E=np.asarray(E, np.int64), A=np.asarray(A, np.int64))
    print(f"{episodes} episodes, {exits} exits, {len(X)} transitions -> {out}")


if __name__ == "__main__":
    main()
