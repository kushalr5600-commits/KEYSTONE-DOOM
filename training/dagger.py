"""DAgger-style aggregation: label the states the student actually visits.

The privileged oracle teacher is used ONLY to label training observations gathered from
student rollouts on training maps. The student never receives privileged inputs and the
teacher is never deployed. This targets the covariate-shift failure of plain behavior
cloning (student wanders off the teacher's trajectory and has no correction there).

    python -m training.dagger --student experiments/bc_init/keystone.pt \
        --demos experiments/demos.npz --out experiments/demos_dagger.npz --episodes 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from doom_env import make_env
from keystone import (
    FeatureBuilder, KeyStoneConfig, KeyStoneNet, legalize_action, macro_action,
)
from training.demos import nearest_macro, oracle_action
from training.exit_finder import find_exit
from training.verify_maps import find_doors, find_keys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--demos", default="experiments/demos.npz")
    ap.add_argument("--config", default="configs/baseline_macro.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17000)
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = KeyStoneConfig.load(ROOT / args.config)
    fb = FeatureBuilder(cfg)
    ck = torch.load(ROOT / args.student, map_location="cpu", weights_only=False)
    model = KeyStoneNet(cfg).eval()
    model.load_state_dict(ck["model"])

    data = np.load(ROOT / args.demos)
    X, E, A = [data["X"]], [data["E"]], [data["A"]]
    maps = sorted((ROOT / "maps" / "train").glob("move_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("nav_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("combat_*.wad")) \
        + sorted((ROOT / "maps" / "train").glob("key_[12].wad"))

    added = 0
    for mi, wad in enumerate(maps):
        ex, keys, doors = find_exit(wad), find_keys(wad), find_doors(wad)
        for ep in range(args.episodes):
            env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
            obs, _ = env.reset(seed=args.seed + mi * 101 + ep)
            state = model.initial_state(1)
            prev_ev = np.zeros(16, np.float32)
            prev_act = np.zeros(9, np.int8)
            use_cd = 0
            for step in range(525):
                st = env.privileged_state()
                vec, ev, event = fb(obs, prev_ev, prev_act)
                teacher_bits, target = oracle_action(obs, st, ex, keys, doors)
                hit = np.asarray(obs["rays"]["hit_type"])[:8]
                if use_cd > 0:
                    use_cd -= 1
                    teacher_bits[7] = 0
                elif np.any((hit == 2) | (hit == 3)) and target is not ex:
                    teacher_bits[7] = 1
                    use_cd = 40
                teacher = nearest_macro(legalize_action(teacher_bits, prev_act))

                x = torch.from_numpy(vec[None, :])
                with torch.inference_mode():
                    out, state = model.step(fb.split(x), state, torch.tensor([event]))
                student = macro_action(int(np.argmax(out["logits"][0].numpy())))
                student = legalize_action(student, prev_act)
                if student[7] and prev_act[7]:
                    student[7] = 0

                X.append(vec[None, :]); E.append(np.array([event])); A.append(np.array([teacher]))
                added += 1
                obs, _, term, trunc, _ = env.step(student)
                prev_ev, prev_act = ev, student.copy()
                if term or trunc:
                    break
            env.close()

    out = ROOT / args.out
    np.savez_compressed(out, X=np.concatenate(X), E=np.concatenate(E), A=np.concatenate(A))
    print(f"aggregated {added} on-policy transitions; dataset now {sum(len(a) for a in A)} -> {out}")


if __name__ == "__main__":
    main()
