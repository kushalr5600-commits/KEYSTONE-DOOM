"""Run the required ablation matrix with an identical test protocol.

Each variant is trained for the same number of steps with the same seed, then evaluated
deterministically on the same held-out maps/seeds. Results append to experiments/results.csv.

    python -m training.run_ablations --steps 6000 --episodes 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    "ablation_1_reactive", "ablation_2_recurrent", "ablation_3_memory",
    "ablation_4_goal", "ablation_5_full", "ablation_6_depth",
]
EVAL_MAPS = [
    "maps/two_rooms.wad:MAP01",
    "tests/fixtures/509aca7cabb8cd7f.wad:MAP01",
    "tests/fixtures/5c688a052900e964.wad:MAP01",
    "tests/fixtures/68d6a90a9de31bb8.wad:MAP01",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    variants = args.only or VARIANTS
    for name in variants:
        cfg = ROOT / "configs" / f"{name}.json"
        outdir = ROOT / "experiments" / name
        print(f"\n=== TRAIN {name} ===", flush=True)
        subprocess.run([
            sys.executable, "-m", "training.train", "--steps", str(args.steps),
            "--config", str(cfg), "--out", str(outdir), "--tag", name, "--threads", "4",
        ], cwd=ROOT, check=True)
        print(f"\n=== EVAL {name} ===", flush=True)
        subprocess.run([
            sys.executable, "-m", "training.evaluate",
            "--weights", str(outdir / "keystone.pt"), "--tag", name,
            "--episodes", str(args.episodes), "--maps", *EVAL_MAPS,
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
