"""Balance the rare USE action in progression demonstrations.

The teacher correctly presses USE on the locked door, but only 64/13664 (0.5%) of frames
carry that label, so plain BC ignores it. This tool upsamples USE-labelled transitions and
 downsamples routine turning, producing a progression-focused BC set. No new privileged
information is added; the student still receives only official observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="transfer/weights/demos_key.npz")
    ap.add_argument("--out", default="transfer/weights/demos_key_bal.npz")
    ap.add_argument("--use-repeat", type=int, default=40)
    ap.add_argument("--other-keep", type=int, default=3000)
    args = ap.parse_args()

    d = np.load(ROOT / args.src)
    X, RGB, E, A = d["X"], d["RGB"], d["E"], d["A"]
    use_mask = np.isin(A, (12, 13, 14, 15))
    other_idx = np.flatnonzero(~use_mask)
    rng = np.random.default_rng(0)
    keep = rng.choice(other_idx, size=min(args.other_keep, len(other_idx)), replace=False)
    idx = np.concatenate([
        np.repeat(np.flatnonzero(use_mask), args.use_repeat),
        keep,
    ])
    rng.shuffle(idx)
    out = ROOT / args.out
    np.savez_compressed(out, X=X[idx], RGB=RGB[idx], E=E[idx], A=A[idx])
    print(f"balanced {len(idx)} samples: USE={int(use_mask[idx].sum())} "
          f"({use_mask[idx].mean():.2%}) -> {out}")


if __name__ == "__main__":
    main()
