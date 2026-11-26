"""Scale the verified single-room base into larger randomized navigation arenas.

Training-time scaffolding only. Geometry/topology is copied from the official fixture (valid
UDMF); coordinates and player start are scaled/randomised. Every generated map is verified
with the privileged oracle before use. No map data reaches inference.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from doom_env.wad import Lump, map_lumps, read_wad, write_wad

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "maps" / "train" / "nav_0.wad"


def generate_large(out: Path, scale: float, player=(96, 96), angle=0, seed=0):
    text = map_lumps(read_wad(BASE), "MAP01")["TEXTMAP"].decode("utf-8")

    def scale_block(match):
        body = match.group(1)
        x = float(re.search(r"x = ([-\d.]+)", body).group(1)) * scale
        y = float(re.search(r"y = ([-\d.]+)", body).group(1)) * scale
        return f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}"

    text = re.sub(r"vertex \{([^}]*)\}", scale_block, text)
    # Replace the player thing with a randomised start inside the scaled room.
    text = re.sub(r"thing \{[^}]*type = 1;[^}]*\}\s*", "", text)
    text += (f"\nthing {{ x = {player[0] * scale:.1f}; y = {player[1] * scale:.1f}; "
             f"angle = {angle}; type = 1; skill1 = true; skill2 = true; skill3 = true; "
             f"skill4 = true; skill5 = true; single = true; }}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode("utf-8")),
                    Lump("ENDMAP", b"")])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", default="maps/large")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    rng = random.Random(7)
    out = ROOT / args.out
    paths = []
    for i in range(args.count):
        scale = rng.choice([2.0, 3.0, 4.0])
        player = (rng.choice([96, 128, 160]), rng.choice([80, 96, 110]))
        angle = rng.choice([0, 45, 90, 135, 180, -90, -135, -45])
        paths.append(generate_large(out / f"large_{i}.wad", scale, player, angle))
    print(f"generated {len(paths)} maps in {out}")
    if args.verify:
        from training.verify_maps import verify_all
        verify_all(paths)


if __name__ == "__main__":
    main()
