"""Generate small training maps with VERIFIED, reachable objectives.

Training-time only. We reuse the official fixture *geometry* (guaranteed valid UDMF for
ViZDoom 1.3.3) and replace its THINGS with controlled player/exit/monster/key placements.
Every generated map is checked with an oracle controller before being used, so a reward can
only be given for a behaviour that is actually achievable.

The deployed agent never reads map files or this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from doom_env.wad import Lump, map_lumps, read_wad, write_wad

ROOT = Path(__file__).resolve().parents[1]
BASES = {
    "room": ROOT / "tests" / "fixtures" / "57a739138e7e276d.wad",
    "two": ROOT / "tests" / "fixtures" / "6a392a907969372f.wad",
    "door": ROOT / "tests" / "fixtures" / "509aca7cabb8cd7f.wad",
    "locked": ROOT / "tests" / "fixtures" / "7965e3db9e16c412.wad",
    "rooms": ROOT / "maps" / "two_rooms.wad",
}

# ZDoom UDMF editor numbers. 2011 is the imp used by the official test map.
# Classic exit thing (type 11). Do NOT use 3004: in UDMF zdoom that is a Demon monster.
PLAYER, EXIT, IMP, BLUE_CARD, YELLOW_CARD, RED_CARD = 1, 11, 2011, 5, 6, 7


def _thing(x, y, typ, angle=0):
    return (f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {angle}; type = {typ}; "
            f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; "
            f"skill5 = true; single = true; }}")


def generate(base: str, out: Path, things: list[tuple[float, float, int, int]]):
    text = map_lumps(read_wad(BASES[base]), "MAP01")["TEXTMAP"].decode("utf-8")
    text = re.sub(r"thing\s*\{[^}]*\}\s*", "", text)
    text += "\n" + "\n".join(_thing(x, y, typ, a) for x, y, typ, a in things) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out, [Lump("MAP01", b""), Lump("TEXTMAP", text.encode("utf-8")),
                    Lump("ENDMAP", b"")])
    return out


def make_curriculum():
    """All exits here are the bases' crossable linedefs (special 243), not inert things."""
    d = ROOT / "maps" / "train"
    specs = []
    # 1. Movement/orientation: single room (two), exit line 224 units east at (320,96).
    for i, (px, py, angle) in enumerate([
        (96, 96, 0), (96, 96, 180), (96, 110, -90), (96, 110, 90),
        (160, 96, 45), (160, 96, -45),
    ]):
        specs.append((f"move_{i}", generate("two", d / f"move_{i}.wad",
                      [(px, py, PLAYER, angle)])))
    # 2. Navigation without threats: same map, varied start orientation.
    for i, (px, py, angle) in enumerate([
        (96, 96, 90), (160, 110, -180), (96, 110, 45), (160, 96, -90),
    ]):
        specs.append((f"nav_{i}", generate("two", d / f"nav_{i}.wad",
                      [(px, py, PLAYER, angle)])))
    # 3. Combat/survival: monster in the verified single-room map (exit line at 320,96).
    for i, (mx, my) in enumerate([(220, 110), (250, 96), (180, 96)]):
        specs.append((f"combat_{i}", generate("two", d / f"combat_{i}.wad",
                      [(96, 96, PLAYER, 0), (mx, my, IMP, 180)])))
    # 4. Key/lock/revisit: blue card in the near room, locked door, exit line behind it.
    #    Verified oracle: key pickup -> USE on locked door -> exit, ~25 steps.
    for i, (kx, ky, angle) in enumerate([(160, 110, 0), (160, 96, 180), (200, 110, 90)]):
        specs.append((f"key_{i}", generate("locked", d / f"key_{i}.wad",
                      [(96, 96, PLAYER, 0), (kx, ky, BLUE_CARD, angle)])))
    return specs


def make_holdout():
    """Separate maps for held-out evaluation; never used in any curriculum stage."""
    d = ROOT / "maps" / "holdout"
    specs = []
    for i, (px, py, angle) in enumerate([(128, 110, 135), (144, 96, -135), (112, 110, 70)]):
        specs.append((f"eval_move_{i}", generate("two", d / f"eval_move_{i}.wad",
                      [(px, py, PLAYER, angle)])))
    for i, (px, py, angle) in enumerate([(128, 110, -70), (144, 110, 160)]):
        specs.append((f"eval_nav_{i}", generate("two", d / f"eval_nav_{i}.wad",
                      [(px, py, PLAYER, angle)])))
    for i, (mx, my) in enumerate([(240, 110), (210, 96)]):
        specs.append((f"eval_combat_{i}", generate("two", d / f"eval_combat_{i}.wad",
                      [(96, 96, PLAYER, 0), (mx, my, IMP, 180)])))
    for i, (kx, ky) in enumerate([(176, 110), (192, 96)]):
        specs.append((f"eval_key_{i}", generate("locked", d / f"eval_key_{i}.wad",
                      [(96, 96, PLAYER, 0), (kx, ky, BLUE_CARD, 0)])))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    specs = make_holdout() if args.holdout else make_curriculum()
    folder = "maps/holdout/" if args.holdout else "maps/train/"
    print(f"generated {len(specs)} maps under {folder}")
    if args.verify:
        from training.verify_maps import verify_all
        verify_all([p for _, p in specs])


if __name__ == "__main__":
    main()
