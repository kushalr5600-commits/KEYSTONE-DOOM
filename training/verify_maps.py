"""Oracle verification of generated training maps.

The oracle uses privileged position/map data to steer and press USE. This is a
curriculum-design check only: it proves each reward is achievable. It is never imported by
the deployed agent or used to label observations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env.wad import map_lumps, read_wad
from training.exit_finder import find_exit, EXIT_TYPES

ROOT = Path(__file__).resolve().parents[1]


def find_keys(wad, map_name="MAP01"):
    """Return [(x,y)] for key things in a UDMF map (training verification only)."""
    import re
    text = map_lumps(read_wad(wad), map_name)["TEXTMAP"].decode("utf-8", "replace")
    pts = []
    for body in re.findall(r"thing\s*\{([^}]*)\}", text):
        f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
        try:
            if int(float(f.get("type", "-1"))) in (5, 6, 7):
                pts.append((float(f["x"]), float(f["y"])))
        except (KeyError, ValueError):
            pass
    return pts


def find_doors(wad, map_name="MAP01"):
    """Midpoints of USE-openable door linedefs (special 12/13), training verification only."""
    import re
    text = map_lumps(read_wad(wad), map_name)["TEXTMAP"].decode("utf-8", "replace")
    verts = [(float(x), float(y)) for x, y in
             re.findall(r"vertex \{ x = ([-\w.]+); y = ([-\w.]+);", text)]
    pts = []
    for body in re.findall(r"linedef \{([^}]*)\}", text):
        f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)", body))
        try:
            if int(float(f.get("special", "-1"))) in (12, 13):
                a, b = verts[int(f["v1"])], verts[int(f["v2"])]
                pts.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        except (KeyError, ValueError, IndexError):
            pass
    return pts


def oracle_episode(wad, ticks=2100, seed=0):
    ex = find_exit(wad, "MAP01")
    keys = find_keys(wad, "MAP01")
    doors = find_doors(wad, "MAP01")
    env = make_env(str(wad), "MAP01", episode_ticks=ticks, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    use_cooldown = 0
    for step in range(ticks // 4):
        st = env.privileged_state()
        target = ex
        if keys and not np.any(np.asarray(st["keys"]) > 0.5):
            target = keys[0]
        elif doors and ex is not None and st["x"] < max(d[0] for d in doors) + 20:
            # Walk to the nearest door first; after crossing it (x > 520) head to exit.
            target = min(doors, key=lambda p: (p[0]-st["x"])**2 + (p[1]-st["y"])**2)
        if target is None:
            break
        dx, dy = target[0] - st["x"], target[1] - st["y"]
        desired = np.degrees(np.arctan2(dy, dx))
        err = (desired - st["angle"] + 180) % 360 - 180
        a = np.zeros(9, np.int8)
        if abs(err) > 8:
            a[4 if err > 0 else 5] = 1
        else:
            a[0] = 1
        hit = np.asarray(obs["rays"]["hit_type"])[:8]
        if use_cooldown > 0:
            use_cooldown -= 1
        elif np.any((hit == 2) | (hit == 3)) and target is not ex:
            # One press, then wait ~1.5 s: repeated USE reverses a moving door.
            a[7] = 1
            use_cooldown = 40
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    s = env.privileged_state()
    result = {"map": Path(wad).name, "exit": bool(s["exit_reached"]),
              "keys": int(np.sum(np.asarray(s["keys"]) > 0.5)),
              "dead": bool(s["dead"]), "steps": step + 1}
    env.close()
    return result


def verify_all(paths):
    ok = 0
    for p in paths:
        r = oracle_episode(p)
        print(("PASS " if r["exit"] else "FAIL ") + str(r), flush=True)
        ok += int(r["exit"])
    print(f"verified {ok}/{len(paths)} maps reachable")
    return ok == len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*")
    args = ap.parse_args()
    paths = [Path(m) for m in args.maps] or sorted((ROOT / "maps" / "train").glob("*.wad"))
    verify_all(paths)


if __name__ == "__main__":
    main()
