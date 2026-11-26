"""Training-only helper: locate the exit thing in a WAD.

This reads map data OUTSIDE the agent process, purely to compute a potential-based reward
for training. The deployed policy never sees map files, positions or this module
(docs/RULES.md §2 explicitly permits training-time engine/map manipulation; the official
CONTRACT.md §7 explicitly permits privileged_state() reward shaping during training).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from doom_env.wad import map_lumps, read_wad

# Classic Doom exit things. NOTE: in the UDMF "zdoom" namespace used by the official
# fixtures, editor numbers 3004/3005 are *monsters* (Demon / Hell Knight), NOT exits.
# Treating them as exits was a serious bug in earlier training runs: it pushed the agent
# toward a monster. UDMF maps use exit linedef specials instead (handled below).
EXIT_TYPES = {11, 52}
UDMF_EXIT_SPECIALS = {243, 51, 124}  # Exit_Normal / Exit_Secret style line specials


def find_exit(wad: str | Path, map_name: str = "MAP01"):
    """Return (x, y) of the first exit thing, or None if the map has no exit."""
    lumps = map_lumps(read_wad(wad), map_name)
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        # UDMF exits are crossable linedefs; check these FIRST. Classic exit things in
        # the zdoom namespace are unreliable (3004 is a monster, 11 may be inert).
        verts = []
        for body in re.findall(r"vertex\s*\{([^}]*)\}", text):
            f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                verts.append((float(f["x"]), float(f["y"])))
            except (KeyError, ValueError):
                pass
        for body in re.findall(r"linedef\s*\{([^}]*)\}", text):
            f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                if int(float(f.get("special", "-1"))) in UDMF_EXIT_SPECIALS:
                    a, b = verts[int(f["v1"])], verts[int(f["v2"])]
                    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            except (KeyError, ValueError, IndexError):
                continue
        for body in re.findall(r"thing\s*\{([^}]*)\}", text):
            fields = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                if int(float(fields.get("type", "-1"))) in EXIT_TYPES:
                    return float(fields["x"]), float(fields["y"])
            except (KeyError, ValueError):
                continue
        # UDMF exits are crossable linedefs, not things. Use their midpoint.
        verts = []
        for body in re.findall(r"vertex\s*\{([^}]*)\}", text):
            f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                verts.append((float(f["x"]), float(f["y"])))
            except (KeyError, ValueError):
                pass
        for body in re.findall(r"linedef\s*\{([^}]*)\}", text):
            f = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                if int(float(f.get("special", "-1"))) in UDMF_EXIT_SPECIALS:
                    a, b = verts[int(f["v1"])], verts[int(f["v2"])]
                    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            except (KeyError, ValueError, IndexError):
                continue
        return None
    things = lumps.get("THINGS", b"")
    for i in range(0, len(things) - 9, 10):
        x, y, angle, typ, flags = struct.unpack_from("<hhHHH", things, i)
        if int(typ) in EXIT_TYPES:
            return float(x), float(y)
    return None
