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

EXIT_TYPES = {11, 52, 3004, 3005}  # Doom normal/secret exits, ZDoom UDMF exits


def find_exit(wad: str | Path, map_name: str = "MAP01"):
    """Return (x, y) of the first exit thing, or None if the map has no exit."""
    lumps = map_lumps(read_wad(wad), map_name)
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        for body in re.findall(r"thing\s*\{([^}]*)\}", text):
            fields = dict(re.findall(r"(\w+)\s*=\s*([-\w.]+)\s*;", body))
            try:
                if int(float(fields.get("type", "-1"))) in EXIT_TYPES:
                    return float(fields["x"]), float(fields["y"])
            except (KeyError, ValueError):
                continue
        return None
    things = lumps.get("THINGS", b"")
    for i in range(0, len(things) - 9, 10):
        x, y, angle, typ, flags = struct.unpack_from("<hhHHH", things, i)
        if int(typ) in EXIT_TYPES:
            return float(x), float(y)
    return None
