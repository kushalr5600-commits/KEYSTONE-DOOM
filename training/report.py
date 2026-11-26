"""Aggregate experiments/results.csv into a compact table with bootstrap CIs.

    python -m training.report
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "experiments" / "results.csv"
METRICS = ("exit", "keys", "kills", "deaths", "damage", "steps",
           "unique_sigs", "stuck_rate", "latency_p50_ms", "latency_p99_ms")


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return tuple(float(v) for v in np.percentile(means, (2.5, 97.5)))


def main():
    if not CSV.exists():
        print("no results.csv yet")
        return
    groups = defaultdict(list)
    with CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row.get("tag", "unknown")].append(row)
    for tag, rows in groups.items():
        print(f"\n## {tag}  (n={len(rows)} episodes)")
        print(f"{'metric':18s} {'mean':>10s} {'95% CI':>22s}")
        for m in METRICS:
            vals = np.asarray([float(r.get(m, 0) or 0) for r in rows])
            # `deaths` may not be present in older rows; fall back to the death flag.
            if m == "deaths" and all("deaths" not in r for r in rows):
                vals = np.asarray([float(r.get("death", 0) or 0) for r in rows])
            lo, hi = boot_ci(vals)
            print(f"{m:18s} {vals.mean():10.3f} [{lo:8.3f}, {hi:8.3f}]")


if __name__ == "__main__":
    main()
