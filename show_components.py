import csv
from statistics import mean

rows = list(csv.DictReader(open("transfer/component_results.csv", encoding="utf-8")))
for tag in sorted(set(r["tag"] for r in rows)):
    rr = [r for r in rows if r["tag"] == tag]
    print("===", tag, len(rr))
    for k in ("exit", "health_end", "damage", "kills", "deaths", "keys",
              "items", "doors_opened", "locks_unlocked", "unique_sigs",
              "revisit_rate", "steps", "use_attempts", "latency_p50_ms"):
        print(" ", k, round(mean(float(r[k]) for r in rr), 3))
