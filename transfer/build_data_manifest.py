"""Reproducible manifest + integrity checks for the small licensed GameWam sample.

Records source/licence, file hashes, action mapping, deduplication, quality filtering and
episode-level train/validation split (no evaluation seeds or maps used).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from transfer.train_combat_conv import ACTIONS, map_action

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_samples" / "gamewam" / "defend"
STRIDE = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files, samples = [], []
    for ep in range(4):
        for ext in (".parquet", ".mp4"):
            p = SRC / f"{ep}{ext}"
            if p.exists():
                files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                              "sha256": sha(p)})
        parquet = SRC / f"{ep}.parquet"
        video = SRC / f"{ep}.mp4"
        actions = pq.read_table(parquet)["action"].to_pylist()
        cap = cv2.VideoCapture(str(video))
        i = 0
        seen = set()
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(actions):
                break
            if i % STRIDE == 0:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 8))
                key = hashlib.md5(small.tobytes()).hexdigest()
                variance = float(small.var())
                duplicate = key in seen
                seen.add(key)
                label = map_action(actions[i])
                samples.append({"episode": ep, "frame": i, "action_class": label,
                                "action_name": ACTIONS[label],
                                "duplicate": duplicate, "variance": round(variance, 2),
                                "keep": (not duplicate) and variance > 3.0})
            i += 1
        cap.release()

    kept = [s for s in samples if s["keep"]]
    hist = {ACTIONS[i]: sum(s["action_class"] == i for s in kept) for i in range(len(ACTIONS))}
    manifest = {
        "source": "https://huggingface.co/datasets/Yunncheng/gamewam-vizdoom",
        "license": "Apache-2.0 (dataset card + HF API verified)",
        "scenario": "defend_the_center (combat only)",
        "files": files,
        "stride": STRIDE,
        "total_sampled_frames": len(samples),
        "kept_frames": len(kept),
        "dropped_duplicates": sum(s["duplicate"] for s in samples),
        "dropped_low_variance": sum((not s["duplicate"]) and s["variance"] <= 3.0 for s in samples),
        "action_histogram": hist,
        "split_by_episode": {"train": [0, 1, 2], "val": [3]},
        "preprocessing": "BGR->RGB, resize 112x64, /255; canonical action mapped to 8 macro classes",
        "notes": "External data used only for offline representation experiments; no test seeds/maps.",
    }
    out = ROOT / "transfer" / "data_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(kept)}/{len(samples)} frames kept; {len(files)} files -> {out}")


if __name__ == "__main__":
    main()
