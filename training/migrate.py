"""Migrate compatible weights between action-head variants.

When switching from the legacy 9-Bernoulli head to the 17-macro categorical head, every
encoder/recurrent/memory/goal/value layer keeps the same shape. Only the actor's final
linear layer changes. This tool preserves the rest of a checkpoint instead of restarting
from random weights, and records exactly what was copied.

    python -m training.migrate --src submission/weights/keystone.pt \
        --config configs/m5_full.json --out experiments/migrated_full/keystone.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from keystone import KeyStoneConfig, KeyStoneNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = ROOT / args.src
    dst = ROOT / args.out
    old = torch.load(src, map_location="cpu", weights_only=False)
    old_cfg = KeyStoneConfig(**old.get("config", {}))
    new_cfg = KeyStoneConfig.load(ROOT / args.config)
    model = KeyStoneNet(new_cfg)

    old_sd, new_sd = old["model"], model.state_dict()
    copied, skipped = [], []
    for k, v in old_sd.items():
        if k in new_sd and tuple(v.shape) == tuple(new_sd[k].shape):
            new_sd[k] = v
            copied.append(k)
        else:
            skipped.append((k, tuple(v.shape)))
    model.load_state_dict(new_sd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": json.loads(new_cfg.to_json()),
                "step": 0, "migrated_from": str(src),
                "copied_layers": copied, "skipped_layers": skipped}, dst)
    new_cfg.save(dst.parent / "config.json")
    print(f"copied {len(copied)} tensors, skipped {len(skipped)}")
    for k, shape in skipped:
        print(f"  skipped {k} {shape} (action head changed)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
