"""Sample-efficient combat representation pretraining from licensed GameWam episodes.

Only 4 small episodes (~88 MB) are used. Apache-2.0 dataset, attribution recorded in
transfer/DATA_REUSE_AUDIT.md. We train a 3-conv visual stem identical to the MIT encoder
already used by the champion, but on combat actions, then test whether swapping only those
visual weights improves combat without regressing navigation/progression.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

ACTIONS = ("stand", "fwd", "turn_left", "turn_right", "attack",
           "attack_left", "attack_right", "fwd_attack")


class CombatConv(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1), nn.ReLU(True))
        self.linear1 = nn.Linear(640, 128)
        self.head = nn.Linear(128, n_actions)

    def forward(self, x):
        return self.head(torch.relu(self.linear1(self.conv_head(x).flatten(1))))


def map_action(v):
    shoot, fwd, back, sl, sr, speed, tl, tr, delta = [float(x) for x in v]
    if tl > 0.5 and shoot > 0.5:
        return 5
    if tr > 0.5 and shoot > 0.5:
        return 6
    if shoot > 0.5:
        return 4
    if tl > 0.5:
        return 2
    if tr > 0.5:
        return 3
    if fwd > 0.5:
        return 1 if shoot <= 0.5 else 7
    return 0


def load_data(directory: Path, stride: int = 4):
    X, Y = [], []
    for ep in range(4):
        parquet = directory / f"{ep}.parquet"
        video = directory / f"{ep}.mp4"
        actions = pq.read_table(parquet)["action"].to_pylist()
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(actions):
                break
            if i % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (112, 64), interpolation=cv2.INTER_AREA)
                X.append(rgb.transpose(2, 0, 1))
                Y.append(map_action(actions[i]))
            i += 1
        cap.release()
    return np.asarray(X, np.uint8), np.asarray(Y, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_samples/gamewam/defend")
    ap.add_argument("--out", default="transfer/weights/combat_conv.pt")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(4)
    X, Y = load_data(ROOT / args.data)
    print(f"samples={len(X)} classes={np.bincount(Y).tolist()}", flush=True)
    x = torch.from_numpy(X).float().div_(255.0)
    y = torch.from_numpy(Y).long()
    model = CombatConv(len(ACTIONS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(idx)
        print(f"epoch {epoch+1} loss={total/len(x):.4f} acc="
              f"{float((model(x).argmax(-1) == y).float().mean()):.3f} "
              f"samples/s={len(x)/(time.perf_counter()-t0):.0f}", flush=True)
    out = ROOT / args.out
    sd = model.state_dict()
    torch.save({k: v.cpu() for k, v in sd.items() if k.startswith(("conv_head", "linear1"))}, out)
    print("wrote visual weights", out)


if __name__ == "__main__":
    main()
