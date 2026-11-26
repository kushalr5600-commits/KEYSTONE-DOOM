"""Isolated adapter for MIT-licensed edbeeching/3d_control_deep_rl CNN weights.

We reuse ONLY the 3-conv visual stem (16/32/16 filters) and its projection to 128 units.
Source: https://github.com/edbeeching/3d_control_deep_rl, LICENSE MIT
Copyright (c) 2019 Ed Beeching. License text preserved in transfer/licenses/.

The adapter resizes DevForge's 160x120 RGB to the source 112x64 input, and injects the
pretrained visual representation additively into our KEYSTONE latent `z`, so every other
layer keeps its shape and a baseline checkpoint can be loaded unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from keystone import KeyStoneConfig, KeyStoneNet

ROOT = Path(__file__).resolve().parents[1]


class PretrainedConvEncoder(nn.Module):
    def __init__(self, ckpt_path: str | Path):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1), nn.ReLU(True),
        )
        self.linear1 = nn.Linear(640, 128)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        mapping = {
            "conv_head.0.weight": "conv_head.0.weight", "conv_head.0.bias": "conv_head.0.bias",
            "conv_head.2.weight": "conv_head.2.weight", "conv_head.2.bias": "conv_head.2.bias",
            "conv_head.4.weight": "conv_head.4.weight", "conv_head.4.bias": "conv_head.4.bias",
            "linear1.weight": "linear1.weight", "linear1.bias": "linear1.bias",
        }
        own = self.state_dict()
        for src, dst in mapping.items():
            assert tuple(sd[src].shape) == tuple(own[dst].shape), (src, sd[src].shape)
            own[dst] = sd[src]
        self.load_state_dict(own)
        self.source = str(ckpt_path)

    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        # (B,H,W,3) uint8 -> (B,3,64,112) float in [0,1]
        x = rgb_uint8.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(64, 112), mode="bilinear", align_corners=False) / 255.0
        x = self.conv_head(x).flatten(1)
        return F.relu(self.linear1(x))


class TransferKeyStoneNet(KeyStoneNet):
    """KEYSTONE + additive pretrained visual features. No core files modified."""

    def __init__(self, cfg: KeyStoneConfig, pretrained_ckpt: str | Path):
        super().__init__(cfg)
        self.pretrained_conv = PretrainedConvEncoder(pretrained_ckpt)
        self.visual_proj = nn.Linear(128, cfg.hidden)
        nn.init.zeros_(self.visual_proj.weight)
        nn.init.zeros_(self.visual_proj.bias)
        self.use_visual = True

    def encode(self, parts):
        z = super().encode(parts)
        if self.use_visual and "rgb" in parts:
            assert parts["rgb"].shape[-1] == 3, "RGB must be (B,H,W,uint8 channels)"
            v = self.pretrained_conv(parts["rgb"])
            z = z + self.visual_proj(v)
        return z
