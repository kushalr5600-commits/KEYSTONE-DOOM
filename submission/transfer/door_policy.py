"""Compliant neural DoorPolicyNet for door interaction distillation.

Runtime: learned model + observation-derived logit mask only. No hand-written
action state machine. DoorController is used as training-time teacher only.
"""
from __future__ import annotations

import torch
from torch import nn


class DoorPolicyNet(nn.Module):
    """Small GRU policy that outputs macro logits and a gate logit.

    Input: feature vector from FeatureBuilder (float32, shape [B, D] or [T,B,D]).
    Outputs:
      macro_logits: [B, num_macros] or [T,B,num_macros]
      gate_logit:   [B, 1] or [T,B,1] - learned override gate
    """

    def __init__(self, feat_dim: int, num_macros: int = 17, hidden: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_macros = num_macros
        self.hidden = hidden
        self.in_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, num_macros + 1)  # +1 gate

    def initial_state(self, batch: int, device=None, dtype=torch.float32):
        return torch.zeros(batch, self.hidden, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, h: torch.Tensor):
        """Single step. x: [B,D], h: [B,H]. Returns (macro_logits, gate_logit, h_new)."""
        z = self.in_proj(x)
        h_new = self.gru(z, h)
        o = self.out(h_new)
        return o[:, : self.num_macros], o[:, self.num_macros : self.num_macros + 1], h_new

    def forward_seq(self, x_seq: torch.Tensor, h0: torch.Tensor):
        """x_seq: [T,B,D], h0: [B,H]. Returns macro_logits [T,B,M], gate [T,B,1], hT."""
        T = x_seq.shape[0]
        h = h0
        macro_list, gate_list = [], []
        for t in range(T):
            ml, gl, h = self.step(x_seq[t], h)
            macro_list.append(ml)
            gate_list.append(gl)
        return torch.stack(macro_list, 0), torch.stack(gate_list, 0), h


def obs_door_mask(obs: dict) -> bool:
    """Observation-derived door evidence mask. Pure function of current obs, no state.

    Returns True when front rays indicate a nearby door/locked-door and no enemy in front.
    """
    import numpy as np

    rays = obs["rays"]
    hit = np.asarray(rays["hit_type"]).reshape(-1)
    dist = np.asarray(rays["distance"]).reshape(-1)
    # Front rays: same convention as DoorController (0-8, 55-63)
    front = list(range(0, 9)) + list(range(55, 64))
    fh = hit[front]
    fd = dist[front]
    door_near = np.any(np.isin(fh, (2, 3)) & (fd < 0.25))
    enemy_front = np.any(fh == 4)
    return bool(door_near and not enemy_front)
