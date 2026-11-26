"""Candidate submission agent: frozen champion + learned DoorPolicyNet.

Compliant runtime: learned champion + learned DoorPolicyNet GRU + observation-derived
door mask. No hand-written action state machine. obs_door_mask is a pure function of
current observation rays (hit_type/distance) and does not select actions directly;
it only gates blending between two learned logit outputs.

Not promoted to submission/ by default. Evaluate against promotion gate before any copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet
from transfer.door_policy import DoorPolicyNet, obs_door_mask

W = Path(__file__).resolve().parent / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
        torch.set_num_threads(1)

        ck = torch.load(W / "champion_frozen.pt", map_location=self.device, weights_only=False)
        self.cfg = KeyStoneConfig(**ck.get("config", {}))
        self.fb = FeatureBuilder(self.cfg)
        self.champion = TransferKeyStoneNet(self.cfg, W / "find_return_11.pth.tar").to(self.device)
        sd = self.champion.state_dict()
        for k, v in ck["model"].items():
            if k in sd and tuple(v.shape) == tuple(sd[k].shape):
                sd[k] = v.to(self.device)
        self.champion.load_state_dict(sd)
        self.champion.eval()

        dck = torch.load(W / "door_policy_gru.pt", map_location=self.device, weights_only=False)
        self.door = DoorPolicyNet(
            feat_dim=dck["feat_dim"],
            num_macros=dck.get("num_macros", 17),
            hidden=dck.get("hidden", 128),
        ).to(self.device)
        self.door.load_state_dict(dck["model"])
        self.door.eval()

        self.reset()

    def reset(self) -> None:
        self.c_state = self.champion.initial_state(1, device=self.device)
        self.d_state = self.door.initial_state(1, device=self.device)
        self.prev_ev = np.zeros(16, np.float32)
        self.prev_act = np.zeros(9, np.int8)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        vec, ev, event = self.fb(observation, self.prev_ev, self.prev_act)
        x = torch.from_numpy(vec[None, :]).to(self.device)
        parts = self.fb.split(x)
        parts["rgb"] = torch.from_numpy(observation["rgb"])[None, :].to(self.device)

        c_out, self.c_state = self.champion.step(
            parts, self.c_state, torch.tensor([event], device=self.device)
        )
        champ_logits = c_out["logits"][0]

        d_logits, d_gate_logit, self.d_state = self.door.step(x, self.d_state)
        gate = torch.sigmoid(d_gate_logit[0, 0])
        mask = 1.0 if obs_door_mask(observation) else 0.0
        effective_gate = float(gate.item()) * mask
        mixed = (1.0 - effective_gate) * champ_logits + effective_gate * d_logits[0]

        idx = int(torch.argmax(mixed).item())
        action = legalize_action(macro_action(idx), self.prev_act)
        self.prev_ev, self.prev_act = ev, action.copy()
        return action
