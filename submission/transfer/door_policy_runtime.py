"""Compliant runtime for DoorPolicyNet distillation.

No hand-written action state machine. Decisions come from:
- frozen champion model
- learned DoorPolicyNet (GRU) macro logits + learned gate
- observation-derived logit mask (obs_door_mask)

The mask is a pure function of the current observation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet
from transfer.door_policy import DoorPolicyNet, obs_door_mask

ROOT = Path(__file__).resolve().parents[1]


class DoorPolicyRuntime:
    def __init__(self, champion_ckpt="transfer/weights/champion_frozen.pt",
                 pretrained="transfer/weights/find_return_11.pth.tar",
                 door_ckpt="transfer/weights/door_policy_gru.pt",
                 device="cpu"):
        self.device = device
        ck = torch.load(ROOT / champion_ckpt, map_location=device, weights_only=False)
        self.cfg = KeyStoneConfig(**ck.get("config", {}))
        self.fb = FeatureBuilder(self.cfg)
        self.champion = TransferKeyStoneNet(self.cfg, ROOT / pretrained).to(device)
        sd = self.champion.state_dict()
        for k, v in ck["model"].items():
            if k in sd and tuple(v.shape) == tuple(sd[k].shape):
                sd[k] = v.to(device)
        self.champion.load_state_dict(sd)
        self.champion.eval()

        dck = torch.load(ROOT / door_ckpt, map_location=device, weights_only=False)
        self.door = DoorPolicyNet(feat_dim=dck["feat_dim"],
                                  num_macros=dck.get("num_macros", 17),
                                  hidden=dck.get("hidden", 128)).to(device)
        self.door.load_state_dict(dck["model"])
        self.door.eval()
        self.reset()

    def reset(self):
        self.c_state = self.champion.initial_state(1, device=self.device)
        self.d_state = self.door.initial_state(1, device=self.device)
        self.prev_ev = np.zeros(16, np.float32)
        self.prev_act = np.zeros(9, np.int8)

    @torch.inference_mode()
    def act(self, observation: dict):
        vec, ev, event = self.fb(observation, self.prev_ev, self.prev_act)
        x = torch.from_numpy(vec[None, :]).to(self.device)
        parts = self.fb.split(x)
        parts["rgb"] = torch.from_numpy(observation["rgb"])[None, :].to(self.device)

        c_out, self.c_state = self.champion.step(parts, self.c_state,
                                                 torch.tensor([event], device=self.device))
        champ_logits = c_out["logits"][0]

        d_logits, d_gate_logit, self.d_state = self.door.step(x, self.d_state)
        d_logits = d_logits[0]
        gate = torch.sigmoid(d_gate_logit[0, 0])

        # Observation-derived mask: pure function of current obs, no state machine.
        mask = 1.0 if obs_door_mask(observation) else 0.0
        effective_gate = float(gate.item()) * mask

        mixed = (1.0 - effective_gate) * champ_logits + effective_gate * d_logits
        idx = int(torch.argmax(mixed).item())
        champ_idx = int(torch.argmax(champ_logits).item())
        overridden = (idx != champ_idx) and (mask > 0.0)

        action = legalize_action(macro_action(idx), self.prev_act)
        self.prev_ev, self.prev_act = ev, action.copy()
        return action, {
            "champ_idx": champ_idx,
            "door_idx": int(torch.argmax(d_logits).item()),
            "chosen_idx": idx,
            "gate": float(gate.item()),
            "mask": mask,
            "effective_gate": effective_gate,
            "overridden": bool(overridden),
        }
