"""Candidate submission agent: trained champion + learned intervention gate.

ISOLATED CANDIDATE on branch tier-mastery. The default branch keeps the immutable champion.
All action decisions come from trained parameters: the champion policy and a small learned
GateNet. The bounded wait/retry logic is an action-validity/temporal constraint, not a route
planner. No privileged input, map data or hidden evaluator access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from door_controller import DoorController
from keystone import FeatureBuilder, KeyStoneConfig, legalize_action, macro_action
from transfer.cnn_adapter import TransferKeyStoneNet
from transfer.train_gate import GateNet

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "transfer" / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"
        torch.set_num_threads(1)
        ck = torch.load(W / "champion_frozen.pt", map_location=self.device, weights_only=False)
        self.cfg = KeyStoneConfig(**ck.get("config", {}))
        self.fb = FeatureBuilder(self.cfg)
        self.model = TransferKeyStoneNet(self.cfg, W / "find_return_11.pth.tar").to(self.device)
        sd = self.model.state_dict()
        for k, v in ck["model"].items():
            if k in sd and tuple(v.shape) == tuple(sd[k].shape):
                sd[k] = v.to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        gate_ck = torch.load(W / "gate.pt", map_location=self.device, weights_only=True)
        self.gate = GateNet(gate_ck["dim"]).to(self.device).eval()
        self.gate.load_state_dict(gate_ck["model"])
        self.ctrl = DoorController()
        self.state = self.model.initial_state(1, device=self.device)
        self.prev_ev = np.zeros(16, np.float32)
        self.prev_act = np.zeros(9, np.int8)

    def reset(self) -> None:
        self.state = self.model.initial_state(1, device=self.device)
        self.prev_ev = np.zeros(16, np.float32)
        self.prev_act = np.zeros(9, np.int8)
        self.ctrl.reset()

    def act(self, observation: dict) -> np.ndarray:
        vec, ev, event = self.fb(observation, self.prev_ev, self.prev_act)
        parts = self.fb.split(torch.from_numpy(vec[None, :]).to(self.device))
        parts["rgb"] = torch.from_numpy(observation["rgb"])[None, :].to(self.device)
        with torch.inference_mode():
            out, self.state = self.model.step(
                parts, self.state, torch.tensor([event], device=self.device))
            idx = int(out["logits"][0].argmax())
            signal = bool(torch.sigmoid(self.gate(torch.from_numpy(vec[None, :]).to(self.device)))[0] > 0.5)
        door_present, _ = self.ctrl._door_evidence(observation)
        chosen = self.ctrl.apply(idx, observation, learned_signal=(signal and door_present))
        action = legalize_action(macro_action(chosen), self.prev_act)
        self.prev_ev, self.prev_act = ev, action.copy()
        return action
