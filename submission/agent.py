"""Evaluation entry point: learned KEYSTONE-DOOM agent.

Contract: docs/CONTRACT.md §6. The official scorer instantiates ``DoomAgent`` once, calls
``reset()`` before every map and ``act(observation)`` once per 4 engine tics.

This agent uses ONLY the official observation dictionary. It holds a recurrent state and
a fixed-capacity learned event memory, both cleared in ``reset()``. All action choices come
from trained network weights; the only post-processing is contract button-validity
enforcement (keystone.actions.legalize_action), shared with training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from keystone import (
    AgentState,
    FeatureBuilder,
    KeyStoneConfig,
    KeyStoneNet,
    legalize_action,
    macro_action,
)

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = "cuda" if str(device) == "cuda" and torch.cuda.is_available() else "cpu"

        cfg_path = WEIGHTS_DIR / "config.json"
        try:
            self.cfg = KeyStoneConfig.load(cfg_path)
        except (OSError, ValueError, TypeError):
            self.cfg = KeyStoneConfig()
        self.fb = FeatureBuilder(self.cfg)
        torch.set_num_threads(1)  # evaluation gives exactly one CPU thread
        self.model = KeyStoneNet(self.cfg).to(self.device).eval()

        ckpt = torch.load(WEIGHTS_DIR / "keystone.pt", map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.state: AgentState = self.model.initial_state(1, device=self.device)
        self.prev_evidence = np.zeros(16, dtype=np.float32)
        self.prev_action = np.zeros(9, dtype=np.int8)
        self.n_calls = 0

    def reset(self) -> None:
        """Called before every map: clear all recurrent and episodic state."""
        self.state = self.model.initial_state(1, device=self.device)
        self.prev_evidence = np.zeros(16, dtype=np.float32)
        self.prev_action = np.zeros(9, dtype=np.int8)
        self.n_calls = 0

    def act(self, observation: dict) -> np.ndarray:
        vec, evidence, event = self.fb(observation, self.prev_evidence, self.prev_action)
        x = torch.from_numpy(vec[None, :]).to(self.device)
        parts = self.fb.split(x)
        with torch.inference_mode():
            out, self.state = self.model.step(
                parts, self.state, torch.tensor([event], device=self.device, dtype=torch.long))
            logits = out["logits"][0].cpu().numpy()
            # Debug-only snapshot for the demo ledger; never used to choose an action.
            self.last_goal_probs = out["goal_logits"][0].softmax(-1).cpu().numpy()
            self.last_confidence = float(out["risk"][0].mean().item())
            self.last_memory = {
                "event": out["read_weights"][0].argmax().item(),
                "events": self.state.mem.etype[0].argmax(-1).detach().cpu().numpy(),
                "conf": self.state.mem.conf[0, :, 0].detach().cpu().numpy(),
                "age": self.state.mem.age[0, :, 0].detach().cpu().numpy(),
            }
        if self.cfg.action_mode == "macro":
            sampled = macro_action(int(np.argmax(logits)))
        else:
            sampled = (logits > 0.0).astype(np.int8)  # argmax per Bernoulli button
        action = legalize_action(sampled, self.prev_action)
        self.prev_action = action.copy()
        self.prev_evidence = evidence
        self.n_calls += 1
        return action
