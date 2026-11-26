"""Isolated, observation-only door intervention wrapper (experiment only).

Does NOT modify the frozen champion weights, architecture or submission inference code. It
wraps the trained policy's macro action with a bounded obstruction-triggered interaction and
hands control straight back once the door evidence clears.

All signals are available from the official observation dict (rays, vitals) and our own prior
actions. No privileged state, map data, evaluator access or hardcoded route is used.

COMPLIANCE NOTE: official RULES.md §1 forbids hand-written if/else action trees in the final
agent. This wrapper is an experiment to measure whether a modular interaction gate helps;
before deployment it must be replaced by a learned gate trained on these interventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRONT = tuple(list(range(0, 9)) + list(range(55, 64)))
USE_MACRO = 12
FORWARD_MACRO = 1
TURN_RIGHT_MACRO = 5


@dataclass
class DoorConfig:
    trigger_distance: float = 0.18      # door within 184 map units
    min_door_rays: int = 1
    persist_steps: int = 3
    wait_steps: int = 36                # ~1.4 s of game time for the door to open
    max_retries: int = 3
    cooldown_steps: int = 80
    escape_turn_steps: int = 12
    max_locked_steps: int = 300         # give up if still blocked for too long


class DoorController:
    def __init__(self, cfg: DoorConfig | None = None):
        self.cfg = cfg or DoorConfig()
        self.reset()

    def reset(self):
        self.door_streak = 0
        self.phase = "policy"            # policy | use | wait | escape
        self.timer = 0
        self.retries = 0
        self.cooldown = 0
        self.locked_steps = 0
        self.interventions = 0
        self.successful = 0
        self.failed = 0

    def _door_evidence(self, obs):
        hit = np.asarray(obs["rays"]["hit_type"]).reshape(-1)
        dist = np.asarray(obs["rays"]["distance"]).reshape(-1)
        door = np.isin(hit[list(FRONT)], (2, 3))
        near = door & (dist[list(FRONT)] < self.cfg.trigger_distance)
        enemies = np.any(hit[list(FRONT)] == 4)
        return bool(np.sum(near) >= self.cfg.min_door_rays), enemies

    def apply(self, macro_idx: int, obs, learned_signal: bool | None = None) -> int:
        c = self.cfg
        if self.cooldown > 0:
            self.cooldown -= 1
        door, enemies = self._door_evidence(obs)
        self.door_streak = self.door_streak + 1 if door else 0

        if self.phase == "policy":
            # Learned gate replaces the hand-written trigger when supplied; otherwise the
            # original observation rule is used (for teacher data collection only).
            trigger = learned_signal if learned_signal is not None else (
                door and not enemies and self.door_streak >= c.persist_steps)
            if trigger and self.cooldown == 0 and self.locked_steps < c.max_locked_steps:
                self.phase = "use"
                self.timer = 1
                self.interventions += 1
                self.door_streak = 0
                return USE_MACRO
            return macro_idx

        if self.phase == "use":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "wait"
                self.timer = c.wait_steps
            return USE_MACRO

        if self.phase == "wait":
            self.timer -= 1
            # Door evidence gone => interaction succeeded; release immediately.
            if not door:
                self.successful += 1
                self.phase = "policy"
                self.cooldown = c.cooldown_steps
                self.retries = 0
                self.locked_steps = 0
                return macro_idx
            if self.timer <= 0:
                self.retries += 1
                self.locked_steps += 1
                if self.retries >= c.max_retries:
                    self.failed += 1
                    self.phase = "escape"
                    self.timer = c.escape_turn_steps
                else:
                    self.phase = "use"
                    self.timer = 1
                return macro_idx
            return FORWARD_MACRO

        if self.phase == "escape":
            self.timer -= 1
            if self.timer <= 0:
                self.phase = "policy"
                self.cooldown = c.cooldown_steps
                self.retries = 0
            return TURN_RIGHT_MACRO

        return macro_idx
