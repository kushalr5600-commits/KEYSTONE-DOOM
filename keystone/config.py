"""Hyperparameters for the KEYSTONE architecture.

These are our own choices; they are NOT official competition constants. Official limits
live in env/doom_env/limits.py and are summarised in docs/official-spec-summary.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class KeyStoneConfig:
    # Architecture
    hidden: int = 128
    embed: int = 32
    memory_slots: int = 16
    event_types: int = 8
    num_goals: int = 7
    depth_h: int = 12
    depth_w: int = 16
    dropout: float = 0.0

    # PPO
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    goal_entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    update_epochs: int = 4
    num_minibatches: int = 4
    sequence_length: int = 16
    rollout_length: int = 64
    num_envs: int = 4

    # Shaping (training only; never used by the deployed agent)
    reward_exit: float = 10.0
    reward_death: float = -3.0
    # Training-only distance-progress shaping toward the exit: k*(d_prev-d_now)/128.
    # Zero for standing still / loops; never part of the deployed observation.
    exit_potential: float = 0.5
    reward_key: float = 1.0
    reward_unlock: float = 0.5
    reward_new_signature: float = 0.02
    reward_kill: float = 0.05
    damage_penalty: float = 0.005
    time_penalty: float = 0.002

    seed: int = 0
    total_steps: int = 50_000
    log_interval: int = 5
    save_interval: int = 20
    eval_episodes: int = 6

    # Ablation switches: disable components without changing the observation interface.
    use_memory: bool = True
    use_recurrence: bool = True
    use_goal: bool = True
    use_confidence: bool = True
    # Depth/RGB branch is off by default: profile showed it costs >2x step time on 1 CPU
    # while rays already carry the navigation signal. Documented ablation: docs/experiment-plan.md.
    use_depth: bool = False

    tags: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "KeyStoneConfig":
        return cls(**json.loads(text))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KeyStoneConfig":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
