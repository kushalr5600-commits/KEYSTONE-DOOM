"""KEYSTONE policy network: structured encoder -> recurrent core -> event memory -> goals -> actions.

Every decision is produced by trained weights. The goal head is a learned latent intention
conditioning the actor; it is not a hand-written if/else tree. Legal-action post-processing
(keystone.actions.legalize_action) only enforces contract button semantics (edge-triggered
USE / weapon select and mutually exclusive movement buttons); it never chooses objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.distributions import Bernoulli

from .config import KeyStoneConfig
from .features import EVIDENCE_DIM, FeatureBuilder
from .memory import EventMemory, MemoryState


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim), nn.Tanh(),
    )


@dataclass
class AgentState:
    h: torch.Tensor
    mem: MemoryState

    def detach_clone(self) -> "AgentState":
        return AgentState(self.h.detach().clone(), self.mem.detach_clone())

    def reset(self, done: torch.Tensor | None = None) -> "AgentState":
        h = self.h if done is None else self.h * (1.0 - done.to(self.h.dtype).unsqueeze(-1))
        if done is None:
            h = torch.zeros_like(self.h)
        return AgentState(h, self.mem.reset(done))


class KeyStoneNet(nn.Module):
    def __init__(self, cfg: KeyStoneConfig, feature_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        fb = FeatureBuilder(cfg)
        self.feature_dim = int(feature_dim or fb.total_dim)
        H, D = int(cfg.hidden), int(cfg.embed)

        self.ray_enc = _mlp(fb.ray_dim, H, 64)
        self.vitals_enc = _mlp(17, 32, 24)
        self.depth_enc = _mlp(fb.depth_dim, 64, 32)
        self.ent_in = nn.Linear(5 + fb.ent_mask_dim // 32 + NUM_ENTITY_CLASSES, D)
        self.ent_out = nn.Sequential(nn.Tanh(), nn.Linear(D, D), nn.Tanh())

        enc_in = 64 + 24 + (32 if cfg.use_depth else 0) + D * 2
        self.encoder = _mlp(enc_in, H, H)
        self.gru = nn.GRUCell(H, H) if cfg.use_recurrence else None
        self.query = nn.Linear(H, D)

        self.memory = EventMemory(cfg, EVIDENCE_DIM)
        self.goal_head = nn.Linear(H + D, cfg.num_goals)
        self.goal_embed = nn.Parameter(torch.randn(cfg.num_goals, D) * 0.1)
        self.risk_head = nn.Sequential(nn.Linear(H + D, D), nn.Tanh(), nn.Linear(D, 1), nn.Sigmoid())

        actor_in = H + H + D + D
        self.actor = nn.Sequential(nn.Linear(actor_in, H), nn.Tanh(), nn.Linear(H, 9))
        self.critic = nn.Sequential(nn.Linear(H + H + D, H), nn.Tanh(), nn.Linear(H, 1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> AgentState:
        return AgentState(
            torch.zeros(batch, self.cfg.hidden, device=device, dtype=dtype),
            self.memory.initial(batch, device=device, dtype=dtype),
        )

    def encode(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        ray = self.ray_enc(parts["rays"])
        vit = self.vitals_enc(parts["vitals"])
        ent_in = torch.cat([parts["ent"], parts["etype"], parts["mask"].unsqueeze(-1)], dim=-1)
        e = self.ent_out(self.ent_in(ent_in))                        # (B,32,D)
        mask = parts["mask"].unsqueeze(-1)                           # (B,32,1)
        e = e * mask
        ent_mean = e.sum(1) / mask.sum(1).clamp(min=1.0)
        ent_max = e.max(1).values
        pieces = [ray, vit, ent_mean, ent_max]
        if self.cfg.use_depth:
            pieces.append(self.depth_enc(parts["depth"].flatten(1)))
        return self.encoder(torch.cat(pieces, dim=-1))

    def step(self, feats: Dict[str, torch.Tensor], state: AgentState,
             event: torch.Tensor, query_override: torch.Tensor | None = None):
        """One decision. ``event`` is an observation-derived event-type id, shape (B,)."""
        z = self.encode(feats)
        if self.gru is not None:
            h = self.gru(z, state.h)
        else:
            h = torch.tanh(z)
        q = self.query(h) if query_override is None else query_override
        ctx, weights, reliability = self.memory.read(q, state.mem)
        if self.cfg.use_confidence:
            conf = self.risk_head(torch.cat([h, ctx], dim=-1))
            ctx_eff = ctx * conf
        else:
            conf = torch.ones(h.shape[0], 1, device=h.device)
            ctx_eff = ctx

        goal_logits = self.goal_head(torch.cat([h, ctx_eff], dim=-1))
        if self.cfg.use_goal:
            goal_w = torch.softmax(goal_logits, dim=-1)
            goal_vec = torch.einsum("bg,gd->bd", goal_w, self.goal_embed)
        else:
            goal_vec = ctx_eff.new_zeros(h.shape[0], self.cfg.embed)

        logits = self.actor(torch.cat([z, h, ctx_eff, goal_vec], dim=-1))
        value = self.critic(torch.cat([z, h, ctx_eff], dim=-1)).squeeze(-1)

        new_mem = self.memory.write(state.mem, feats["ev"], event, q)
        new_state = AgentState(h, new_mem)
        return {
            "logits": logits, "value": value, "goal_logits": goal_logits,
            "risk": conf.squeeze(-1), "reliability": reliability.squeeze(-1),
            "read_weights": weights,
        }, new_state

    def forward_sequence(self, feats_seq: Dict[str, torch.Tensor], state0: AgentState,
                         events: torch.Tensor):
        """Unroll a T-step sequence for PPO gradient flow through recurrence and memory.

        feats_seq values have shape (T, B, ...); events (T, B). Returns outputs with a
        leading T dimension and the final state.
        """
        T = next(iter(feats_seq.values())).shape[0]
        B = state0.h.shape[0]
        state = state0
        outs = {k: [] for k in ("logits", "value", "goal_logits", "risk", "reliability", "read_weights")}
        for t in range(T):
            parts = {k: v[t] for k, v in feats_seq.items()}
            out, state = self.step(parts, state, events[t])
            for k in outs:
                outs[k].append(out[k])
        return {k: torch.stack(v, 0) for k, v in outs.items()}, state


NUM_ENTITY_CLASSES = 12

__all__ = ["KeyStoneNet", "AgentState", "Bernoulli"]
