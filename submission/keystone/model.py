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

        # Ray encoder: default MLP keeps champion checkpoint compatibility.
        self.ray_enc = _mlp(fb.ray_dim, H, 64)
        self.ray_encoder_mode = str(getattr(cfg, "ray_encoder", "mlp"))
        if self.ray_encoder_mode == "conv1d":
            # Input channels: distance + 6 hit-type onehot = 7, length 64 with circular wrap.
            self.ray_conv = nn.Sequential(
                nn.Conv1d(7, 32, kernel_size=5, padding=0), nn.ReLU(True),
                nn.Conv1d(32, 32, kernel_size=3, padding=0), nn.ReLU(True),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(32 * 8, 64), nn.Tanh(),
            )

        self.vitals_enc = _mlp(17, 32, 24)
        self.depth_enc = _mlp(fb.depth_dim, 64, 32)

        # Entity encoder: default pool keeps champion checkpoint compatibility.
        ent_in_dim = 5 + fb.ent_mask_dim // 32 + NUM_ENTITY_CLASSES
        self.ent_in = nn.Linear(ent_in_dim, D)
        self.ent_out = nn.Sequential(nn.Tanh(), nn.Linear(D, D), nn.Tanh())
        self.entity_encoder_mode = str(getattr(cfg, "entity_encoder", "pool"))
        if self.entity_encoder_mode == "attn":
            heads = int(getattr(cfg, "entity_attn_heads", 4))
            self.ent_attn = nn.MultiheadAttention(D, heads, batch_first=True)
            self.ent_attn_norm = nn.LayerNorm(D)

        enc_in = 64 + 24 + (32 if cfg.use_depth else 0) + D * 2
        self.use_prev_action = bool(getattr(cfg, "use_prev_action", False))
        self.prev_action_dim = 9 if self.use_prev_action else 0
        if self.use_prev_action:
            self.prev_action_enc = nn.Sequential(nn.Linear(9, 16), nn.Tanh())
            enc_in += 16
        self.encoder = _mlp(enc_in, H, H)
        self.gru = nn.GRUCell(H, H) if cfg.use_recurrence else None
        self.query = nn.Linear(H, D)

        self.memory = EventMemory(cfg, EVIDENCE_DIM)
        self.goal_head = nn.Linear(H + D, cfg.num_goals)
        self.goal_embed = nn.Parameter(torch.randn(cfg.num_goals, D) * 0.1)
        self.risk_head = nn.Sequential(nn.Linear(H + D, D), nn.Tanh(), nn.Linear(D, 1), nn.Sigmoid())

        self.action_dim = int(cfg.num_macros) if cfg.action_mode == "macro" else 9
        actor_in = H + H + D + D + (16 if self.use_prev_action else 0)
        self.actor = nn.Sequential(nn.Linear(actor_in, H), nn.Tanh(),
                                   nn.Linear(H, self.action_dim))
        self.critic = nn.Sequential(nn.Linear(H + H + D, H), nn.Tanh(), nn.Linear(H, 1))

        # Training-only auxiliary heads. Not used by inference unless explicitly requested.
        self.use_aux_heads = bool(getattr(cfg, "use_aux_heads", False))
        if self.use_aux_heads:
            self.aux_enemy_visible = nn.Linear(H, 1)
            self.aux_key_held = nn.Linear(H, 6)
            self.aux_door_in_view = nn.Linear(H, 1)
            self.aux_locked_door_in_view = nn.Linear(H, 1)

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

    def _encode_rays(self, rays: torch.Tensor) -> torch.Tensor:
        if self.ray_encoder_mode == "conv1d" and hasattr(self, "ray_conv"):
            # rays: (B, 64*(1+6)) = [dist(64), onehot(64*6)]
            B = rays.shape[0]
            dist = rays[:, :64].view(B, 1, 64)
            onehot = rays[:, 64:].view(B, 64, 6).permute(0, 2, 1)
            x = torch.cat([dist, onehot], dim=1)  # (B,7,64)
            # circular padding for wrap-around rays
            x = torch.cat([x[:, :, -2:], x, x[:, :, :2]], dim=2)
            x = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)  # total pad 3 each side for k5+k3
            return self.ray_conv(x)
        return self.ray_enc(rays)

    def _encode_entities(self, parts: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ent_in = torch.cat([parts["ent"], parts["etype"], parts["mask"].unsqueeze(-1)], dim=-1)
        e = self.ent_out(self.ent_in(ent_in))                        # (B,32,D)
        mask = parts["mask"] > 0                                     # (B,32)
        if self.entity_encoder_mode == "attn" and hasattr(self, "ent_attn"):
            # Query from global masked mean; attend over entity slots.
            denom = mask.unsqueeze(-1).to(e.dtype).sum(1).clamp(min=1.0)
            q = (e * mask.unsqueeze(-1).to(e.dtype)).sum(1) / denom
            q = q.unsqueeze(1)  # (B,1,D)
            attn_out, _ = self.ent_attn(q, e, e, key_padding_mask=~mask)
            ent_mean = self.ent_attn_norm(q + attn_out).squeeze(1)
            ent_max = e.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
            ent_max = torch.where(mask.any(1, keepdim=True), ent_max, torch.zeros_like(ent_max))
            return ent_mean, ent_max
        mask_f = mask.unsqueeze(-1).to(e.dtype)
        e = e * mask_f
        ent_mean = e.sum(1) / mask_f.sum(1).clamp(min=1.0)
        ent_max = e.max(1).values
        return ent_mean, ent_max

    def encode(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        ray = self._encode_rays(parts["rays"])
        vit = self.vitals_enc(parts["vitals"])
        ent_mean, ent_max = self._encode_entities(parts)
        pieces = [ray, vit, ent_mean, ent_max]
        if self.cfg.use_depth:
            pieces.append(self.depth_enc(parts["depth"].flatten(1)))
        if self.use_prev_action and "prev_action" in parts:
            pieces.append(self.prev_action_enc(parts["prev_action"].to(ray.dtype)))
        elif self.use_prev_action:
            pieces.append(ray.new_zeros(ray.shape[0], 16))
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

        actor_parts = [z, h, ctx_eff, goal_vec]
        if self.use_prev_action and "prev_action" in feats:
            actor_parts.append(self.prev_action_enc(feats["prev_action"].to(z.dtype)))
        elif self.use_prev_action:
            actor_parts.append(z.new_zeros(z.shape[0], 16))
        logits = self.actor(torch.cat(actor_parts, dim=-1))
        value = self.critic(torch.cat([z, h, ctx_eff], dim=-1)).squeeze(-1)

        aux = {}
        if self.use_aux_heads:
            aux["aux_enemy_visible"] = self.aux_enemy_visible(h).squeeze(-1)
            aux["aux_key_held"] = self.aux_key_held(h)
            aux["aux_door_in_view"] = self.aux_door_in_view(h).squeeze(-1)
            aux["aux_locked_door_in_view"] = self.aux_locked_door_in_view(h).squeeze(-1)

        new_mem = self.memory.write(state.mem, feats["ev"], event, q)
        new_state = AgentState(h, new_mem)
        out = {
            "logits": logits, "value": value, "goal_logits": goal_logits,
            "risk": conf.squeeze(-1), "reliability": reliability.squeeze(-1),
            "read_weights": weights,
        }
        out.update(aux)
        return out, new_state

    def forward_sequence(self, feats_seq: Dict[str, torch.Tensor], state0: AgentState,
                         events: torch.Tensor):
        """Unroll a T-step sequence for PPO gradient flow through recurrence and memory.

        feats_seq values have shape (T, B, ...); events (T, B). Returns outputs with a
        leading T dimension and the final state.
        """
        T = next(iter(feats_seq.values())).shape[0]
        B = state0.h.shape[0]
        state = state0
        out_keys = ["logits", "value", "goal_logits", "risk", "reliability", "read_weights"]
        if self.use_aux_heads:
            out_keys += ["aux_enemy_visible", "aux_key_held", "aux_door_in_view", "aux_locked_door_in_view"]
        outs = {k: [] for k in out_keys}
        for t in range(T):
            parts = {k: v[t] for k, v in feats_seq.items()}
            out, state = self.step(parts, state, events[t])
            for k in outs:
                if k in out:
                    outs[k].append(out[k])
        return {k: torch.stack(v, 0) for k, v in outs.items() if v}, state


NUM_ENTITY_CLASSES = 12

__all__ = ["KeyStoneNet", "AgentState", "Bernoulli"]
