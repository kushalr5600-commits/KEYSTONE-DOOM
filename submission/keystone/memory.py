"""Differentiable fixed-capacity progression-event memory.

A learned, slot-based key/value store. Every slot holds an event embedding, an event-type
distribution, a confidence and an age. Writes are content-addressed with softmax attention
(a soft, differentiable equivalent of "update the matching slot, otherwise allocate"),
confidence grows when repeated evidence agrees and decays with age, and retrieval is a
query-conditioned attention read. No absolute pose, map coordinates or hand-written map
solver is involved: the memory stores observation-derived event evidence only.

This is deliberately small (default 16 slots x 32 dims) so it runs comfortably inside the
40 ms/decision, 1-CPU, 2 GB-VRAM evaluation budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .config import KeyStoneConfig


@dataclass
class MemoryState:
    emb: torch.Tensor       # (B, K, D)
    etype: torch.Tensor     # (B, K, T)
    conf: torch.Tensor      # (B, K, 1)
    age: torch.Tensor       # (B, K, 1)

    def detach_clone(self) -> "MemoryState":
        return MemoryState(
            self.emb.detach().clone(),
            self.etype.detach().clone(),
            self.conf.detach().clone(),
            self.age.detach().clone(),
        )

    def reset(self, mask: torch.Tensor | None = None) -> "MemoryState":
        """Zero selected batch rows (used at episode boundaries)."""
        if mask is None:
            return MemoryState(
                torch.zeros_like(self.emb), torch.zeros_like(self.etype),
                torch.zeros_like(self.conf), torch.zeros_like(self.age),
            )
        m = mask.to(self.emb.dtype).view(-1, 1, 1)
        return MemoryState(
            self.emb * (1 - m), self.etype * (1 - m),
            self.conf * (1 - m), self.age * (1 - m),
        )


class EventMemory(nn.Module):
    def __init__(self, cfg: KeyStoneConfig, evidence_dim: int):
        super().__init__()
        self.K = int(cfg.memory_slots)
        self.D = int(cfg.embed)
        self.T = int(cfg.event_types)
        self.use_memory = bool(cfg.use_memory)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_dim, self.D), nn.Tanh(), nn.Linear(self.D, self.D)
        )
        self.write_gate = nn.Linear(self.D * 2, 1)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.conf_decay = 0.02
        self.age_cost = 0.01

    def initial(self, batch: int, device=None, dtype=torch.float32) -> MemoryState:
        return MemoryState(
            torch.zeros(batch, self.K, self.D, device=device, dtype=dtype),
            torch.zeros(batch, self.K, self.T, device=device, dtype=dtype),
            torch.zeros(batch, self.K, 1, device=device, dtype=dtype),
            torch.zeros(batch, self.K, 1, device=device, dtype=dtype),
        )

    def read(self, query: torch.Tensor, state: MemoryState):
        """Attention read. Returns (context D, read_weights K, reliability scalar)."""
        if not self.use_memory:
            z = torch.zeros(query.shape[0], self.D, device=query.device, dtype=query.dtype)
            return z, torch.zeros(query.shape[0], self.K, device=query.device), z.new_zeros(query.shape[0], 1)
        keys = state.emb / math.sqrt(max(self.D, 1))
        logits = torch.einsum("bd,bkd->bk", query, keys)
        logits = logits + torch.sigmoid(state.conf.squeeze(-1)) * 0.5 - state.age.squeeze(-1) * self.age_cost
        w = torch.softmax(logits / self.tau.clamp(0.25, 4.0), dim=-1)
        ctx = torch.einsum("bk,bkd->bd", w, state.emb)
        reliability = (w * torch.sigmoid(state.conf.squeeze(-1))).sum(-1, keepdim=True)
        return ctx, w, reliability

    def write(self, state: MemoryState, evidence: torch.Tensor, event: torch.Tensor,
              query: torch.Tensor) -> MemoryState:
        """Content-addressed soft write with confidence and age dynamics."""
        if not self.use_memory:
            return state
        cand = self.evidence_encoder(evidence)                     # (B,D)
        etoh = torch.nn.functional.one_hot(
            event.long().clamp(0, self.T - 1), self.T
        ).to(cand.dtype)                                            # (B,T)
        sim = torch.einsum("bd,bkd->bk", cand, state.emb) / math.sqrt(max(self.D, 1))
        agree = torch.einsum("bt,bkt->bk", etoh, state.etype)
        score = sim + 0.5 * agree + torch.log(state.conf.squeeze(-1) + 1e-3) \
            - state.age.squeeze(-1) * self.age_cost
        w = torch.softmax(score / self.tau.clamp(0.25, 4.0), dim=-1)
        gate = torch.sigmoid(self.write_gate(torch.cat([cand, query], dim=-1))).unsqueeze(-1)  # (B,1,1)
        delta = w.unsqueeze(-1) * gate                              # (B,K,1)
        emb = state.emb + delta * (cand.unsqueeze(1) - state.emb)
        etype = state.etype + delta * (etoh.unsqueeze(1) - state.etype)
        # Confidence rises on repeated/consistent evidence, decays slowly otherwise.
        conf = torch.clamp(state.conf * (1 - self.conf_decay) + delta * (1 - state.conf), 0.0, 1.0)
        age = (state.age + 1.0) * (1.0 - w.unsqueeze(-1))           # retrieval refreshes a slot
        return MemoryState(emb, etype, conf, age)
