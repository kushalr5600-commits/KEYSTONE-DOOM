"""PPO with sequence replay through the recurrent core and event memory.

Implementation is our own (no Stable-Baselines3). Rollouts are collected from the official
doom_env wrapper; training-time reward shaping uses privileged_state() only for labels --
the deployed agent never sees it. Sequence minibatches restore stored hidden/memory states
so gradients flow through recurrence and memory writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

from keystone import AgentState, KeyStoneConfig, KeyStoneNet, legalize_action


@dataclass
class Rollout:
    feats: np.ndarray       # (T,N,F)
    events: np.ndarray      # (T,N)
    h0: np.ndarray          # (T,N,H)
    mem_emb: np.ndarray     # (T,N,K,D)
    mem_etype: np.ndarray   # (T,N,K,T)
    mem_conf: np.ndarray    # (T,N,K,1)
    mem_age: np.ndarray     # (T,N,K,1)
    actions: np.ndarray     # (T,N,9)
    prev_actions: np.ndarray
    old_logp: np.ndarray    # (T,N)
    values: np.ndarray      # (T,N)
    rewards: np.ndarray     # (T,N)
    dones: np.ndarray       # (T,N)
    ep_ids: np.ndarray      # (T,N)
    last_h: np.ndarray      # (N,H)
    last_mem: tuple
    last_value: np.ndarray  # (N,)

    @property
    def T(self):
        return self.feats.shape[0]

    @property
    def N(self):
        return self.feats.shape[1]


def compute_gae(rew: np.ndarray, val: np.ndarray, done: np.ndarray,
                last_val: np.ndarray, gamma: float, lam: float):
    T, N = rew.shape
    adv = np.zeros((T, N), np.float32)
    ret = np.zeros((T, N), np.float32)
    gae = np.zeros(N, np.float32)
    next_v = last_val.copy()
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = rew[t] + gamma * next_v * mask - val[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        ret[t] = gae + val[t]
        next_v = val[t]
    return adv, ret


class RewardShaper:
    """Training-only reward shaping. Privileged state is used as a label source here only."""

    def __init__(self, cfg: KeyStoneConfig):
        self.cfg = cfg
        self.signatures: list[set] = []
        self.exits: list = []

    def reset(self, n: int):
        self.signatures = [set() for _ in range(n)]
        if len(self.exits) != n:
            self.exits = [None] * n

    def __call__(self, env_i: int, prev_priv: dict, priv: dict,
                 prev_dist: np.ndarray, action: np.ndarray) -> float:
        c = self.cfg
        r = 0.0
        if priv["exit_reached"]:
            r += c.reward_exit
        if priv["dead"]:
            r += c.reward_death
        # Potential-based shaping toward the exit: F = gamma*Phi(s') - Phi(s),
        # Phi(s) = -exit_potential * distance(s, exit)/128. Training labels only.
        ex = self.exits[env_i] if env_i < len(self.exits) else None
        if ex is not None:
            d0 = math.hypot(prev_priv["x"] - ex[0], prev_priv["y"] - ex[1])
            d1 = math.hypot(priv["x"] - ex[0], priv["y"] - ex[1])
            r += c.exit_potential * (d0 - c.gamma * d1) / 128.0
        keys = np.asarray(priv["keys"], dtype=np.float32)
        prev_keys = np.asarray(prev_priv["keys"], dtype=np.float32)
        r += c.reward_key * float(np.sum(keys > prev_keys + 0.5))
        # A locked door disappearing while a key is held is the observable unlock event.
        locked_before = int(np.sum(prev_dist == 3))
        locked_after = int(np.sum(priv.get("locked_rays", np.zeros(64)) == 3))
        if locked_before > 0 and locked_after < locked_before and np.any(keys > 0.5):
            r += c.reward_unlock
        r += c.reward_kill * max(0, int(priv["kills"]) - int(prev_priv["kills"]))
        r -= c.damage_penalty * max(0.0, float(priv["damage_taken"]) - float(prev_priv["damage_taken"]))
        # Small observation-only novelty term. Never enough to outweigh exit/progression.
        sig = bytes(np.clip((prev_dist * 16).astype(np.uint8), 0, 255))
        if sig not in self.signatures[env_i]:
            self.signatures[env_i].add(sig)
            r += c.reward_new_signature
        r -= c.time_penalty
        return float(r)


class PPOTrainer:
    def __init__(self, model: KeyStoneNet, optimizer: torch.optim.Optimizer, cfg: KeyStoneConfig):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def collect(self, envs, obs_list, states: AgentState, fb, prev_evs, prev_actions,
                ep_ids: np.ndarray, shaper: RewardShaper, cfg: KeyStoneConfig):
        T, N = int(cfg.rollout_length), len(envs)
        F = fb.total_dim
        H = cfg.hidden
        K, D, NT = cfg.memory_slots, cfg.embed, cfg.event_types
        buf = Rollout(
            np.zeros((T, N, F), np.float32), np.zeros((T, N), np.int64),
            np.zeros((T, N, H), np.float32),
            np.zeros((T, N, K, D), np.float32), np.zeros((T, N, K, NT), np.float32),
            np.zeros((T, N, K, 1), np.float32), np.zeros((T, N, K, 1), np.float32),
            np.zeros((T, N, 9), np.int8), np.zeros((T, N, 9), np.int8),
            np.zeros((T, N), np.float32), np.zeros((T, N), np.float32),
            np.zeros((T, N), np.float32), np.zeros((T, N), np.float32),
            np.zeros((T, N), np.int64),
            np.zeros((N, H), np.float32),
            (np.zeros((N, K, D), np.float32), np.zeros((N, K, NT), np.float32),
             np.zeros((N, K, 1), np.float32), np.zeros((N, K, 1), np.float32)),
            np.zeros((N,), np.float32),
        )
        prev_privs = [e.privileged_state() for e in envs]
        self.model.eval()
        with torch.no_grad():
            for t in range(T):
                # Current state snapshots BEFORE the step (for sequence replay).
                buf.h0[t] = states.h.numpy()
                buf.mem_emb[t] = states.mem.emb.numpy()
                buf.mem_etype[t] = states.mem.etype.numpy()
                buf.mem_conf[t] = states.mem.conf.numpy()
                buf.mem_age[t] = states.mem.age.numpy()
                feats = np.zeros((N, F), np.float32)
                events = np.zeros(N, np.int64)
                for i, env in enumerate(envs):
                    vec, ev, event = fb(obs_list[i], prev_evs[i], prev_actions[i])
                    feats[i], events[i] = vec, event
                x = torch.from_numpy(feats)
                parts = fb.split(x)
                out, new_state = self.model.step(parts, states, torch.from_numpy(events))
                dist = Bernoulli(logits=out["logits"])
                sample = dist.sample().numpy().astype(np.int8)
                actions = np.stack([
                    legalize_action(sample[i], prev_actions[i]) for i in range(N)
                ])
                logp = dist.log_prob(torch.from_numpy(actions.astype(np.float32))).sum(-1).numpy()
                buf.feats[t], buf.events[t] = feats, events
                buf.actions[t], buf.prev_actions[t] = actions, prev_actions
                buf.old_logp[t] = logp
                buf.values[t] = out["value"].numpy()
                buf.ep_ids[t] = ep_ids

                for i, env in enumerate(envs):
                    priv_before = prev_privs[i]
                    obs, reward, term, trunc, info = env.step(actions[i])
                    obs_list[i] = obs
                    priv = env.privileged_state()
                    dist_before = np.argmax(feats[i, 64:448].reshape(64, 6), axis=1).astype(np.int64)
                    priv["locked_rays"] = np.asarray(obs["rays"]["hit_type"], dtype=np.int64).reshape(64)
                    shaped = shaper(i, priv_before, priv, dist_before, actions[i])
                    buf.rewards[t, i] = shaped
                    buf.dones[t, i] = float(term or trunc)
                    if term or trunc:
                        ep_ids[i] += 1
                        obs = env.reset(seed=int(cfg.seed + i * 100003 + ep_ids[i] * 7919) % (2**31))[0]
                        obs_list[i] = obs
                        prev_evs[i] = np.zeros(16, np.float32)
                        prev_actions[i] = np.zeros(9, np.int8)
                        prev_privs[i] = env.privileged_state()
                        shaper.signatures[i].clear()
                        continue
                    prev_evs[i] = ev
                    prev_actions[i] = actions[i].copy()
                    prev_privs[i] = priv

                done_mask = torch.from_numpy(buf.dones[t].astype(np.float32))
                states = new_state if t == T - 1 else new_state.reset(done_mask)
                if t == T - 1:
                    buf.last_h = states.h.numpy().copy()
                    buf.last_mem = tuple(x.numpy().copy() for x in (
                        states.mem.emb, states.mem.etype, states.mem.conf, states.mem.age))
        # Bootstrap value from the final state.
        with torch.no_grad():
            last_feats = torch.from_numpy(feats)
            out, _ = self.model.step(fb.split(last_feats), states, torch.from_numpy(events))
            buf.last_value = out["value"].numpy().copy()
        return buf, states, prev_evs, prev_actions, ep_ids

    def update(self, buf: Rollout, fb) -> dict:
        """PPO update with sequence minibatches over the flattened (time, env) rollout."""
        from keystone.memory import MemoryState

        cfg = self.cfg
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, buf.last_value,
                               cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        device = next(self.model.parameters()).device
        T, N = buf.T, buf.N
        flat = T * N
        L = min(int(cfg.sequence_length), T)
        starts_all = np.arange(0, flat - L + 1, max(1, L // 2))
        rng = np.random.default_rng(cfg.seed + int(time.time() * 1000) % 100000)
        metrics = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "goal_ent": 0.0, "n": 0}

        f = buf.feats.reshape(flat, -1)
        ev_flat = buf.events.reshape(flat)
        act_flat = buf.actions.reshape(flat, 9)
        lp_flat = buf.old_logp.reshape(flat)
        adv_flat = adv.reshape(flat)
        ret_flat = ret.reshape(flat)
        ep_flat = buf.ep_ids.reshape(flat)
        h_flat = buf.h0.reshape(flat, cfg.hidden)
        me, mt, mc, ma = (buf.mem_emb.reshape(flat, cfg.memory_slots, cfg.embed),
                          buf.mem_etype.reshape(flat, cfg.memory_slots, cfg.event_types),
                          buf.mem_conf.reshape(flat, cfg.memory_slots, 1),
                          buf.mem_age.reshape(flat, cfg.memory_slots, 1))

        bounds, i = {}, 0
        for key, width in (
            ("rays", fb.ray_dim), ("vitals", 17), ("depth", fb.depth_dim),
            ("ent", 160), ("etype", 384), ("mask", 32), ("ev", 16),
        ):
            bounds[key] = (i, i + width)
            i += width

        def to_dev(a):
            return torch.as_tensor(np.ascontiguousarray(a), device=device)

        for _ in range(int(cfg.update_epochs)):
            rng.shuffle(starts_all)
            for chunk in np.array_split(starts_all, int(cfg.num_minibatches)):
                if len(chunk) == 0:
                    continue
                M = len(chunk)
                idx = chunk[:, None] + np.arange(L)[None, :]        # (M,L) flattened indices
                seq_feats = {}
                for key, (a, b) in bounds.items():
                    seq_feats[key] = to_dev(f[idx, a:b]).transpose(0, 1).contiguous()
                seq_feats["depth"] = seq_feats["depth"].view(L, M, cfg.depth_h, cfg.depth_w)
                seq_feats["ent"] = seq_feats["ent"].view(L, M, 32, 5)
                seq_feats["etype"] = seq_feats["etype"].view(L, M, 32, 12)
                seq_feats["mask"] = seq_feats["mask"].view(L, M, 32)

                events = to_dev(ev_flat[idx]).long().transpose(0, 1)
                actions = to_dev(act_flat[idx]).float().transpose(0, 1)
                old_logp = to_dev(lp_flat[idx]).transpose(0, 1)
                adv_mb = to_dev(adv_flat[idx]).transpose(0, 1)
                ret_mb = to_dev(ret_flat[idx]).transpose(0, 1)
                # Mask steps crossing an episode or environment boundary.
                keep = (ep_flat[idx] == ep_flat[idx][:, :1]) & ((idx % N) == (idx % N)[:, :1])
                mask = to_dev(keep.astype(np.float32)).transpose(0, 1)

                state0 = AgentState(
                    to_dev(h_flat[chunk]),
                    MemoryState(to_dev(me[chunk]), to_dev(mt[chunk]),
                                to_dev(mc[chunk]), to_dev(ma[chunk])),
                )
                out, _ = self.model.forward_sequence(seq_feats, state0, events)
                dist = Bernoulli(logits=out["logits"])
                logp = dist.log_prob(actions).sum(-1)
                ratio = torch.exp(logp - old_logp)
                obj = torch.min(ratio * adv_mb,
                                torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * adv_mb)
                denom = mask.sum().clamp(min=1.0)
                policy_loss = -(obj * mask).sum() / denom
                value_loss = ((out["value"] - ret_mb).pow(2) * mask).sum() / denom
                entropy = (dist.entropy().sum(-1) * mask).sum() / denom
                gd = torch.distributions.Categorical(logits=out["goal_logits"])
                goal_ent = (gd.entropy() * mask).sum() / denom
                loss = (policy_loss + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy - cfg.goal_entropy_coef * goal_ent)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                metrics["policy"] += float(policy_loss.detach())
                metrics["value"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                metrics["goal_ent"] += float(goal_ent.detach())
                metrics["n"] += 1
        for k in metrics:
            if k != "n":
                metrics[k] /= max(1, metrics["n"])
        return metrics

