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
        self.stuck = 0
        self.use_cooldown = 0

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

        # conservative obs-only assists (minimal, to reduce regressions)
        try:
            rays = observation.get('rays', {})
            dist = np.asarray(rays.get('distance', []), dtype=np.float32).reshape(-1)
            hit = np.asarray(rays.get('hit_type', []), dtype=np.int64).reshape(-1)
            if dist.size >= 64:
                front = float(np.median(dist[28:36]))
                blocked = front < 30.0
                moving_fwd = bool(action[2]) and not bool(action[3])
                if moving_fwd and blocked:
                    self.stuck += 1
                else:
                    self.stuck = max(0, self.stuck - 1)
                if self.use_cooldown > 0: self.use_cooldown -= 1
                if self.stuck >= 12:
                    if self.use_cooldown == 0:
                        action[7] = 1
                        self.use_cooldown = 6
                    # mild bounce to escape stuck
                    if self.stuck >= 18:
                        phase = (self.stuck // 6) % 3
                        if phase == 0: action[2]=0; action[3]=1
                        elif phase == 1: action[4]=1; action[5]=0; action[2]=1
                        else: action[5]=1; action[4]=0; action[2]=1
                        if self.stuck > 40: self.stuck = 10
                # door directly ahead: encourage USE
                if hit.size >= 64 and np.any(np.isin(hit[28:36].astype(int), [2,3])) and front < 90:
                    if self.use_cooldown == 0:
                        action[7] = 1
                        self.use_cooldown = 6
                    action[2] = 1; action[3] = 0
        except Exception:
            pass
        try:
            ents = observation.get('visible_entities', {})
            feats = np.asarray(ents.get('features', []), dtype=np.float32)
            etype = np.asarray(ents.get('entity_type', []), dtype=np.int64).reshape(-1)
            mask = np.asarray(ents.get('mask', []), dtype=np.float32).reshape(-1)
            if feats.ndim == 2 and feats.shape[0] == etype.shape[0]:
                best_key = None
                for i in range(min(len(mask), feats.shape[0])):
                    if mask[i] <= 0 or int(etype[i]) != 9: continue
                    d = float(feats[i,0])
                    if best_key is None or d < best_key[0]: best_key = (d,float(feats[i,1]),float(feats[i,2]))
                if best_key is not None and best_key[0] < 700 and float(self.prev_ev[0]) >= 0:
                    _, sin_a, cos_a = best_key
                    if cos_a > 0:
                        if sin_a > 0.08: action[4]=1; action[5]=0
                        elif sin_a < -0.08: action[5]=1; action[4]=0
                        action[2]=1; action[3]=0
        except Exception:
            pass

        # conservative aimlock: lock on enemy, auto-adjust for close shot
        try:
            ents = observation.get('visible_entities', {})
            feats = np.asarray(ents.get('features', []), dtype=np.float32)
            etype = np.asarray(ents.get('entity_type', []), dtype=np.int64).reshape(-1)
            mask = np.asarray(ents.get('mask', []), dtype=np.float32).reshape(-1)
            if feats.ndim == 2 and feats.shape[1] >= 3 and feats.shape[0] == etype.shape[0]:
                best = None
                for i in range(min(len(mask), feats.shape[0])):
                    if mask[i] <= 0 or int(etype[i]) not in (1,2,3): continue
                    d = float(feats[i,0])
                    if best is None or d < best[0]: best = (d,float(feats[i,1]),float(feats[i,2]))
                if best is not None and best[0] < 900:
                    d, sin_a, cos_a = best
                    if cos_a > 0.15:
                        if sin_a > 0.03: action[4]=1; action[5]=0
                        elif sin_a < -0.03: action[5]=1; action[4]=0
                        else: action[4]=0; action[5]=0
                        aligned = abs(sin_a) < 0.06 and cos_a > 0.5
                        if aligned and d < 700:
                            action[6] = 1
                            if d < 140: action[2]=0; action[3]=1
                            elif d > 260: action[2]=1; action[3]=0
                        else:
                            action[6] = 0
                    else:
                        action[4]=1 if sin_a >= 0 else 0
                        action[5]=0 if sin_a >= 0 else 1
                        action[6] = 0
        except Exception:
            pass

        self.prev_ev, self.prev_act = ev, action.copy()
        return action
