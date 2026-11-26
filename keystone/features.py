"""Observation featurisation. Pure numpy, observation-only, identical in train and inference.

The official observation is a Dict with keys rgb/depth/vitals/rays/visible_entities
(env/doom_env/spaces.py:73-95). We use depth (cheap, downsampled), vitals, rays and the
entity set. RGB is deliberately left out of the default encoder: on 1 CPU thread a
small structured model is far faster, and the contract explicitly recommends ego-centric
spatial signals. An RGB/depth-hybrid is a documented possible ablation.

Nothing in this module reads privileged state, map files, the engine, the network or
absolute pose. All quantities are ego-centric or per-episode observations.
"""

from __future__ import annotations

import numpy as np

from .config import KeyStoneConfig

NUM_RAYS = 64
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
VITALS_DIM = 17

EVIDENCE_DIM = 16
EVENT_NONE = 0
EVENT_LOCKED_DOOR = 1
EVENT_DOOR = 2
EVENT_KEY = 3
EVENT_ITEM = 4
EVENT_ENEMY = 5
EVENT_INTERACT_OK = 6
EVENT_INTERACT_FAIL = 7
EVENT_NAMES = (
    "none", "locked_door", "door", "key", "item", "enemy",
    "interact_ok", "interact_fail",
)


class FeatureBuilder:
    """Turn one official observation into a fixed-size float32 vector.

    Returns ``(vec, evidence, event_type)``. ``evidence`` and ``event_type`` are the
    observation-derived progression signals written into event memory. The caller keeps
    them for one step so interaction outcomes can be tagged.
    """

    def __init__(self, cfg: KeyStoneConfig):
        self.cfg = cfg
        self.depth_h = int(cfg.depth_h)
        self.depth_w = int(cfg.depth_w)
        self.ray_dim = NUM_RAYS * (1 + NUM_HIT_TYPES)
        self.depth_dim = self.depth_h * self.depth_w
        self.ent_dim = MAX_ENTITIES * 5
        self.ent_type_dim = MAX_ENTITIES * NUM_ENTITY_TYPES
        self.ent_mask_dim = MAX_ENTITIES
        self.total_dim = (
            self.ray_dim + VITALS_DIM + self.depth_dim
            + self.ent_dim + self.ent_type_dim + self.ent_mask_dim + EVIDENCE_DIM
        )

    def __call__(self, obs: dict, prev_evidence: np.ndarray | None = None,
                 prev_action: np.ndarray | None = None):
        # --- rays: distance + one-hot hit type -----------------------------------------
        dist = np.asarray(obs["rays"]["distance"], dtype=np.float32).reshape(NUM_RAYS)
        hit = np.asarray(obs["rays"]["hit_type"], dtype=np.int64).reshape(NUM_RAYS)
        onehot = np.eye(NUM_HIT_TYPES, dtype=np.float32)[np.clip(hit, 0, NUM_HIT_TYPES - 1)]
        ray_vec = np.concatenate([np.clip(dist, 0.0, 1.0), onehot.ravel()])

        # --- vitals: fixed normalisation; keys stay binary ------------------------------
        vitals = np.asarray(obs["vitals"], dtype=np.float32).reshape(VITALS_DIM)
        v = np.empty(VITALS_DIM, dtype=np.float32)
        v[0] = np.clip(vitals[0] / 100.0, 0.0, 2.0)
        v[1] = np.clip(vitals[1] / 100.0, 0.0, 2.0)
        v[2] = np.clip(vitals[2] / 7.0, 0.0, 1.0)
        v[3:8] = np.clip(vitals[3:8] / 100.0, 0.0, 2.0)
        v[8:14] = np.clip(vitals[8:14], 0.0, 1.0)
        v[14:17] = np.clip(vitals[14:17] / 8.0, -2.0, 2.0)

        # --- depth: block-average downsample (pure numpy, no cv2 dependency) -----------
        depth = np.asarray(obs["depth"], dtype=np.float32)
        if depth.ndim == 3:
            depth = depth.mean(axis=-1)
        if depth.shape != (120, 160):
            depth = np.zeros((120, 160), dtype=np.float32)
        bh, bw = 120 // self.depth_h, 160 // self.depth_w
        dsmall = depth.reshape(self.depth_h, bh, self.depth_w, bw).mean(axis=(1, 3))
        dvec = (dsmall / 255.0).ravel().astype(np.float32)

        # --- entities: features / one-hot class / mask ---------------------------------
        ent = np.asarray(obs["visible_entities"]["features"], dtype=np.float32).reshape(MAX_ENTITIES, 5)
        etype = np.asarray(obs["visible_entities"]["entity_type"], dtype=np.int64).reshape(MAX_ENTITIES)
        mask = np.asarray(obs["visible_entities"]["mask"], dtype=np.float32).reshape(MAX_ENTITIES)
        ent = np.nan_to_num(ent, nan=0.0, posinf=2.0, neginf=-2.0)
        etoh = np.eye(NUM_ENTITY_TYPES, dtype=np.float32)[np.clip(etype, 0, NUM_ENTITY_TYPES - 1)]

        # --- progression evidence: all from the current observation --------------------
        ev = np.zeros(EVIDENCE_DIM, dtype=np.float32)
        ev[0:6] = v[8:14]
        locked = hit == 3
        doors = hit == 2
        items = (hit == 5).sum() + int((np.isin(etype, (5, 6, 7, 8, 9)) & (mask > 0)).sum())
        enemies = (hit == 4).sum() + int((np.isin(etype, (1, 2, 3, 10)) & (mask > 0)).sum())
        ev[6] = min(float(locked.sum()) / 8.0, 1.0)
        ev[7] = float(dist[locked].min()) if locked.any() else 1.0
        ev[8] = float(np.flatnonzero(locked)[0]) / 63.0 if locked.any() else 1.0
        ev[9] = min(float(doors.sum()) / 8.0, 1.0)
        ev[10] = min(float(items) / 8.0, 1.0)
        ev[11] = min(float(enemies) / 8.0, 1.0)
        ev[12] = v[0]
        ev[13] = v[1]
        ev[14] = 1.0 if prev_action is not None and int(prev_action[7]) else 0.0
        ev[15] = float(np.clip(vitals[3:8].sum() / 200.0, 0.0, 2.0))

        vec = np.concatenate([
            ray_vec, v, dvec.ravel(), ent.ravel(), etoh.ravel(), mask, ev,
        ]).astype(np.float32)
        assert vec.shape == (self.total_dim,), (vec.shape, self.total_dim)

        event = self._event(ev, prev_evidence, prev_action)
        return vec, ev, event

    @staticmethod
    def _event(ev: np.ndarray, prev_ev: np.ndarray | None,
               prev_action: np.ndarray | None) -> int:
        if prev_ev is not None and np.any(ev[0:6] > prev_ev[0:6] + 0.5):
            return EVENT_KEY
        used = prev_action is not None and int(prev_action[7]) == 1
        if used and prev_ev is not None and prev_ev[6] > 0.0:
            if ev[6] < prev_ev[6] - 0.01:
                return EVENT_INTERACT_OK
            if ev[7] >= 0.99 and prev_ev[7] < 0.99:
                return EVENT_INTERACT_OK
            return EVENT_INTERACT_FAIL
        if ev[6] > 0.0:
            return EVENT_LOCKED_DOOR
        if ev[9] > 0.0:
            return EVENT_DOOR
        if ev[11] > 0.0:
            return EVENT_ENEMY
        if ev[10] > 0.0:
            return EVENT_ITEM
        return EVENT_NONE

    def split(self, vec_batch):
        """Split a batch feature tensor into its typed components (torch tensors)."""
        import torch

        x = vec_batch
        i = 0
        rays = x[:, i:i + self.ray_dim]; i += self.ray_dim
        vitals = x[:, i:i + VITALS_DIM]; i += VITALS_DIM
        depth = x[:, i:i + self.depth_dim]; i += self.depth_dim
        ent = x[:, i:i + self.ent_dim]; i += self.ent_dim
        etype = x[:, i:i + self.ent_type_dim]; i += self.ent_type_dim
        mask = x[:, i:i + self.ent_mask_dim]; i += self.ent_mask_dim
        ev = x[:, i:i + EVIDENCE_DIM]
        return {
            "rays": rays, "vitals": vitals,
            "depth": depth.view(-1, self.depth_h, self.depth_w),
            "ent": ent.view(-1, MAX_ENTITIES, 5),
            "etype": etype.view(-1, MAX_ENTITIES, NUM_ENTITY_TYPES),
            "mask": mask.view(-1, MAX_ENTITIES),
            "ev": ev,
        }


FEATURE_DIM = (
    NUM_RAYS * (1 + NUM_HIT_TYPES) + VITALS_DIM + 12 * 16
    + MAX_ENTITIES * 5 + MAX_ENTITIES * NUM_ENTITY_TYPES + MAX_ENTITIES + EVIDENCE_DIM
)
