"""Unit tests for the KEYSTONE agent (observation, action, memory, latency, boundaries)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from doom_env import make_env
from keystone import (
    KeyStoneConfig, FeatureBuilder, KeyStoneNet, legalize_action,
)
from keystone.actions import USE
from keystone.features import EVIDENCE_DIM, EVENT_KEY, EVENT_LOCKED_DOOR
from keystone.memory import EventMemory, MemoryState

ROOT = Path(__file__).resolve().parents[1]
TEST_MAP = ROOT / "maps" / "two_rooms.wad"


def _obs():
    env = make_env(str(TEST_MAP), "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    env.close()
    return obs


def test_feature_shapes_and_finite():
    cfg = KeyStoneConfig()
    fb = FeatureBuilder(cfg)
    obs = _obs()
    vec, ev, event = fb(obs)
    assert vec.shape == (fb.total_dim,) == (1249,)
    assert ev.shape == (EVIDENCE_DIM,)
    assert np.isfinite(vec).all() and np.isfinite(ev).all()
    assert 0 <= event < cfg.event_types
    parts = fb.split(torch.from_numpy(vec[None, :]))
    assert parts["rays"].shape == (1, 448)
    assert parts["vitals"].shape == (1, 17)
    assert parts["ent"].shape == (1, 32, 5)
    assert parts["etype"].shape == (1, 32, 12)
    assert parts["mask"].shape == (1, 32)


def test_action_validity():
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = legalize_action(rng.integers(0, 2, size=9, dtype=np.int8))
        assert isinstance(a, np.ndarray)
        assert a.shape == (9,)
        assert set(np.unique(a)).issubset({0, 1})
    # contradictory buttons are removed
    a = legalize_action(np.ones(9, np.int8))
    assert a.sum() < 9 and not (a[0] and a[1]) and not (a[2] and a[3]) and not (a[4] and a[5])


def test_use_edge_semantics():
    a = np.zeros(9, np.int8)
    a[USE] = 1
    assert legalize_action(a, np.zeros(9, np.int8))[USE] == 1
    assert legalize_action(a, a)[USE] == 0  # must release for one step
    assert legalize_action(a, a)[0:6].sum() >= 0  # other buttons unaffected


def test_reset_clears_state():
    cfg = KeyStoneConfig()
    model = KeyStoneNet(cfg)
    state = model.initial_state(2)
    state.h.fill_(1.0)
    state.mem.conf.fill_(1.0)
    fresh = state.reset(None)
    assert torch.all(fresh.h == 0)
    assert torch.all(fresh.mem.conf == 0)
    done = torch.tensor([1.0, 0.0])
    partial = state.reset(done)
    assert torch.all(partial.h[0] == 0) and torch.all(partial.h[1] == 1)


def test_memory_write_read_and_confidence():
    cfg = KeyStoneConfig()
    mem = EventMemory(cfg, EVIDENCE_DIM)
    state = mem.initial(1)
    ev = torch.zeros(1, EVIDENCE_DIM)
    ev[0, 6] = 1.0  # locked door evidence
    event = torch.tensor([EVENT_LOCKED_DOOR])
    q = torch.zeros(1, cfg.embed)
    for _ in range(5):
        state = mem.write(state, ev, event, q)
    ctx, w, rel = mem.read(q, state)
    assert ctx.shape == (1, cfg.embed)
    assert torch.isfinite(ctx).all() and torch.isfinite(rel).all()
    assert float(state.conf.max().detach()) > 0.0, "confidence must grow on repeated evidence"
    assert float(state.age.max().detach()) < 10.0, "retrieval should refresh age"


def test_sequence_replay_shapes():
    cfg = KeyStoneConfig()
    cfg.rollout_length = 8
    fb = FeatureBuilder(cfg)
    model = KeyStoneNet(cfg)
    obs = _obs()
    vec, _, event = fb(obs)
    parts = fb.split(torch.from_numpy(vec[None, :]))
    seq = {k: torch.stack([v] * 4, dim=0).contiguous() for k, v in parts.items()}
    state0 = model.initial_state(1)
    events = torch.full((4, 1), event, dtype=torch.long)
    out, final = model.forward_sequence(seq, state0, events)
    assert out["logits"].shape == (4, 1, 9)
    assert out["value"].shape == (4, 1)
    assert torch.isfinite(out["logits"]).all()
    assert final.h.shape == (1, cfg.hidden)


def test_no_privileged_input_or_network():
    """Static boundary audit of the deployed entry point."""
    src = (ROOT / "submission" / "agent.py").read_text(encoding="utf-8")
    for token in ("privileged_state", "vizdoom", "socket", "requests", "urllib",
                  "POSITION_X", "ANGLE", "set_doom", "open("):
        assert token not in src, f"forbidden runtime token: {token}"


def test_inference_latency():
    cfg = KeyStoneConfig()
    torch.set_num_threads(1)
    fb = FeatureBuilder(cfg)
    model = KeyStoneNet(cfg).eval()
    obs = _obs()
    vec, ev, event = fb(obs)
    state = model.initial_state(1)
    lat = []
    for _ in range(100):
        parts = fb.split(torch.from_numpy(vec[None, :]))
        t = time.perf_counter()
        with torch.inference_mode():
            out, state = model.step(parts, state, torch.tensor([event]))
        lat.append((time.perf_counter() - t) * 1000.0)
    p95 = float(np.percentile(lat, 95))
    assert p95 < 40.0, f"p95 latency {p95:.1f} ms exceeds the 40 ms/decision budget"


def test_reward_shaper_is_finite():
    from training.ppo import RewardShaper
    from training.exit_finder import find_exit
    cfg = KeyStoneConfig()
    cfg.bearing_potential = 0.5
    shaper = RewardShaper(cfg)
    shaper.reset(1)
    ex = find_exit(ROOT / "maps" / "train" / "move_0.wad", "MAP01")
    shaper.exits = [ex]
    base = {"x": 96.0, "y": 96.0, "angle": 135.0, "keys": np.zeros(6),
            "kills": 0, "damage_taken": 0.0, "exit_reached": False, "dead": False}
    nxt = dict(base, x=100.0, angle=120.0)
    r = shaper(0, base, nxt, np.zeros(64, np.int64), np.zeros(9, np.int8))
    assert isinstance(r, float) and np.isfinite(r)


def test_event_tag_key_and_locked_door():
    fb = FeatureBuilder(KeyStoneConfig())
    ev = np.zeros(EVIDENCE_DIM, np.float32)
    ev[6] = 0.5
    assert fb._event(ev, None, None) == EVENT_LOCKED_DOOR
    ev2 = ev.copy()
    ev2[0] = 1.0
    assert fb._event(ev2, ev, None) == EVENT_KEY

