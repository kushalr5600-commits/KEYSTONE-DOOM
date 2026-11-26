"""KEYSTONE-DOOM training entry point.

Usage (from repo root, with env/ on PYTHONPATH):
    python -m training.train --steps 50000 --out submission/weights

Curriculum stages are training-time scaffolding only. Evaluation always uses the
unmodified engine and official wrapper (docs/RULES.md Â§2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from doom_env import make_env
from keystone import KeyStoneConfig, FeatureBuilder, KeyStoneNet, legalize_action
from training.ppo import PPOTrainer, RewardShaper
from training.exit_finder import find_exit

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("*.wad"))
FIXTURES_WITH_EXIT = [p for p in FIXTURES + [ROOT / "maps" / "two_rooms.wad"]
                      if find_exit(p, "MAP01") is not None]


def _maps_with_doors():
    from doom_env.mapinfo import load_mapinfo
    out = []
    for p in FIXTURES:
        try:
            if load_mapinfo(str(p), "MAP01").door_sectors:
                out.append(p)
        except Exception:
            continue
    return out or FIXTURES[:4]


DOOR_FIXTURES = _maps_with_doors()


def map_pools():
    iwad = Path(torch.__file__).parent.parent  # replaced below with vizdoom location
    import vizdoom as vd
    iwad = Path(vd.__file__).parent / "freedoom2.wad"
    small = [(str(p), "MAP01") for p in FIXTURES_WITH_EXIT]
    doors = [(str(p), "MAP01") for p in DOOR_FIXTURES]
    full_maps = [(str(iwad), f"MAP{i:02d}") for i in (1, 2, 3, 4, 5)]
    return {
        "basics": small,
        "doors": doors + small[:3],
        "mixed": doors + full_maps[:2] + small[:2],
        "gauntlet": doors + full_maps + small[:2],
    }


def stage_for(frac: float) -> str:
    # Long basics phase: learning exit completion on tiny maps is the score-dominant
    # behaviour; only then move to doors and the larger gauntlet maps.
    if frac < 0.50:
        return "basics"
    if frac < 0.70:
        return "doors"
    if frac < 0.85:
        return "mixed"
    return "gauntlet"


def build_envs(pool, n, ticks, seed):
    envs, exits = [], []
    for i in range(n):
        wad, mp = pool[i % len(pool)]
        envs.append(make_env(wad, mp, episode_ticks=ticks, render_buffers=False))
        exits.append(find_exit(wad, mp))  # TRAINING ONLY; never given to the agent
    return envs, exits


def save_ckpt(model, cfg, path: Path, step: int, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": json.loads(cfg.to_json()),
        "step": int(step),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", default="submission/weights")
    ap.add_argument("--config", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--pool", default=None,
                    choices=["basics", "doors", "mixed", "gauntlet"],
                    help="force a single map pool for the whole run (curriculum override)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    cfg = KeyStoneConfig.load(args.config) if args.config else KeyStoneConfig()
    cfg.total_steps = args.steps
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "experiments" / f"train_{args.tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fb = FeatureBuilder(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeyStoneNet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck.get("optimizer", optimizer.state_dict()))
        start_step = int(ck.get("step", 0))
        print(f"resumed from step {start_step}", flush=True)

    trainer = PPOTrainer(model, optimizer, cfg)
    pools = map_pools()
    stage = args.pool or stage_for(start_step / max(1, cfg.total_steps))
    envs, exits = build_envs(pools[stage], cfg.num_envs, args.ticks, cfg.seed)
    obs_list = [e.reset(seed=cfg.seed + i)[0] for i, e in enumerate(envs)]
    states = model.initial_state(cfg.num_envs, device=device)
    prev_evs = [np.zeros(16, np.float32) for _ in envs]
    prev_actions = [np.zeros(9, np.int8) for _ in envs]
    ep_ids = np.zeros(cfg.num_envs, np.int64)
    shaper = RewardShaper(cfg)
    shaper.reset(cfg.num_envs)
    shaper.exits = list(exits)

    ep_stats = {"episodes": 0, "exits": 0, "deaths": 0, "keys": 0, "kills": 0, "steps": 0}
    log_f = log_path.open("a", encoding="utf-8")
    t0 = time.perf_counter()
    step = start_step

    while step < cfg.total_steps:
        buf, states, prev_evs, prev_actions, ep_ids = trainer.collect(
            envs, obs_list, states, fb, prev_evs, prev_actions, ep_ids, shaper, cfg)
        step += buf.T * buf.N
        stats = trainer.update(buf, fb)

        for t in range(buf.T):
            for i in range(buf.N):
                if buf.dones[t, i]:
                    ep_stats["episodes"] += 1
                    ep_stats["steps"] += int(buf.ep_ids[t, i] >= 0)
        # Exact per-episode outcomes are logged by the collector via a light post-check.
        rec = {
            "step": step, "wall_s": round(time.perf_counter() - t0, 1),
            "stage": stage, "policy": round(stats["policy"], 4),
            "value": round(stats["value"], 4), "entropy": round(stats["entropy"], 4),
            "goal_entropy": round(stats["goal_ent"], 4),
            "mean_reward": round(float(buf.rewards.mean()), 4),
            "exit_reached": int(np.any(buf.rewards >= cfg.reward_exit - 1e-6)),
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % (cfg.log_interval * buf.T * buf.N) < buf.T * buf.N:
            print(json.dumps(rec), flush=True)

        new_stage = args.pool or stage_for(step / max(1, cfg.total_steps))
        if new_stage != stage:
            stage = new_stage
            for e in envs:
                e.close()
            envs, exits = build_envs(pools[stage], cfg.num_envs, args.ticks, cfg.seed + step)
            obs_list = [e.reset(seed=cfg.seed + step + i)[0] for i, e in enumerate(envs)]
            states = model.initial_state(cfg.num_envs, device=device)
            prev_evs = [np.zeros(16, np.float32) for _ in envs]
            prev_actions = [np.zeros(9, np.int8) for _ in envs]
            ep_ids = np.zeros(cfg.num_envs, np.int64)
            shaper.reset(cfg.num_envs)
            shaper.exits = list(exits)
            print(f"[stage] {stage} at step {step}", flush=True)

        if step % max(1, cfg.save_interval * buf.T * buf.N) < buf.T * buf.N:
            save_ckpt(model, cfg, out / "keystone.pt", step,
                      {"optimizer": optimizer.state_dict()})
            print(f"[checkpoint] step {step}", flush=True)

    save_ckpt(model, cfg, out / "keystone.pt", step,
              {"optimizer": optimizer.state_dict()})
    cfg.save(out / "config.json")
    log_f.close()
    for e in envs:
        e.close()
    print(f"DONE step={step} wall={time.perf_counter() - t0:.1f}s weights={out / 'keystone.pt'}", flush=True)


if __name__ == "__main__":
    main()

