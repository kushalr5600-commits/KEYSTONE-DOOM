"""Demo-only memory ledger / debug overlay.

Runs the exact submission agent on a local map and prints agent-visible state: objective
distribution, memory-slot confidence and age, vitals, action. It never reads privileged
state and is NOT part of submission/agent.py or the inference path.

    python tools/memory_ledger.py --map maps/two_rooms.wad --every 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT)]

from doom_env import make_env  # noqa: E402
from agent import DoomAgent  # noqa: E402

GOALS = ("engage", "explore", "investigate", "pursue_key",
         "revisit_gate", "test_use", "exit")
EVENTS = ("none", "locked_door", "door", "key", "item", "enemy",
          "interact_ok", "interact_fail")
BUTTONS = ("F", "B", "SL", "SR", "TL", "TR", "ATK", "USE", "WPN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="maps/two_rooms.wad")
    ap.add_argument("--map-name", default="MAP01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    from keystone import KeyStoneConfig
    cfg = KeyStoneConfig.load(ROOT / "submission" / "weights" / "config.json")
    env = make_env(str(ROOT / args.map), args.map_name, episode_ticks=args.ticks,
                   render_buffers=False)
    obs, _ = env.reset(seed=args.seed)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.reset()
    print("REPLAY / deterministic debug ledger (agent-visible fields only)\n")

    step = 0
    done = False
    while not done:
        action = agent.act(obs)
        if step % args.every == 0:
            probs = agent.last_goal_probs
            top = int(np.argmax(probs))
            conf, age = agent.last_memory["conf"], agent.last_memory["age"]
            active = [(i, float(conf[i]), float(age[i]))
                      for i in np.argsort(-conf)[:4] if conf[i] > 0.01]
            pressed = [BUTTONS[i] for i in range(9) if action[i]]
            events = agent.last_memory["events"]
            slots = ", ".join(f"s{i}:{EVENTS[int(events[i])]} c={c:.2f} a={a:.0f}"
                              for i, c, a in active)
            vit = obs["vitals"]
            print(f"step {step:4d} | hp={vit[0]:5.1f} ar={vit[1]:5.1f} "
                  f"keys={int(np.sum(vit[8:14]))} | goal={GOALS[top]:12s} "
                  f"p={probs[top]:.2f} | conf={agent.last_confidence:.2f} | "
                  f"action={'+'.join(pressed) or '-'} | {slots}")
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        done = term or trunc
    print(f"\nend: step={step} exit={info.get('exit_reached', False)}")
    env.close()


if __name__ == "__main__":
    main()
