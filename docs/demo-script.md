# Demo script and judge-facing narrative

## 60-second USP pitch (truthful)

"Most DOOM agents either memorize button sequences or use a recurrent network that forgets a
locked door it saw three rooms ago. KEYSTONE-DOOM keeps a tiny learned memory of
progression events — a locked door, a key pickup, a successful or failed interaction — with a
confidence and age for each entry. A goal-conditioned policy reads that memory before acting,
so it can revisit a remembered door after finding a key instead of wandering. Everything the
agent sees is the official ego-centric observation: 64 rays, vitals and visible entities. No
engine handle, no map coordinates, no scripted route. The whole agent is ~314k parameters and
runs at about 2.6 ms per decision locally, well inside the 40 ms budget. We measure the memory
and goal components with matched-seed ablations and report results honestly, including cases
where they do not help."

## Demo sequence (recorded replay preferred; label it as replay)

1. **Spawn in an unfamiliar room.** Overlay shows current vitals, objective distribution and
   an empty memory ledger.
2. **Observe a locked door.** Ray hit type 3 appears; a `locked_door` entry is written with
   confidence and age; the objective distribution shifts toward `investigate / pursue_key`.
3. **Explore / fight.** Enemy evidence is stored as a separate event; the policy engages or
   evades using learned logits. Health/ammo context updates.
4. **Acquire a key.** `vitals[8:14]` rises; a `key` entry is written (event tag derived from
   observation only). The remembered gate is refreshed rather than overwritten.
5. **Revisit and interact.** Back at the door, the policy selects `USE`; `legalize_action`
   enforces a one-step release so the door is not reversed. Interaction outcome tags the
   memory entry as success/failure.
6. **Exit.** If achieved, overlay shows tick count, health and score-relevant events. If the
   run did not reach the exit, say so and show the partial-progress metrics instead.

Use `training/evaluate.py` with a fixed seed to record a reproducible replay. A stochastic
live run is acceptable only if labelled as such; never re-run until a favourable outcome and
present it as typical.

## Debug/demo overlay (allowed)

Show only agent-visible information: current objective probabilities, last N memory entries
(event type, confidence, age, number of read slots), vitals, action, and score-relevant
counters derived from observation. **Do not** display privileged state, hidden map data, or
the engine. The overlay is a debug tool outside `submission/agent.py`, not an inference input.

## Architecture diagram

See `docs/architecture.md` for the dataflow ASCII diagram. A slide version should show:
obs → typed encoders → GRU → event memory (read/write) → confidence gate → goal head →
actor/critic → 9 buttons. Highlight that the memory is learned and differentiable.

## Benchmark and ablation table

Populate from `experiments/results.csv` after running:

```
python -m training.evaluate --weights submission/weights/keystone.pt --tag final --episodes 8
python -m training.train --config configs/ablation_1.json --steps 30000 --tag a1
python -m training.evaluate --weights submission/weights/a1.pt --tag a1 --episodes 8
...
```

Report: exit rate, key/unlock rate, unique areas proxy, kills, damage, deaths, steps to exit,
stuck rate, p50/p99 latency, sample size and bootstrap CI. Do not fabricate numbers.

## Likely judge questions

- **"Isn't this just an LSTM with a dictionary?"** The memory has learned content-addressed
  writes, event-type slots, confidence/age dynamics and a confidence-gated read that
  conditions a learned goal head; the ablation matrix isolates each part. We do not claim
  the individual mechanisms are novel.
- **"Where does the memory get positions?"** It doesn't. The contract exposes no global pose;
  entries are keyed by observation-derived evidence, not coordinates.
- **"Do you use privileged state?"** Only as training labels in `training/ppo.py`. The
  deployed `submission/agent.py` receives the observation dict only; a unit test statically
  checks forbidden tokens.
- **"Any pathfinding / aimbot?"** No A*/BFS/Dijkstra, no `atan2`. Turning and movement come
  from Bernoulli logits. `legalize_action` only enforces contract button semantics.
- **"Why no pixels?"** CPU profiling showed structured signals are >2× faster; the contract
  recommends them. A depth branch exists as an ablation.
- **"Did it actually beat the baseline on Tier 4?"** Answer with the measured table, or say
  "not verified" — never assert an unmeasured improvement.

## Limitations to state unprompted

Hidden scoring weights; CPU-only local training; no local GPU/VRAM measurement; small fixture
maps plus Freedoom2 scaffolding rather than hidden evaluation maps; confidence not calibrated;
memory is not a metric map. See `docs/novelty-and-limitations.md`.
