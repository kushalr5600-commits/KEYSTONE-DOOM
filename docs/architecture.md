# KEYSTONE-DOOM architecture

**Project:** Key-Event Structured Topological Memory for Full-Gauntlet DOOM.
Goal: a compact, learned, contract-compatible agent that remembers progression evidence
across partial observations and conditions its policy on retrieved evidence.

## Dataflow

```
official obs dict (rgb, depth, vitals, rays, visible_entities)
        │
        ▼
FeatureBuilder (keystone/features.py)  -- pure numpy, no privileged input
  rays(64×7) ─┐
  vitals(17)  ├─► typed encoders ─► concat ─► MLP ─► z (128)
  entities    ├─ masked pooling (mean+max) ─┘
  depth (optional, off by default) ─┘
        │
        ▼
GRUCell(z, h) ─► h (128) ─► query (32)
        │
        ├──► EventMemory.read(query, slots) ─► context c (32), reliability
        │         │  slots: embedding(32), event-type(8), confidence(1), age(1) × 16
        │         └──► EventMemory.write(observation-derived evidence, type, query)
        │
        ├──► risk/confidence head: sigmoid([h,c]) ─► gate ─► c_eff = c·gate
        ├──► goal head: softmax([h,c_eff]) ─► 7 learned goal embeddings ─► goal_vec
        │
        ├──► actor([z,h,c_eff,goal_vec]) ─► 9 Bernoulli logits ─► action
        └──► critic([z,h,c_eff]) ─► value
```

`submission/agent.py` loads `submission/weights/keystone.pt`, recreates this exact graph,
keeps `AgentState(h, MemoryState)` between decisions and clears it in `reset()`.
Inference uses `torch.inference_mode()`, one CPU thread and deterministic argmax of the
per-button Bernoullis. `keystone.actions.legalize_action` then enforces contract button
validity only (see below).

## Module responsibilities

| File | Responsibility |
|---|---|
| `keystone/config.py` | All architecture/PPO/shaping hyperparameters; serialized with every checkpoint. Ablation switches: `use_memory`, `use_recurrence`, `use_goal`, `use_confidence`, `use_depth`. |
| `keystone/features.py` | Fixed 1249-dim observation vector, 16-dim progression-evidence vector, and 8-way event tag (key, locked door, door, item, enemy, interaction ok/fail). |
| `keystone/memory.py` | Fixed-capacity differentiable event memory: content-addressed soft write, confidence growth/decay, age decay, attention read. |
| `keystone/model.py` | Encoders, GRU, memory read/write, goal head, confidence gate, actor and critic; `step` and `forward_sequence` APIs shared by training and inference. |
| `keystone/actions.py` | Contract button-validity post-processing (shared train/inference). |
| `training/ppo.py` | PPO with GAE, sequence minibatches that replay stored hidden/memory states through the graph; observation-only novelty shaping plus privileged *labels* for keys/kills/damage. |
| `training/train.py` | Curriculum (`basics → doors → mixed → gauntlet`), checkpoints, JSONL logs. |
| `training/evaluate.py` | Deterministic held-out evaluation and CSV metrics. |

## Memory schema (fixed 16 slots)

Each slot stores: event embedding (32 floats), event-type distribution (8), confidence [0,1],
age (steps). Writes are content-addressed: a softmax over embedding similarity + event-type
agreement + log-confidence − age cost decides how strongly each slot is updated (a soft,
differentiable equivalent of "update matching slot / allocate a new one"). Confidence rises
when evidence repeatedly agrees and decays otherwise; retrieval refreshes age. Reading is
query-conditioned attention; a reliability signal scales the retrieved context before it
reaches the goal and actor heads. Stale or contradicted evidence therefore loses influence
gradually rather than being hard-coded as true.

Events are derived *only* from the observation: rising `vitals` key flags, `rays.hit_type==3`
(locked door), `hit_type==2` (door), `hit_type==5`/entity types (item/key), `hit_type==4`/
enemy types, and `USE`-release outcomes measured as a change in locked-door evidence. No
absolute pose, sector id, map file or engine variable is stored.

## Goal head

Seven latent objectives: `engage, explore, investigate, pursue_key, revisit_gate, test_use,
exit`. These are *learned* embeddings mixed by a softmax over goal logits; the actor sees the
resulting vector. They are not a hand-authored decision tree: no code maps a goal id to an
action. The head is trained end-to-end by PPO plus a small goal-entropy regulariser.

## Action interface and edge-trigger handling

Actions are 9 binary buttons, held for 4 tics (`docs/CONTRACT.md:76-93`). The policy samples
Bernoulli buttons from logits. `legalize_action` removes contradictory simultaneous movement
buttons and forces `USE`/`SELECT_NEXT_WEAPON` to be released for one step after being held,
matching the documented vanilla edge trigger (`docs/CONTRACT.md:93-100`). This is a validity
constraint, not a decision rule: it never selects *which* button to press, and it is applied
identically in training and evaluation.

## Boundary compliance

| Concern | Status |
|---|---|
| Hidden/privileged state at inference | None. `submission/agent.py` touches only `observation`, `keystone/*`, weights and torch. |
| Absolute pose / map coordinates | Never observed, never stored. |
| Hardcoded routes, A*/BFS/Dijkstra | None. No map representation, no path search. |
| `atan2` / trig aimbot | None. Turning is learned from logits. |
| Scripted decision tree | None. Goals are learned latent vectors; no if/else action selection. |
| Evaluator modification | None. `env/`, `validate.py`, `docs/` official files unchanged. |
| Network at runtime | None. No socket/requests imports. |
| Repository writes at runtime | None. Weights read-only; scratch only in memory. |
| Reward shaping | Training only (`training/ppo.py`). Privileged values (including training-WAD exit coordinates via `training/exit_finder.py`) are labels/targets never visible to the deployed policy. Potential-based exit shaping cannot reward loops; intrinsic term ≤0.02/step vs exit 10.0. |
| Training-only engine changes | None used so far; curriculum varies map/seed only, all on the vanilla engine. |

## Resource profile (measured locally)

314,156 parameters; checkpoint ~1.3 MB (well under the 1 GB / LFS thresholds). Validator on
the test map at 420 ticks: init 1.7 s (limit 60), first act 9 ms (limit 10 s), p50 2.6 ms,
p99 4.0 ms, max 9 ms (limit 5 s; budget 40 ms/decision). GPU/VRAM not measurable on this CPU
machine — recorded as **not verified** locally; the model is tiny and uses `map_location` plus
the official VRAM cap, and `--docker` validation is recommended on a GPU host.

## Simplifications and risks

- RGB is unused by default: pixel-only training is far slower on CPU and the contract
  recommends structured signals. Depth downsampling code exists for an ablation.
- The event memory is slot-attention rather than a true graph/topological map. "Topological"
  in the codename refers to room-transition *evidence*; we do not claim metric mapping.
- Confidence is learned/continuous, not a calibrated probability. Calibration is future work.
- Long Freedoom levels have sparse exit rewards; training on them is a curriculum experiment,
  not a guarantee of Tier 4 completion.
