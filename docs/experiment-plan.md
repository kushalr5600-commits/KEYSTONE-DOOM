# Experiment plan: curriculum, reward, benchmark, ablations

## Priority order (derived from the official docs, not assumed)

1. Exit completion — documented "largest" award (`docs/EVALUATION.md:44`).
2. Progression — keycards, doors, locked doors (`docs/EVALUATION.md:45`).
3. Useful exploration — unique areas visited (`docs/EVALUATION.md:46`).
4. Combat where it supports the above (`docs/EVALUATION.md:47`).
5. Survival / damage / time efficiency (`docs/EVALUATION.md:44,47`).

Scoring coefficients are hidden (`docs/EVALUATION.md:3-4`), so training weights below are
**our own choices**, strictly subordinate to exit (10.0) and progression (1.0/0.5), and are
never described as official.

## Curriculum (training-time scaffolding only)

| Stage | Step fraction | Maps | Purpose |
|---|---|---|---|
| A `basics` | 0–25% | 13 tiny fixture maps (`tests/fixtures/*.wad`) | legal actions, movement, exit discovery, basic survival |
| B `doors` | 25–50% | fixture maps containing doors (incl. 2 locked-door maps) | door evidence, USE semantics, key flags |
| C `mixed` | 50–75% | doors + Freedoom2 MAP01–02 | larger multi-room layouts |
| D `gauntlet` | 75–100% | doors + Freedoom2 MAP01–05 | long horizon, ambushes, scarce resources, multiple locks |

Each stage resets env/seed; hidden evaluation maps are never accessed. Training uses
`render_buffers=False` (zero RGB/depth) and the model has depth disabled by default, so
train and inference see the same structured fields.

## Reward (training only)

`training/ppo.py:RewardShaper` computes:

```
r = +10.0·exit  -3.0·death
  + 1.0·new_key_flags
  + 0.5·locked_door_disappeared_while_key_held
  + 0.05·kills
  - 0.005·damage_taken
  + 0.02·new_observation_signature
  - 0.002·per_step_time_cost
```

Privileged fields (`keys`, `kills`, `damage_taken`, `exit_reached`, `dead`) are used here as
**labels only** and come from `DoomEnv.privileged_state()`, which the deployed agent never
receives (`docs/CONTRACT.md:145-146`). The novelty term is 500× smaller than exit and 50×
smaller than a key, so it cannot outrank finishing. Explicit anti-hacking checks:
loops are discouraged by the per-step cost; killing is small and cannot replace exit;
key pickup without unlock earns less than unlock+exit; health protection comes from the
damage penalty and the exit efficiency term, not from an "avoid everything" bonus.

## PPO settings

`KeyStoneConfig`: γ 0.995, λ 0.95, clip 0.2, value coef 0.5, entropy 0.01, goal entropy
0.005, grad norm 0.5, lr 3e-4, 4 epochs, 4 minibatches, sequence length 16, rollout 64 per
env, 4 envs, seed 0. Sequence minibatches restore stored `h`/`MemoryState` so gradients flow
through recurrence and memory writes. Full hyperparameters are serialized into every
checkpoint (`submission/weights/config.json`).

## Benchmark protocol

- Held-out maps: fixture WADs not used in the final training stage plus Freedoom2 maps held
  out from training where possible. Training/evaluation sets are recorded in
  `experiments/results.csv`.
- Deterministic policy (argmax), same seeds for every ablation, `episode_ticks=2100`.
- Record: exit rate, score proxy (exit/key/unlock/exploration/kill accounting), key acquisition,
  locked-door unlock, unique observation signatures, kills, damage, health at end, deaths,
  steps to exit, stuck/loop rate, p50/p99/max inference latency, RAM/VRAM where measurable,
  validator result.
- Report sample size and uncertainty (bootstrap CI over episodes where n≥8). No single-run
  cherry-picking.

## Required ablation matrix (same protocol)

| # | Variant | Flag |
|---|---|---|
| 1 | Reactive/structured PPO baseline | `use_recurrence=False, use_memory=False, use_goal=False, use_confidence=False` |
| 2 | + recurrent core | `use_recurrence=True` |
| 3 | + event/topological memory | `use_memory=True` |
| 4 | + learned goal conditioning | `use_goal=True` |
| 5 | + confidence/risk verification | `use_confidence=True` (full agent) |
| 6 | + depth encoder (time permitting) | `use_depth=True` |

Run with `python -m training.train --config configs/ablation_X.json` and evaluate each with
`python -m training.evaluate --weights <ckpt>`. **Do not assert the USP helps until matched
evaluations support it.** If neutral/negative, simplify and report honestly.

## Execution budget (planning targets, not guarantees)

Repository audit + validator baseline first (done); then encoder/PPO + memory schema/tests;
then curriculum training; then progression tests; then benchmark, simplification, validator,
packaging. A valid compliant submission always takes priority over an untrained large model.

## Status of this plan

- [x] Stage 0 audit and runnable baseline (validator PASSED).
- [x] Architecture implemented, smoke training loop PASSED.
- [ ] Full training run and checkpoint.
- [ ] Ablations and held-out evaluation.
- [ ] Final validator + packaging.
