# Official specification summary (source-grounded)

**Status:** extracted from the official template at template HEAD `63b90c4` / local clone commit
`0afe3be`. Every claim below has a source path and line number. Anything not disclosed is
marked **not disclosed** — we do not infer coefficients.

Companion source: `Yugaantar DevForge Tracks (1).pdf`, Track 1 pages 1–6 (universal submission
rules, hardware, timeouts, scoring categories, ground rules).

## 1. Contract and interface

| Item | Exact value | Source |
|---|---|---|
| Contract version | `1.0.0-draft` | `docs/CONTRACT.md:3`; `env/doom_env/__init__.py:9` |
| Observation | Dict with `rgb`, `depth`, `vitals`, `rays`, `visible_entities` | `docs/CONTRACT.md:22-30`; `env/doom_env/spaces.py:73-95` |
| `rgb` | `Box(0,255,(120,160,3),uint8)`, HUD hidden | `env/doom_env/spaces.py:76`; `docs/CONTRACT.md:26` |
| `depth` | `Box(0,255,(120,160),uint8)` | `env/doom_env/spaces.py:77` |
| `vitals` | `Box(-inf,inf,(17,),float32)`; 0 health, 1 armor, 2 weapon, 3 selected ammo, 4–7 ammo pools, 8–13 six key flags, 14–16 ego-frame velocity | `env/doom_env/spaces.py:36-55,78`; `docs/CONTRACT.md:34-44`; `env/doom_env/wrapper.py:184-187` |
| `rays.distance` | `Box(0,1,(64,),float32)`; hit distance / 1024, `1.0` = nothing | `env/doom_env/spaces.py:14-16,81`; `docs/CONTRACT.md:46-60`; `env/doom_env/sensors.py:267-271` |
| `rays.hit_type` | `MultiDiscrete([6]*64)`; 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item | `env/doom_env/spaces.py:16,82`; `docs/CONTRACT.md:60` |
| Ray geometry | index 0 ahead, counter-clockwise, 64 rays / 360°, index 16 left, 32 behind | `docs/CONTRACT.md:48-49`; `env/doom_env/sensors.py:190,210` |
| `visible_entities.features` | `Box(-inf,inf,(32,5),float32)` = `[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]` | `env/doom_env/spaces.py:87-89`; `docs/CONTRACT.md:70` |
| `visible_entities.entity_type` | `MultiDiscrete([12]*32)`; 0 = pad, 1–3 enemies, 4 projectile, 5 health, 6 armor, 7 ammo, 8 weapon, 9 key, 10 barrel, 11 other | `env/doom_env/spaces.py:18-34,90` |
| `visible_entities.mask` | `MultiBinary(32)`, nearest-first, zero-padded, 4096 range, ±45° FOV, LOS-filtered | `env/doom_env/spaces.py:91`; `env/doom_env/sensors.py:22-23,239-298`; `docs/CONTRACT.md:64-72` |
| Action | `MultiBinary(9)`; 0 fwd, 1 back, 2 strafe-L, 3 strafe-R, 4 turn-L, 5 turn-R, 6 attack, 7 USE, 8 next weapon | `env/doom_env/spaces.py:57-68,98-99`; `docs/CONTRACT.md:76-88` |
| Action duration | 4 engine tics per `act()` (`FRAME_SKIP=4`), buttons held for all 4 tics | `env/doom_env/spaces.py:70`; `env/doom_env/wrapper.py:128-130`; `docs/CONTRACT.md:93,109` |
| Edge triggers | `USE` fires on button-down; must release ≥1 step before re-use; re-press on moving door reverses it. `SELECT_NEXT_WEAPON` also edge-triggered | `docs/CONTRACT.md:93-100` (documented; no separate local edge-detection code — vanilla ViZDoom 1.3.3 enforces it) |
| Invalid action | Wrong shape/values terminate the episode with score-so-far kept | `docs/CONTRACT.md:90-91`; `env/doom_env/remote.py:209-217`; `docs/EVALUATION.md:65-67` |
| Entry point | `submission/agent.py`, class `DoomAgent`, `__init__(observation_space, action_space, device)`, `reset()`, `act(observation) -> np.ndarray` | `docs/CONTRACT.md:113-130`; `submission/agent.py:16-29`; `validate.py:203-251` |
| Episodic reset | `reset()` before every map must clear recurrent state, memory buffers, trackers | `docs/CONTRACT.md:126`; `validate.py:401-402` |
| Engine | ViZDoom 1.3.3, FreeDoom2 IWAD, skill 3, unmodified vanilla rules at evaluation | `docs/CONTRACT.md:102-105`; `env/doom_env/wrapper.py:24,66-90`; PDF p.5 |
| Seed | wrapper seeds `int(seed) % 2**31`; validator uses `seed=0`; hidden seeds fixed but **values not disclosed** | `env/doom_env/wrapper.py:108-109`; `validate.py:401`; `docs/EVALUATION.md:25-27` |

## 2. Evaluation tiers

Exact documented descriptions (`docs/EVALUATION.md:29-34`):

| Tier | Category | Skills tested |
|---|---|---|
| 1 | Combat & Survival | Aiming, dodging projectiles, managing health/ammo |
| 2 | Multi-Room Exploration | Partial observability, spatial memory, avoiding dead-end traps, doors |
| 3 | Long-Horizon Progression | Finding keycards, remembering locked doors, backtracking to the exit |
| 4 | Full Gauntlet | All of the above with ambushes, scarce resources and multiple locks |

All maps are unseen; every agent plays every map in increasing difficulty order with fixed
seeds (`docs/EVALUATION.md:25-27`). The bundled `maps/two_rooms.wad` is **not** an evaluation map
(`docs/EVALUATION.md:36`; `maps/README.md:3-7`).

## 3. Scoring rubric — exact terms, undisclosed weights

`docs/EVALUATION.md:3-4` states: "Exact weights and map details are kept hidden" and values
marked *(draft)* are announced before the event. **No scoring implementation exists in the
template** (exhaustive grep: only docs, the validator's status strings, and the wrapper's
convenience reward). Therefore:

| Term | Documented behaviour | Coefficients |
|---|---|---|
| Exit reached | "the largest" award, plus a small efficiency bonus for remaining health and time | **not disclosed** |
| Progression | keycards picked up, doors opened, locked doors unlocked | **not disclosed** |
| Exploration | unique areas visited; "spinning in place earns nothing" (tracked privately by scorer) | **not disclosed** (area granularity also undisclosed) |
| Combat & survival | enemies killed; small penalties for self-damage and dying early | **not disclosed** |
| Aggregation | harder tiers carry more weight; leaderboard = tier-weighted sum over maps | tier weights **not disclosed** |
| Map tick cap | about 2,100–4,200 ticks (≈525–1050 decisions) | `docs/CONTRACT.md:110`; PDF p.5 |
| Timeout / crash | agent stands still for rest of map; points so far kept; penalties still apply; restart next map | `docs/CONTRACT.md:90-91`; `docs/EVALUATION.md:65-67` |

**Important boundary:** the only numeric reward in code is `env/doom_env/wrapper.py:140-147`
(`+1.0` exit, `-1.0` death, `+0.1` per kill), explicitly labelled "not the evaluation score"
at `docs/CONTRACT.md:141-143` and `README.md:61`. We never treat it as the official score and
never fit coefficients to it.

**Optimisation order** (from documented "largest" exit term): exit completion → progression →
useful exploration → combat where it contributes → survival/time efficiency. This is the
order used by our reward shaper (`training/ppo.py:RewardShaper`) and evaluation report.

## 4. Runtime, resources, sandbox

| Limit | Exact value | Source |
|---|---|---|
| Init (`__init__`) | 60 s | `env/doom_env/limits.py:4`; `validate.py:388-393`; PDF p.6 |
| `reset()` | 10 s per map | `env/doom_env/limits.py:5`; `validate.py:402` |
| First `act()` | 10 s per map | `env/doom_env/limits.py:6`; `validate.py:408,423-429` |
| Other `act()` | 5 s | `env/doom_env/limits.py:7`; `validate.py:408,425-429` |
| Agent compute budget | `0.04 * ceil(ticks/4)` s per map (21 s @2100 ticks, 42 s @4200) | `env/doom_env/limits.py:8-18`; `validate.py:404-411` |
| Speed warning | p50 > 24 ms (0.6 × 40 ms) | `validate.py:430-431` |
| Reporting | p50/p99/max and first-step latency (no mean/p95 in validator) | `validate.py:355-357,420-429` |
| VRAM | 2048 MB | `env/doom_env/limits.py:19`; `validate.py:456-462`; `env/doom_env/remote.py:324-337` |
| RAM | 8192 MB (warn >6553.6 MB) | `env/doom_env/limits.py:20`; `validate.py:451-455` |
| CPU | 1 thread; OMP/MKL/OPENBLAS/NUMEXPR = 1 | `env/doom_env/limits.py:21,23-31`; `validate.py:347-352,369` |
| Network | none at evaluation | `docs/CONTRACT.md:129`; `docs/EVALUATION.md:18`; PDF p.6 |
| Repository | mounted read-only; `/tmp` writable (2 GB) | `docs/CONTRACT.md:128`; `docs/EVALUATION.md:18` |
| Hardware | RTX 3090, 2 GB VRAM / agent, 8 GB RAM, 1 CPU thread | `docs/EVALUATION.md:6-18`; PDF p.6 |

## 5. Submission package and validator

| Rule | Exact requirement | Source |
|---|---|---|
| Repository | public GitHub repo created from official template during hackathon | `docs/SUBMISSION.md:3-7`; `docs/RULES.md:27-32`; PDF p.1, p.4 |
| Size | ≤ 1 GiB tracked files including LFS objects | `env/doom_env/limits.py:3`; `validate.py:66-89`; `docs/SUBMISSION.md:10` |
| Large files | >50 MiB non-LFS → WARN; >100 MiB → FAIL; `.pt/.pth/...` must be LFS | `validate.py:36-37,84-96`; `.gitattributes:1-12`; `docs/SUBMISSION.md:9,16-30` |
| `submission/` state | no ignored/untracked/uncommitted files; scorer uses last pushed commit | `validate.py:110-124`; `docs/SUBMISSION.md:57-59` |
| Extra packages | `submission/requirements.txt`, ≤50 lines, `package==version` only | `env/doom_env/limits.py:33-41`; `validate.py:127-149`; `docs/SUBMISSION.md:38-42` |
| Dockerfile | do **not** submit one; agents run in organizers' base image | PDF p.4 (`Dependencies`) |
| Base image | Python 3.13, vizdoom, gymnasium, torch, jax/flax, numpy, scipy, opencv | PDF p.4; `docs/SUBMISSION.md:34-36`; `docker/requirements.lock` |
| Validator | `python validate.py`; recommended `python validate.py --docker`; custom maps via `--map file.wad:MAP01` | `docs/SUBMISSION.md:44-53`; `README.md:16-26`; `validate.py:551-557` |
| Validator checks | size, committed state, requirements, import whitelist, interface, init/reset/act latency, full dry-run episode, RAM/VRAM | `validate.py:66-600` |
| Training code | keep in repo; judges read it | `docs/RULES.md:30-32`; `docs/SUBMISSION.md:13` |

## 6. Prohibited / required conduct

`docs/RULES.md:6-17,34-40`, `docs/SUBMISSION.md:14`, PDF p.6:

- **Prohibited:** hand-written if/else action trees, A*/Dijkstra/BFS or other classical
  pathfinding/planning, `atan2`-style trigonometric aimbots, unmodified off-the-shelf SB3
  policies, engine/map/host/network access at evaluation, resource hogging.
- **Allowed:** preprocessing/encoding, action masking from observation fields, frame stacking,
  learned memory, all training-time scaffolding (engine/ACS/game-variable changes, curricula).
- **Required:** decisions from trained neural parameters; core architecture and training
  pipeline built during the event; training code in the repo.

## 7. Rule → implementation / test map

| Rule | Enforcement in our code | Test |
|---|---|---|
| Observation shape/keys/finiteness | `keystone/features.py` asserts 1249-dim vector; nan-to-num on entity features | `tests/test_keystone.py::test_feature_shapes_and_finite` |
| Action shape/values in {0,1} | `keystone/actions.py`; agent returns `np.int8` (9,) | `tests/test_keystone.py::test_action_validity` |
| USE edge semantics | `legalize_action` clears a repeated USE / weapon-select; identical in train and inference | `tests/test_keystone.py::test_use_edge_semantics` |
| Reset clears memory | `DoomAgent.reset` builds fresh `AgentState` | `tests/test_keystone.py::test_reset_clears_state` |
| Memory is observation-only | feature builder never receives privileged state | code audit + `tests/test_keystone.py::test_no_privileged_input` |
| Inference latency | small 314k-param model, 1 thread, no_grad | `python validate.py` reports p50/p99/max; `tests/test_keystone.py::test_inference_latency` |
| Offline / read-only | no sockets, no file writes in `submission/agent.py` | static audit + validator dry run |
| LFS / size | `.gitattributes`, weights under `submission/weights/` | `python validate.py` size check |
| Tier 4 focus | curriculum gauntlet stage, held-out fixture evaluation | `training/evaluate.py`, `experiments/results.csv` |

## 8. Ambiguities recorded (do not treat as resolved)

1. Scoring weights/formulas/tier weights explicitly hidden (`docs/EVALUATION.md:3-4`).
2. Timeouts marked "(draft)" in docs but pinned in `limits.py`; we trust the code, as
   `docs/EVALUATION.md:61` says the validator enforces identical numbers.
3. Docker memory discrepancy: validator container uses `MAX_RAM_MB+2048` (10 GB) while agent
   limit is 8 GB (`validate.py:509,522,527` vs `limits.py:20`). We target 8 GB.
4. `check-attr` LFS detection only inspects the first 5000 files (`validate.py:76`); our repo
   has far fewer, so this does not affect us.
5. `USE` edge behaviour is documented but not re-implemented locally; we rely on vanilla
   ViZDoom 1.3.3 and enforce a conservative one-step release in `legalize_action`.
