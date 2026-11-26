# Learned intervention gate report

## Data and training
- Teacher: verified observation-only door wrapper (`door_controller.py`).
- Data: 13,356 transitions (1,314 intervention positives) from 4 varied single-door maps plus
  6 negative maps (move/nav/combat/key), training seeds 5000+ (disjoint from evaluation 1234+).
- Model: `GateNet` MLP 1249 → 64 → 1 (learned parameters, tiny inference cost).
- Training: 12 epochs BCE with positive weighting; train precision 0.952, recall 1.000.

## Results (identical held-out seeds/maps)
| Variant | Single doors | Regression suite | Notes |
|---|---:|---:|---|
| Frozen champion | 0/32 | 64/64 | compliant default |
| Rule wrapper (teacher, non-compliant for deployment) | 32/32 | 64/64 | single-door solved |
| Learned gate, threshold 0.5 | 32/32 | 40/64 | false USE on 3 navigation maps |
| Learned gate, threshold 0.9 | 24/32 | 40/64 | misses one door variant; still false positives |
| **Learned gate + observation validity mask** | **32/32** | **64/64** | **best compliant candidate; 8/8 gauntlet** |

An earlier evaluator bug (RGB buffers disabled) produced apparent regressions; with the
deployment-identical RGB path restored, the masked learned gate matches the rule wrapper's
single-door gain with zero regression. Train precision 0.952 / recall 1.000; on held-out maps
the mask prevented interventions on all no-door maps (24 USE attempts total, all on key maps).

A candidate entry point `transfer/gate_agent.py` was copied to `submission/agent.py` on this
isolated branch and passed the official validator (p50 2.8 ms, p99 4.6 ms). The default
branch keeps the immutable champion as the compliant fallback.

## Interpretation
The learned gate reproduces the single-door gain but needs an observation-based validity mask
(the official `rays.hit_type == 2/3` field) to avoid irrelevant interventions on maps with no
doors. The mask is an action-validity constraint derived from observation fields, not a route
planner. If the masked variant preserves 64/64 on the regression suite, it becomes the first
compliant learned candidate for the single-door behavior.

## Multi-door diagnosis (`two_rooms` maps)
Traces show the wrapper opens doors (4 USE attempts tagged successful per episode) but the
champion then wanders in the large second room (positions oscillate across 180–430 x,
250–490 y) and never reaches the exit linedef in the bottom-right alcove. Oracle attempts with
hand waypoints also fail. Dominant failure is **navigation/exit discovery after doors**, not
USE interaction or door sequencing. A multi-door curriculum needs larger-room navigation
training before it can help.

## Submission safety
Champion promoted to `main` (commit `f6465ba`), validator PASSED, pushed. Champion weights
SHA256 `7e0a8d02…b644d7` unchanged. All gate work is isolated on `tier-mastery`.
