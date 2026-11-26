# Tier-oriented score proxy dashboard (branch `tier-mastery`)

Official scoring coefficients/tier weights are **not disclosed** (`docs/EVALUATION.md:3-4`), so
this is a transparent local proxy, not an official score estimate.

## Tier capability matrix (from official docs)

| Tier | Documented challenge | Our local proxy | Champion status |
|---|---|---|---|
| 1 Combat & Survival | aim, dodge, health/ammo | `maps/tiers/t1_combat_*` (1–3 imps in corridor, exit) | 24/24 exits, 0 kills, 0 damage — completion yes, combat score unproven |
| 2 Multi-Room Exploration | partial observability, memory, doors, dead ends | `maps/train/door_*` (two_rooms geometry, 2–3 normal doors) | 0/32 exits — **major bottleneck** |
| 3 Long-Horizon Progression | keycard, remember locked door, backtrack, exit | `maps/holdout/eval_key_{1,2,3}.wad` | 48/48 exits, keys 48/48 |
| 4 Full Gauntlet | all above + ambush/scarcity/multiple locks | `maps/holdout/eval_gauntlet_2.wad` (key+lock+imp) | 8/8 exits, 0 damage/deaths |

## Score-relevant proxy metrics (champion frozen)

| Benchmark | Episodes | Exit | Keys | Kills | Damage | Deaths | Stuck | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tier1 combat (3 maps) | 24 | 1.00 | – | 0.00 | 0.0 | 0 | 0.08–0.10 | ~1.4 |
| Tier2 doors (4 maps) | 32 | 0.00 | 0 | 0 | 0.0 | 0 | ~0.99 | ~1.1 |
| Tier3 keys (3 maps) | 48 | 1.00 | 1.00 | 0 | 0.0 | 0 | 0.17 | ~1.2 |
| Tier4 proxy | 8 | 1.00 | 1.00 | 0 | 0.0 | 0 | 0.09 | 1.1 |

## Improvement loop log

| Change | Hypothesis | Baseline | Result | Decision |
|---|---|---|---|---|
| Harder combat maps | Test combat score under threat | 24/24 exits but 0 kills | still 0 kills/damage; exits fast | keep as control, no combat reward signal |
| Normal-door demos + 40% USE oversampling | Teach USE on event-2 doors | 0/32 tier2 | 8/8 on one door_simple variant, 0 on others; door bouncing | rejected |
| Same demos with 12% USE | Avoid door reversal | 0/32 tier2 | 0 USE presses, stuck | rejected |
| Champion | freeze | 128/128 generated holdout | unchanged | **kept as control** |

Tier2 door interaction remains the highest-impact verified bottleneck. The underlying issue is
that the reactive champion sees a door ray for hundreds of steps but either never presses USE
or presses it on alternate steps, reversing the door. A sequence-aware policy or a
curriculum with explicit door-opening state labels is the next plausible intervention.

## Generalization / compliance

- All new tier maps were generated from verified fixture geometry with oracle reachability
  checks; evaluation seeds (1234+) were never used in training (9000+).
- Champion and rollbacks preserved; no evaluator or hidden-state access.
