# Component-level score dashboard (proxy, not official score)

Official scoring categories from `docs/EVALUATION.md:38-49`: exit (largest + health/time bonus),
progression (keys, doors, unlocks), exploration (unique areas, privately tracked), combat
(kills minus damage/death penalties). Exact weights/tier weights are **not disclosed**, so no
official score is computed here.

## Component metrics (from `transfer/component_results.csv`)

The component evaluator records per episode: exit, steps, health at exit, damage, kills,
deaths, keys, items, doors opened, locks unlocked, unique ray signatures (exploration proxy),
revisit rate, USE attempts/success/failure, latency p50/p99. Privileged state is used only for
reporting, never as agent input.

## Tier proxy results

| Tier | Benchmark | Champion | Learned gate | Notes |
|---|---|---:|---:|---|
| 1 | combat corridor ×3 | 24/24 exits, 0 kills, 0 damage | same | combat score opportunity remains |
| 2 | single-door ×4 | 0/32 | **32/32** | gate solves interaction |
| 2 | large-room navigation ×8 | 64/64 | 64/64 | not a bottleneck once rooms are open |
| 2 | multi-door `two_rooms` ×4 | 0/32 | 0/32 | navigation after doors still fails |
| 3 | key/lock maps ×3 | 48/48 | 48/48 | preserved |
| 4 | gauntlet proxy | 8/8 | 8/8 | preserved |

## Experiments in this branch

| Candidate | Change | Result | Decision |
|---|---|---|---|
| `nav_candidate.pt` | BC with 8 oracle-verified large-room maps mixed into replay | 64/64 large maps, but regression suite 38/62 (key maps collapsed) | rejected; champion already 64/64 on large maps |
| champion | no change | 64/64 large, 64/64 regression | retained |
| learned gate (from `tier-mastery`) | trained intervention gate + door-ray validity mask | 32/32 single doors, 64/64 regression, 8/8 gauntlet | retained as experimental candidate |

## Protected artifacts
- `main`: frozen champion SHA256 `7e0a8d02…b644d7`, validator PASSED, pushed (`f6465ba`).
- `tier-mastery`: learned gate candidate (`2e22609`), pushed.
- `score-max`: this dashboard and component/infrastructure work; champion checkpoint unchanged.

## Remaining score opportunities (ranked)
1. Tier 2 multi-door large-room navigation (0/32) — needs a teacher that can actually traverse the geometry.
2. Tier 1 kills (0) — combat demos with engaged enemies; must preserve exits/survival.
3. Exploration component — unique-area tracking unavailable locally; ray-signature proxy recorded.
4. Official hidden tiers/score — unverified by design.
