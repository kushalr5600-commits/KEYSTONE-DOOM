# DOORPOLICY_FINAL_PROMOTION_REPORT

## 0. Safety / isolation

- **Frozen champion preserved**: `submission/weights/champion_frozen.pt` unchanged. `main` and `final-submission` not modified. No existing submission files overwritten.
- **Candidate evaluated as isolated package**: `submission_candidate_doorpolicy/` in `KEYSTONE-DOOM-integrate` (branch `integrate-experiment`, base 332cf3b). Temp validator copy used at `C:\Users\kkr\AppData\Local\Temp\opencode\doorpolicy_validate`.
- **No new training**. No hand-coded navigation/combat/aiming/action-selection rules added. No privileged inference. Telemetry/eval artifacts kept separate.

## 1. Starting commits / hashes

| Artifact | Value |
|---|---|
| main | `f6465ba` |
| final-submission | `332cf3b` |
| maxscore/score-opt | `332cf3b` |
| integrate-experiment | `332cf3b` (working branch) |
| door-fix | `332cf3b` |
| Champion frozen weight | `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7` |
| Champion find_return | `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A` |
| DoorPolicyNet weight | `1A8F4B979F8ADD03A93A0A9B2CBC17DC901B2FFA79A1D6F9AAFC4FE1A1AF8B55` |
| Anchored combat (NOT promoted) | `B2E2177D54C725AF9EC7C5A3A043888390CF5D6C456E0EADE2521FEE861DB721` |

## 2. Matched champion-vs-candidate evaluation

**Identical battery, seeds, episode ticks (2100), render_buffers=True, 8 eps/map, seeds 1234-1241.**  
Battery (15 maps, 120 episodes): eval_move_0/1, eval_nav_0/1, eval_combat_0, eval_key_1/2/3, eval_gauntlet_2, door_simple_0-3, combat_holdout_0/1.  
Sources: `transfer/verify_candidate_full_v2.json`, `transfer/verify_champion_full_v2.json`, seed-level `transfer/seed_level_comparison_v2.csv`.

### Aggregate comparison

| Metric | Champion | DoorPolicyNet | Difference (Cand-Champ) |
|---|---:|---:|---:|
| Exits | 88 | 120 | +32 |
| Exit rate | 0.733 | 1.000 | +0.267 |
| Deaths | 0 | 0 | 0 |
| Death rate | 0.000 | 0.000 | 0.000 |
| Mean damage/episode | 3.075 | 3.075 | 0.000 |
| Median damage/episode | 0.000 | 0.000 | 0.000 |
| Damage per successful exit | 4.193 | 3.075 | -1.118 |
| Mean health at termination | 96.925 | 96.925 | 0.000 |
| Mean health at successful exit | 95.807 | 96.925 | +1.118 |
| Timeout rate | 0.267 | 0.000 | -0.267 |
| Mean steps per exit | 25.625 | 33.458 | +7.833 |
| Keys collected (total) | 32 | 32 | 0 |
| Kills (total) | 0 | 0 | 0 |

**Paired seed summary**: exit_better=32, exit_worse=0, exit_same=88, dmg_better=0, dmg_worse=0, dmg_same=120. No paired damage regression; all exit gains are door_simple.

## 3. Per-map survival/damage/death breakdown

| Map | C exits | H exits | C deaths | H deaths | C dmg mean | H dmg mean | C hp mean | H hp mean | C tout | H tout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combat_holdout_0.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 25.88 | 25.88 | 74.12 | 74.12 | 0 | 0 |
| combat_holdout_1.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 20.25 | 20.25 | 79.75 | 79.75 | 0 | 0 |
| door_simple_0.wad:MAP01 | 8/8 | 0/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 8 |
| door_simple_1.wad:MAP01 | 8/8 | 0/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 8 |
| door_simple_2.wad:MAP01 | 8/8 | 0/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 8 |
| door_simple_3.wad:MAP01 | 8/8 | 0/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 8 |
| eval_combat_0.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_gauntlet_2.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_key_1.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_key_2.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_key_3.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_move_0.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_move_1.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_nav_0.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |
| eval_nav_1.wad:MAP01 | 8/8 | 8/8 | 0 | 0 | 0.00 | 0.00 | 100.00 | 100.00 | 0 | 0 |

## 4. Death/high-damage analysis

- **Deaths**: 0 for both agents across 120/120 episodes. No death regressions.
- **Damage**: all observed damage is in combat_holdout_0/1 only; identical per paired seed (dmg_same=120). No new damage on door_simple or reg/key/gauntlet maps.
- **Interpretation**: candidate does not increase exposure. Exit gains are from door_simple where no damage occurs. Combat damage unchanged; candidate still takes damage in combat holdouts but exits (no combat kills improvement).
- **Limitation**: only 8 episodes/map; no statistical confidence intervals computed beyond paired counts. Damage semantics from `privileged_state.damage_taken`; health from `privileged_state.health`.

## 5. Candidate performance / regression audit

| Check | Result |
|---|---|
| 120/120 on recorded battery | PASS (v2 with health) |
| 32/32 door_simple | PASS |
| Standard reg 64/64 (move/nav/combat_0/key_1-3) | PASS |
| Gauntlet 8/8 | PASS |
| Keys 24/24 (3 key maps * 8) | PASS |
| Combat holdout exits 16/16 | PASS, no regression |
| Combat kills/damage | 0 kills, 369 dmg total same as champion; no improvement, no regression |
| Survival | 0 deaths, 0 timeouts vs champion 0 deaths, 32 timeouts |
| two_rooms (verification-only) | 0/8, no improvement; separate diagnostic, not used as official criterion |

## 6. Compliance audit

See `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`. Recheck summary:
- Observation-only inference; no privileged coords/map state; no network; no training imports.
- No hand-written action state machine in runtime. `DoorController` training-only, not imported.
- Learned DoorPolicyNet logits+gate used in action selection; `obs_door_mask` is observation-derived ray predicate (disclosed). It gates blending only.
- `legalize_action` enforces contract button semantics only.
- Deterministic argmax given same obs/state. Weights packaged and loaded.

## 7. Runtime and validator

**Official validate.py (temp candidate package, cpu, ticks 420)**: PASS with 3 expected warnings (not git repo, cannot pin CPU, no /proc RAM).  
- Init: 1.8s (limit 60s)  
- Episode two_rooms.wad:MAP01: 105 steps, tick cap, agent time 0.4s of 4.2s budget  
- Latency: first 6ms, p50 3.4ms, p99 6.2ms, max 17ms (<40ms)  
- Repo size: 10.0MB

**Not run**: Docker/Linux official validator - environment limitation; report honestly as not verified.

## 8. Clean checkout / reproducibility

- **Not fully run in this session**. Temp-dir validator pass indicates package is runnable; but full clean clone + LFS + dependency install + git status + remote publication not completed.
- Required before final promotion/publication: clean clone, `git lfs pull`, hash verify, `python validate.py --device cpu --ticks 420`, Docker if available, confirm intended remote/default branch.

## 9. Promotion decision

## CONDITIONAL PROMOTION

**Rationale**: DoorPolicyNet is a strict improvement on the matched battery with no survival/damage/key/combat-exit regression. It passes local official validator, compliance audit (with disclosed obs mask), and runtime budget. Missing: clean checkout + Docker/Linux validation + remote publication verification.

- **Promote if** clean checkout and Docker/Linux validator pass, hashes match, and packaging is verified on target environment.
- **Keep champion fallback** until those are complete.
- **Do not** modify `main` or `final-submission` before verification.

## 10. Deliverables

1. This report: `DOORPOLICY_FINAL_PROMOTION_REPORT.md`
2. Matched comparison: section 2/3 above
3. Per-map breakdown: section 3
4. Seed-level CSV: `transfer/seed_level_comparison_v2.csv`
5. Death/high-damage analysis: section 4
6. Compliance audit: `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`
7. Validator: section 7, temp run PASS
8. Clean checkout: section 8, incomplete
9. Weight hashes: section 1
10. Decision: **CONDITIONAL PROMOTION** — strong evidence, pending clean checkout/Docker/publication verification.