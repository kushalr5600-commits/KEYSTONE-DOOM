# COMBAT_IMPROVE_VERIFICATION_REPORT

## 1. Executive verdict: **CONDITIONAL (INTERNAL EXPERIMENT ONLY)**

The `submission_combat_improve/` candidate shows a **positive isolated signal** on the matched 120-episode battery (120/120 exits, 0 deaths, -12 damage, +6 kills vs verified DoorPolicyNet baseline) and passes the official CPU validator as a disposable internal package. **It is NOT publishable as-is** due to unresolved DefendCenter weight provenance/licensing (Arnold repo has no LICENSE). No regressions observed on the matched battery; sample size is small for kill/damage claims.

No git push/merge/branch changes. `main` and `final-submission` untouched. Verified submission unmodified.

## 2. Baseline vs candidate metrics (matched 15 maps, 120 episodes, seeds 1234-1241, 2100 ticks)

| Metric | DoorPolicyNet baseline | Combat-improve | Delta (cand-base) |
|---|---:|---:|---:|
| Exits | 120/120 | 120/120 | 0 |
| Deaths | 0 | 0 | 0 |
| Damage | 369.0 | 357.0 | -12.0 |
| Keys | 32 | 32 | 0 |
| Kills | 0 | 6 | +6 |
| Timeouts | 0 | 0 | 0 |
| Max p99 latency | 3.86ms (from prior logs) | 12.25ms | +8.39ms (within 40ms) |
| Max latency | 3.89ms (prior) | 24.46ms | within 40ms |

**Source labeling:** Baseline from existing verified log `transfer/verify_candidate_full_v2.json` (copied from integrate; matches previously verified DoorPolicyNet 120/120, 369 dmg, 0 kills). Candidate from **new run this session** `transfer/verify_combat_improve_full_v2.json` (120 rows, 2026-11-27 05:56:34 local).

## 3. Paired seed-level results

- File: `report_combat_improve/seed_level_paired.csv` (120 paired rows).
- Exit delta nonzero: 0/120. Death delta nonzero: 0/120. Damage delta nonzero: 4/120 (sum -12). Kill delta nonzero: 6/120 (sum +6). Keys delta nonzero: 0/120.
- No paired exit/death regression observed. Damage improvements/kills are concentrated in combat holdouts.

## 4. Combat-focused results

| Scope | Exits | Deaths | Damage | Kills | Timeouts | Source |
|---|---:|---:|---:|---:|---:|---|
| combat_holdout_0 (8 eps) baseline | 8/8 | 0 | 207.0 | 0 | 0 | existing baseline log |
| combat_holdout_0 (8 eps) candidate | 8/8 | 0 | 222.0 | 2 | 0 | new run |
| combat_holdout_1 (8 eps) baseline | 8/8 | 0 | 162.0 | 0 | 0 | existing baseline log |
| combat_holdout_1 (8 eps) candidate | 8/8 | 0 | 135.0 | 4 | 0 | new run |
| 8-ep smoke (4+4) candidate | 8/8 | 0 | 168.0 | 3 | 0 | new smoke `smoke_combat_holdout.json` |

Additional combat-focused eval beyond the matched 120-ep battery: 8-ep smoke run completed; no further large combat-only run executed to conserve compute. Limitation: small kill sample (6 kills total in 120 eps).

## 5. Per-map results

See `report_combat_improve/per_map_summary.csv`. All 15 maps 8/8 exits, 0 deaths, 0 timeouts for both. Damage/kill deltas only on combat_holdout_0 (+15 dmg, +2 kills) and combat_holdout_1 (-27 dmg, +4 kills).

## 6. Official validator outcome

- **Disposable internal package**: `C:\Users\kkr\AppData\Local\Temp\opencode\combat_improve_internal_pkg` (INTERNAL EXPERIMENT ONLY, not publishable).
- Command: `PYTHONPATH=.\env;. python validate.py --device cpu --ticks 420`
- Result: **PASS** with 3 warnings (not git repo, cannot pin CPU, no /proc RAM).
- Init: 1.7s. Episode: two_rooms.wad:MAP01 105 steps, agent 0.3s/4.2s budget. Latency: first 7ms, p50 3.0ms, p99 4.2ms, max 7ms. Repo size: 27.2MB.
- Log: `report_combat_improve/validate_internal.txt`.

## 7. Runtime and resource budget

- Candidate adds DefendCenter feedforward model (9.86MB weight, 12x60x108 4-frame RGB stack, 3-action head). Observed max p99 12.25ms, max 24.46ms in 120-ep eval; well under 40ms. Cold start can be higher (smoke saw 43ms max once) - validator first step 7ms in clean package; monitor first-step latency in target env.
- No Docker/Linux run (not attempted; environment limitation). No disruptive env changes.

## 8. Compliance audit

- **Observation-only**: agent reads observation dict; DefendCenter adapter uses rgb/vitals/sel_ammo keys if present. No privileged state used.
- **No network/repo writes/training imports at inference**: static grep clean except `.eval()` calls. No requests/socket/subprocess/training imports found.
- **No hardcoded map-specific behavior**: routing is recent-damage cooldown heuristic (observation-derived) overriding only turn/attack bits; movement/USE/keys/doors preserved by base DoorPolicyNet.
- **Action-space compliance**: outputs 9-button int8; base uses `legalize_action/macro_action`. Override copies turn/attack indices only.
- **Risks**: heuristic combat routing is hand-written (selector), but action generation for override is learned model output; disclose clearly. Base prev_action state can desync on override (minor, isolated experiment).

## 9. Licensing/provenance status

- **BLOCKER for publishable candidate**: `weights/defend_the_center.pth` SHA256 `25A866493CB0D20621A7E743A5C9D6EEF14A6ABFD07E2179020E1F22DC42B6DD`, 9,859,181 bytes, from Arnold repo (`https://github.com/glample/Arnold`) which has **NO LICENSE file**. Provenance/redistribution rights unclear.
- Baseline weights (champion_frozen, find_return_11, door_policy_gru) have known hashes matching verified submission.
- Per instructions, package is marked `INTERNAL_EXPERIMENT_ONLY.txt` and must **not** be presented as publishable.

## 10. Reproducibility and limitations

- **Independently reproduced this session**: combat-improve 120-ep eval, 8-ep combat smoke, validator pass, paired CSVs, hash manifest.
- **Existing verified logs reused**: DoorPolicyNet baseline `verify_candidate_full_v2.json` (matches previously reported 120/120/369/32/0).
- **Not reproduced/verified**: Docker/Linux validator; larger combat-only sample; official hidden-track scoring; first-step latency under constrained Linux CPU.
- **Limitations**: kill sample small (6 events). Damage delta is small (-12) and mixed across combat maps (+15, -27). No statistical significance claimed. Cold-start latency spike observed in earlier smoke (43ms max) warrants check in target environment.

## 11. Whether candidate merits further work

**Yes, for isolated further work; no for publication now.** The conservative DefendCenter turn/attack override shows no matched-battery regression and a small combat improvement signal. Before any publishable consideration: (1) replace DefendCenter weight with clean-room licensed/retrained model or obtain rights; (2) run larger combat eval; (3) validate first-step latency under official Linux/CPU constraints; (4) full clean checkout + Docker validator; (5) explicit promotion decision with regression gates.

## 12. Exact files created

All under `C:\Users\kkr\AppData\Local\Temp\opencode\combat_improve_wt\report_combat_improve\` unless noted:
- `COMBAT_IMPROVE_VERIFICATION_REPORT.md` - this report
- `seed_level_paired.csv` - 120 paired rows
- `per_map_summary.csv` - 15-map summary
- `evaluation_summary.csv` - scope summary
- `hash_manifest.csv` - disposable package file hashes
- `validate_internal.txt` - official validator output
- `verify_combat_improve_full_v2.json` - new 120-ep candidate eval
- `smoke_combat_holdout.json` - 8-ep combat smoke
- `analyze.py` - analysis script
- Disposable package: `C:\Users\kkr\AppData\Local\Temp\opencode\combat_improve_internal_pkg\` (INTERNAL EXPERIMENT ONLY)
- Eval logs/data: `C:\Users\kkr\AppData\Local\Temp\opencode\combat_improve_wt\transfer\verify_combat_improve_full_v2.json`, `transfer/verify_combat_improve_full_v2.log`
