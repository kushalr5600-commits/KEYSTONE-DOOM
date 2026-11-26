# KEYSTONE-DOOM (DoorPolicyNet Candidate)

Yugaantar 2026 DevForge, Track 1 ("Can It Play DOOM?"). Built from the official DevForge DOOM template.

## Status (evidence-based)

- **Frozen champion preserved** as fallback (`submission/weights/champion_frozen.pt`, `find_return_11.pth.tar`). `main` untouched.
- **DoorPolicyNet candidate**: published to `origin/final-submission` (authorized, non-force push). Matched 120-episode battery (15 maps x 8 eps, seeds 1234-1241, 2100 ticks, render_buffers=True): **120/120 exits** vs champion **88/120**. 0 deaths, no paired damage regression, 64/64 reg, 8/8 gauntlet, 24/24 keys, 32/32 door_simple, 16/16 combat holdout exits. Combat damage/kills unchanged from champion (0 kills, 369 total damage).
- **Validation**: fresh remote clean clone + LFS/weight hash verification + `python validate.py --device cpu --ticks 420` **PASS** (init 1.6s, p50 2.9ms, p99 4.6ms, max 7ms). Docker/Linux validator **not run** in this session.
- **Publication**: **PUBLISHED** to `origin/final-submission` at commit `5e7c45487f570bfcf52bc8d5bcdd95de24ec6d70` (doc update follows). No force-push; `main` untouched.

See `FINAL_SUBMISSION_VERIFICATION.md`, `FINAL_CANDIDATE_MANIFEST.md`, `DOORPOLICY_FINAL_PROMOTION_REPORT.md`, `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`.

## Architecture

- **Champion backbone**: `TransferKeyStoneNet` (ray/vitals/event/RGB features + GRU) with frozen `champion_frozen.pt` and `find_return_11.pth.tar` nav backbone (from 3d_control_deep_rl, MIT).
- **DoorPolicyNet**: learned GRU policy (`door_policy_gru.pt`) distilled from training-time DoorController signals; used at inference only through observation-derived features.
- **Blending/gate**: candidate logit blend uses learned DoorPolicyNet logits/gate; `obs_door_mask` is an observation-derived ray predicate disclosed in compliance audit (gates blending, not a hand-written action state machine).
- **Inference**: observation-only, deterministic argmax given same obs/state, `legalize_action` enforces contract button semantics only. No privileged coords/map state, no network, no training imports, no repo writes.

## Reproducibility / how to run

```bash
git clone --branch final-submission --single-branch https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git
cd KEYSTONE-DOOM
git lfs pull
$env:PYTHONPATH=".\env;."   # PowerShell
python validate.py --device cpu --ticks 420
```

## Evaluation methodology

- Matched champion vs candidate: identical maps/seeds/ticks/config. Battery: eval_move_0/1, eval_nav_0/1, eval_combat_0, eval_key_1/2/3, eval_gauntlet_2, door_simple_0-3, combat_holdout_0/1. 8 episodes/map, seeds 1234-1241, 2100 ticks.
- Metrics: exits, deaths, damage, health, timeouts, keys, kills, latency. Seed-level CSV: `transfer/seed_level_comparison_v2.csv`.
- two_rooms is a verification-only diagnostic and **not solved** (0/8); not counted as official gate.

## Limitations

- No combat kill improvement over champion; damage unchanged.
- two_rooms unresolved (post-door multi-room navigation).
- Docker/Linux validator not run in this session.
- Official scoring coefficients undisclosed; results are proxy metrics.

## OSS dependencies / licenses

- DevForge DOOM template: cited as competition foundation; no visible software license in starter repo.
- 3d_control_deep_rl / find_return_11 backbone: MIT (Ed Beeching). Verify redistribution terms for packaged weights in final submission context.
- VizDoom/Gymnasium/PyTorch/numpy/scipy/opencv: per their respective licenses.

## Submission entry point

- `submission/agent.py` (`DoomAgent.reset`, `DoomAgent.act`)
- Weights: `submission/weights/champion_frozen.pt`, `find_return_11.pth.tar`, `door_policy_gru.pt`

## Safety

- No unlicensed OSS combat model weights/code included. Open-source combat audit is research-only (`OPEN_SOURCE_COMBAT_MODEL_AUDIT.md`).
- No telemetry/RRD/temp training artifacts in submission package.