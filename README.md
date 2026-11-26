# KEYSTONE-DOOM (DoorPolicyNet Candidate)

Yugaantar 2026 DevForge, Track 1 ("Can It Play DOOM?"). Built from the official DevForge DOOM template.

## Status (evidence-based)

- **Frozen champion preserved** as fallback (`submission/weights/champion_frozen.pt`, `find_return_11.pth.tar`). `main` and `final-submission` not modified by this finalization pass.
- **DoorPolicyNet candidate**: isolated package in `submission_candidate_doorpolicy/`. Matched 120-episode battery (15 maps x 8 eps, seeds 1234-1241, 2100 ticks, render_buffers=True): **120/120 exits** vs champion **88/120**. 0 deaths, no paired damage regression, 64/64 reg, 8/8 gauntlet, 24/24 keys, 32/32 door_simple, 16/16 combat holdout exits. Combat damage/kills unchanged from champion (0 kills, 369 total damage).
- **Validation**: local official `validate.py --device cpu --ticks 420` PASS in clean temp package (init 1.7s, p50 3.1ms, p99 5.4ms, max 82ms cold start). Docker/Linux validator **not run** in this session.
- **Publication**: **NOT PUBLISHED**. Remote publication target ambiguous (no `final-submission` branch on origin observed). No force-push/history rewrite performed.

See `FINAL_SUBMISSION_VERIFICATION.md`, `FINAL_CANDIDATE_MANIFEST.md`, `DOORPOLICY_FINAL_PROMOTION_REPORT.md`, `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`.

## Architecture

- **Champion backbone**: `TransferKeyStoneNet` (ray/vitals/event/RGB features + GRU) with frozen `champion_frozen.pt` and `find_return_11.pth.tar` nav backbone (from 3d_control_deep_rl, MIT).
- **DoorPolicyNet**: learned GRU policy (`door_policy_gru.pt`) distilled from training-time DoorController signals; used at inference only through observation-derived features.
- **Blending/gate**: candidate logit blend uses learned DoorPolicyNet logits/gate; `obs_door_mask` is an observation-derived ray predicate disclosed in compliance audit (gates blending, not a hand-written action state machine).
- **Inference**: observation-only, deterministic argmax given same obs/state, `legalize_action` enforces contract button semantics only. No privileged coords/map state, no network, no training imports, no repo writes.

## Reproducibility / how to run

```bash
# from repo root with env/ on PYTHONPATH
$env:PYTHONPATH=".\env;."   # PowerShell
python validate.py --device cpu --ticks 420
```

Clean package verification performed by copying `submission_candidate_doorpolicy/` to `submission/` (excluding `__pycache__`) and running validator. Full clean git clone + `git lfs pull` + Docker/Linux validation still required before final publication.

## Evaluation methodology

- Matched champion vs candidate: identical maps/seeds/ticks/config. Battery: eval_move_0/1, eval_nav_0/1, eval_combat_0, eval_key_1/2/3, eval_gauntlet_2, door_simple_0-3, combat_holdout_0/1. 8 episodes/map, seeds 1234-1241, 2100 ticks.
- Metrics: exits, deaths, damage, health, timeouts, keys, kills, latency. Seed-level CSV: `transfer/seed_level_comparison_v2.csv`.
- two_rooms is a verification-only diagnostic and **not solved** (0/8); not counted as official gate.

## Limitations

- No combat kill improvement over champion; damage unchanged.
- two_rooms unresolved (post-door multi-room navigation).
- Docker/Linux validator not run in this session.
- Remote LFS/clean-clone publication verification incomplete.
- Official scoring coefficients undisclosed; results are proxy metrics.

## OSS dependencies / licenses

- DevForge DOOM template: cited as competition foundation; no visible software license in starter repo.
- 3d_control_deep_rl / find_return_11 backbone: MIT (Ed Beeching). Verify redistribution terms for packaged weights in final submission context.
- VizDoom/Gymnasium/PyTorch/numpy/scipy/opencv: per their respective licenses.

## Submission entry point

- Candidate: `submission_candidate_doorpolicy/agent.py` (`DoomAgent.reset`, `DoomAgent.act`)
- Champion fallback: `submission/agent.py` and frozen weights under `submission/weights/`

## Safety

- No unlicensed OSS combat model weights/code included. Open-source combat audit is research-only (`OPEN_SOURCE_COMBAT_MODEL_AUDIT.md`).
- No telemetry/RRD/temp training artifacts in candidate submission package.