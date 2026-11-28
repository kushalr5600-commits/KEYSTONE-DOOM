# FINAL_SUBMISSION_VERIFICATION

## Final status
**CONDITIONAL** — candidate is locally verified in clean temp directory and passes official validator; **not published** because remote publication target is ambiguous and Docker/Linux + remote LFS clean-clone checks are incomplete.

## Identity
- **Local worktree**: `C:\Users\kkr\Documents\Default Project\KEYSTONE-DOOM-integrate`
- **Branch**: `integrate-experiment`
- **Commit SHA**: `332cf3b0b72e4695ad5631404805bdbdb485c8e0`
- **Local branches at same SHA**: `final-submission`, `door-fix`, `score-opt`, `integrate-experiment`, etc. (per `git branch -a`)
- **Remote**: `origin https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git`
- **Remote heads observed**: `main`, `data-reuse`, `progression-transfer`, `tier-mastery`, `transfer-research`; **no `final-submission` remote branch observed**.
- **Candidate package**: `submission_candidate_doorpolicy/` (untracked, not committed/published)

## Weight hashes (exact evaluated candidate)
| Weight | SHA256 |
|---|---|
| `champion_frozen.pt` | `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7` |
| `find_return_11.pth.tar` | `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A` |
| `door_policy_gru.pt` | `1A8F4B979F8ADD03A93A0A9B2CBC17DC901B2FFA79A1D6F9AAFC4FE1A1AF8B55` |

## Clean clone / clean package verification
- **Clean temp package**: `C:\Users\kkr\AppData\Local\Temp\opencode\final_verify_clean`
- **Method**: copied `submission_candidate_doorpolicy` contents into `submission/` excluding `__pycache__`; copied `validate.py, env, maps, keystone, transfer` from integrate for validator; ran with `PYTHONPATH=.\env;.`
- **LFS**: not verified via git-lfs command (command unavailable/timed out). Candidate weight files are real binaries with expected sizes/hashes.
- **Hashes**: verified by file size/hash identity against known evaluated values.
- **Result**: clean package validator **PASS**.

## Official validator
- **Command**: `python validate.py --device cpu --ticks 420` (clean temp dir, PYTHONPATH set)
- **Result**: **PASS** with 3 warnings
- **Warnings**: not a git repo (temp dir); cannot pin CPU; no /proc RAM measurement
- **Init**: 1.7s (limit 60s)
- **Episode**: `two_rooms.wad:MAP01`, 105 steps, tick cap, agent 0.4s / 4.2s budget
- **Latency**: first 82ms, p50 3.1ms, p99 5.4ms, max 82ms (cold start)
- **Repo/package size**: 10.1 MB measured by validator

## Docker/Linux
- **NOT RUN**. `docker --version` timed out/unavailable in this session. Not claimed as pass.

## Compliance audit
- Observation-only inference: PASS (no privileged coords/map state at inference).
- No hardcoded routes/aiming/action-selection tree in submission runtime: PASS (learned DoorPolicyNet + disclosed obs_door_mask gating).
- No training imports/network/writes in submission: PASS by static scan.
- No unlicensed OSS combat weights/code in submission: PASS (champion/frozen/doorpolicy/find_return only; 3d_control nav backbone already used).
- No telemetry/RRD/temp training artifacts in clean submission package: PASS.
- See `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`.

## Repository size
- Candidate submission package (clean): ~4.0MB weights + small code; validator-measured temp package 10.1MB including env/maps/validate.

## Publication status
- **NOT PUBLISHED**. No remote candidate/final-submission branch observed. Publication target ambiguous. No force-push, no history rewrite, no protected branch modification performed.

## Fallback champion
- Frozen champion preserved: `submission/weights/champion_frozen.pt` SHA256 `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7`, `find_return_11.pth.tar` SHA256 `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A`. `main` at `f6465ba`, local `final-submission` at `332cf3b` unchanged.

## Outstanding blockers / exact next action
1. **Commit candidate package to a clear target branch** (e.g. create/push `final-submission` or agreed name) after confirming remote target with repo owner; include only submission-required files, no pycache/telemetry/tmp artifacts. Do not force-push.
2. **Clean git clone from remote + `git lfs pull` + hash verify + validator** on the published branch.
3. **Run Docker/Linux validator** if environment available.
4. After all pass, mark FINAL VERIFIED and record published commit SHA. Keep champion fallback. Do not alter candidate after verification.