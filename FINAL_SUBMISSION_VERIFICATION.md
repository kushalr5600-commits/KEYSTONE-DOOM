# FINAL_SUBMISSION_VERIFICATION

## Final status
**FINAL VERIFIED** (clean remote clone + LFS/weights + official CPU validator passed). Docker/Linux validator not run in this environment (unavailable/timed out); not claimed.

## Published identity
- **Remote**: `origin https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git`
- **Published branch**: `final-submission`
- **Remote commit SHA**: `5e7c45487f570bfcf52bc8d5bcdd95de24ec6d70` (code/weights publication commit)
- **Remote verification**: `git ls-remote --heads origin final-submission` matches local; fresh clone verified at same commit.
- **Publication method**: non-force push from isolated temp clone; `main` untouched.

## Weight hashes (verified in fresh remote clone)
| Weight | SHA256 | Bytes | LFS pointer |
|---|---|---:|---:|
| `submission/weights/champion_frozen.pt` | `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7` | 1,326,425 | False |
| `submission/weights/find_return_11.pth.tar` | `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A` | 1,590,097 | False |
| `submission/weights/door_policy_gru.pt` | `1A8F4B979F8ADD03A93A0A9B2CBC17DC901B2FFA79A1D6F9AAFC4FE1A1AF8B55` | 1,049,469 | False |

## Clean remote verification
- **Fresh clone**: `git clone --branch final-submission --single-branch https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git` (temp dir) — OK.
- **Git LFS**: `git lfs pull` completed; weights are real binaries, not LFS pointers. door_policy_gru.pt tracked by LFS; others present as real binaries.
- **Validator command**: `PYTHONPATH=.\env;. python validate.py --device cpu --ticks 420`
- **Result**: **PASS** (2 warnings: cannot pin CPU on Windows, no /proc RAM)
- **Init**: 1.6s (limit 60s)
- **Episode**: `two_rooms.wad:MAP01`, 105 steps, tick cap, agent 0.3s / 4.2s budget
- **Latency**: first 7ms, p50 2.9ms, p99 4.6ms, max 7ms
- **Repo size**: 7.2 MB tracked (limit 1.0 GB)

## Docker/Linux
- **NOT RUN** — docker unavailable/timed out in this session. Not claimed.

## Compliance summary
- Observation-only inference; no privileged coords/map state; no hardcoded routes/aiming/action-selection state machine.
- Learned DoorPolicyNet used; `obs_door_mask` observation-derived and disclosed.
- No training imports, no network, no repo writes required.
- No unlicensed OSS combat weights/code in submission package.
- No telemetry/RRD/temp training artifacts in submission package.

## Fallback champion
- Preserved unchanged: `champion_frozen.pt` / `find_return_11.pth.tar` hashes above; local `main` at `f6465ba`, original local `final-submission` base `332cf3b`.

## Remaining limitations
- Docker/Linux validator not run.
- two_rooms 0/8 (verification-only diagnostic; post-door multi-room navigation unresolved).
- No combat kill improvement; combat damage unchanged from champion.
- Official scoring coefficients undisclosed; results are proxy metrics.