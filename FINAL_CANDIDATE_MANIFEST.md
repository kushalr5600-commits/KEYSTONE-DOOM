# FINAL_CANDIDATE_MANIFEST

## Status
**FINAL VERIFIED (CPU validator + fresh remote clone)** — published to `origin/final-submission`. Docker/Linux not run. No behavior changes made.

## Candidate identity
- **Repo (published)**: `https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git`
- **Branch**: `final-submission`
- **Published commit (code/weights)**: `5e7c45487f570bfcf52bc8d5bcdd95de24ec6d70`
- **Remote verification**: fresh clone of `origin/final-submission` verified at matching commit; hashes + validator passed.
- **Publication target**: `origin/final-submission` (explicitly authorized). Non-force push; `main` untouched.

## Exact evaluated candidate package
Source: `KEYSTONE-DOOM-integrate/submission_candidate_doorpolicy/` (evaluated). Hashes match published submission.

| File | SHA256 | Bytes |
|---|---|---:|
| `submission/weights/champion_frozen.pt` | `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7` | 1,326,425 |
| `submission/weights/find_return_11.pth.tar` | `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A` | 1,590,097 |
| `submission/weights/door_policy_gru.pt` | `1A8F4B979F8ADD03A93A0A9B2CBC17DC901B2FFA79A1D6F9AAFC4FE1A1AF8B55` | 1,049,469 |

## Submission package layout
```
submission/
  agent.py
  keystone/
    __init__.py
    actions.py
    config.py
    features.py
    memory.py
    model.py
  transfer/
    cnn_adapter.py
    door_policy.py
    door_policy_runtime.py
  weights/
    champion_frozen.pt
    find_return_11.pth.tar
    door_policy_gru.pt
```

## Entry point / interface
- `submission/agent.py` — `DoomAgent(observation_space, action_space, device)` with `reset()` and `act(observation)`.
- Observation-only; no training imports; no network; no privileged state.

## Verification
- **Fresh remote clone**: PASS (`git clone --branch final-submission --single-branch ...`).
- **LFS/binaries**: PASS — real binaries, not LFS pointers; hashes match.
- **Official validator**: PASS — `python validate.py --device cpu --ticks 420`, init 1.6s, p50 2.9ms, p99 4.6ms, max 7ms, 2 warnings (CPU pin /proc RAM unavailable on Windows).
- **Docker/Linux**: NOT RUN.

## Evaluation references
- `DOORPOLICY_FINAL_PROMOTION_REPORT.md`
- `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`
- `transfer/seed_level_comparison_v2.csv`