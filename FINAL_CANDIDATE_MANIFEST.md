# FINAL_CANDIDATE_MANIFEST

## Status
**CONDITIONAL / NOT PUBLISHED** — candidate verified in clean temp directory; not committed to a remote submission branch. No behavior changes made.

## Candidate identity
- **Repo (local worktree inspected)**: `C:\Users\kkr\Documents\Default Project\KEYSTONE-DOOM-integrate`
- **Current branch**: `integrate-experiment`
- **HEAD commit**: `332cf3b0b72e4695ad5631404805bdbdb485c8e0` (same as local `final-submission`, `door-fix`, `score-opt`, etc.)
- **Working tree**: dirty (unrelated modified/untracked files; candidate package is untracked at `submission_candidate_doorpolicy/`)
- **Remote**: `origin https://github.com/kushalr5600-commits/KEYSTONE-DOOM.git` (fetch/push)
- **Remote branches observed**: `origin/main`, `origin/data-reuse`, `origin/progression-transfer`, `origin/tier-mastery`, `origin/transfer-research` (no `final-submission`/candidate branch observed)
- **Intended publication target**: **AMBIGUOUS** — not specified; remote has no final-submission branch. STOP before publishing.

## Exact evaluated candidate package
Source: `KEYSTONE-DOOM-integrate/submission_candidate_doorpolicy/` (untracked). Hashes match previously evaluated candidate.

| File | SHA256 | Bytes |
|---|---|---:|
| `submission_candidate_doorpolicy/weights/champion_frozen.pt` | `7E0A8D022A3E8DCE04090100211A7A2151792B15E30C69B10AA516AAA0B644D7` | 1,326,425 |
| `submission_candidate_doorpolicy/weights/find_return_11.pth.tar` | `9B15E5818A91FF93A4AF92919E311DCD608B6FE2C0670B7A6FA8AAC69736C87A` | 1,590,097 |
| `submission_candidate_doorpolicy/weights/door_policy_gru.pt` | `1A8F4B979F8ADD03A93A0A9B2CBC17DC901B2FFA79A1D6F9AAFC4FE1A1AF8B55` | 1,049,469 |

## Submission package layout (clean, no __pycache__)
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
- Inference: observation-only; no training imports; no network; no privileged state. Uses `keystone.*`, `transfer.door_policy_runtime`, `transfer.cnn_adapter`.

## Dependencies
- As per repo `requirements.txt` / official validator environment; no extra packages added by candidate package. Validator clean-dir pass used `PYTHONPATH=.\env;.`, Python 3.12, CPU.

## Evaluation references
- `DOORPOLICY_FINAL_PROMOTION_REPORT.md` — CONDITIONAL PROMOTION, matched 120-episode battery, survival/damage audit.
- `transfer/verify_candidate_full_v2.json`, `transfer/verify_champion_full_v2.json`, `transfer/seed_level_comparison_v2.csv`
- `transfer/DOORPOLICY_COMPLIANCE_AUDIT.md`
- Prior local validator pass: 1.8s init, p50 3.4ms, p99 6.2ms, max 17ms.

## Clean verification (this mission)
- Clean temp dir: `C:\Users\kkr\AppData\Local\Temp\opencode\final_verify_clean` (submission package copied without `__pycache__`; env/validate/maps copied for validator).
- `python validate.py --device cpu --ticks 420`: **PASS** (3 warnings: not a git repo, cannot pin CPU, no /proc RAM). Init 1.7s, p50 3.1ms, p99 5.4ms, max 82ms (cold start), 105 steps, 10.1MB.
- Docker/Linux: **NOT RUN** (docker command unavailable/timed out in this session).
- Git LFS: **NOT VERIFIED** (git lfs command timed out/unavailable here); weight files in candidate dir are real binaries with expected sizes/hashes.
- Remote publication: **NOT DONE / AMBIGUOUS TARGET**. No force-push/history rewrite performed.

## Risk / blockers
1. Candidate package is **untracked** and not on a remote submission branch.
2. Remote publication target ambiguous (no `final-submission` on origin; local `final-submission` exists at same SHA as HEAD).
3. Docker/Linux validator not run.
4. Git LFS clean-clone verification from remote not run.
5. `main`/`final-submission` not modified; champion fallback preserved.