# Progression breakthrough report (branch `progression-transfer`)

## Integrated candidate

- `submission/agent.py` = isolated transfer agent using `TransferKeyStoneNet` with the MIT
  pretrained conv stem and checkpoint `transfer/weights/transfer_prog_bal.pt`.
- Official validator on this branch: **PASSED** (3 Windows-only warnings), init 1.7 s,
  first act 7 ms, p50 2.7 ms, p99 6.6 ms (budget 40 ms), repo 11.0 MB << 1 GB.
- Visual branch confirmed active (`visual_proj` max |w| = 0.032, `use_visual=True`);
  action output is legal `MultiBinary(9)` int8.

## Progression failure diagnosis (measured trace, `diagnose_progression.py`)

On held-out `eval_key_1`, the previous transfer candidate:
- acquired key at step 6;
- saw a locked-door ray for 519 of 525 steps;
- **pressed USE zero times** and held "forward" (macro 1) for all 525 steps;
- goal head stayed on `pursue_key` (519/525), confidence saturated at 1.0;
- ended at (240,96), wedged against the locked door.

So the bottleneck was the **learned action/goal policy**, not observation, key detection or
memory storage.

## Why BC failed and the fix

Teacher demos on key maps contained only **64/13,664 (0.5%)** USE-labelled transitions;
plain BC ignored the rare action. `transfer/build_progression_dataset.py` upsamples USE
transitions 40× and downsamples routine turning, producing 5,560 samples with 46% USE.
Three epochs of BC at lr 2e-4 from `transfer_bc.pt` produced `transfer_prog_bal.pt`.

After the fix, the trace shows USE at step 0, 8 USE presses, INTERACT_OK/FAIL events, goal
switch to `exit`, and **exit in 24 steps** (was 525 with no exit).

## Held-out results (same seeds/maps, deterministic argmax)

| Candidate | Original 6-map suite | New unseen key maps | Notes |
|---|---:|---:|---|
| baseline2 (protected) | 32/96 = 0.333 | 0/16 | reference |
| transfer_find_return_11 | 80/96 = 0.833 | 0/16 | nav/combat gain only |
| transfer + plain progression BC | 40/48 | 0/8 | no gain |
| **transfer + balanced USE demos** | **48/48 = 1.000** | **16/16 = 1.000** | 24-step key→door→exit |

Per-map for the final candidate: 8/8 on both move, both nav, combat, key_1; 8/8 on each of
the two newly generated oracle-verified key maps (`eval_key_2`, `eval_key_3`). No
navigation/combat regression. Latency p50 ≈1.0–1.3 ms, p99 ≈1.9–2.2 ms.

**16-episode full-suite confirmation:** `final_prog16` achieved **128/128 exits (1.000)**
across all eight holdout maps (including both new independently generated key maps), with
zero deaths and zero damage. One first-episode p99 latency spike of 71 ms was observed during
warm-up; subsequent p99 values were ~1.5–4 ms, all far below the 5 s per-action timeout and
the 40 ms average budget. Full results: `transfer/final_eval16.log`,
`transfer/transfer_results.csv`.

## Experiments kept / rejected

- Kept: MIT conv-stem transfer, oracle RGB demos, balanced USE oversampling.
- Rejected: bearing shaping (entropy collapse), plain progression BC (USE too rare),
  BC-only transfer without visual branch (inferior).
- Not yet demonstrated: Tier 4 Full Gauntlet on hidden maps; official score; Linux/docker
  validation. These are explicitly unverified.

## Compliance

- Teacher uses privileged map/position data only to label training observations; it is not in
  the deployed path. Student sees only the official observation dict.
- No hardcoded routes, map coordinates, evaluator access, hidden maps or scripted navigation.
- MIT licence preserved (`transfer/licenses/EDBEECHING_LICENSE.txt`), Ed Beeching 2019 cited.
- New held-out key maps were generated from training geometry and verified by an oracle; they
  were never used in training.
