# Phase 1 — root-cause analysis (measured, not speculative)

Evidence: `python -m training.diagnose --weights submission/weights/keystone.pt --episodes 4`
(12 episodes over 3 maps), `python -m training.report`, training JSONL logs, and direct
inspection of `keystone/*` and `training/*`. Raw outputs are in `experiments/results.csv`,
`experiments/train_t4v2.jsonl`, `experiments/focused_console.txt`.

## Headline finding: the "exit" target was a monster

`training/exit_finder.py` originally treated UDMF thing type `3004` as an exit. In the
official fixtures and `maps/two_rooms.wad`, `3004` is a **Demon monster** in the UDMF
`zdoom` namespace. Direct probe (`maps/train/_probe_11.wad` vs `_probe_3004.wad`):

| thing type | idle 525 steps |
|---|---|
| 11 (classic exit) | hp 100, no damage |
| 3004 | player takes continuous damage and dies at step 179 |
| 3005 | dies at step 51 |

All earlier exit-distance shaping therefore pushed the policy **toward a monster**, not an
exit. This alone explains a large part of the 0/32 exits, 0.719 deaths/episode and 78.19
damage. Fixed by: UDMF exit **linedefs** (special 243) are checked first; 3004/3005 removed
from `EXIT_TYPES`; generated maps use verified crossable exit lines.

## 1–6. What the policy actually selects

Button counts over 4,555 decisions (12 episodes, legacy Bernoulli policy):

| Button | presses/step | Note |
|---|---:|---|
| FWD | 0.620 | forward IS learned |
| BACK | 0.000 | 215 raw presses suppressed by legalization |
| STRAFE_L | 0.065 | low |
| STRAFE_R | 0.935 | near-constant |
| TURN_L | 0.727 | dominant |
| TURN_R | 0.003 | 681 raw presses suppressed (paired with TURN_L) |
| ATTACK | 0.595 | learned, but aim poor |
| USE | 0.371 | learned; 1,642 raw presses suppressed; **290 door openings** |
| NEXT_WEAPON | 0.385 | 1,736 raw presses suppressed (edge spam) |

- **Forward is learned** but not directed: mean distance-to-exit progress is **−34.5**
  (moving *away*), because the target was a monster and the map geometry was not verified.
- **Turning is learned but pathological**: TURN_L 0.727 vs TURN_R 0.003 after legalization —
  the agent spins left. The policy outputs nearly all nine buttons as "on" (top macro
  strings contain 4–6 buttons), and `legalize_action` discards most of them. This wastes
  most of the action head's learning signal on contradictory combinations.
- **Attack is learned**, 0.595/step, but kills are only 0.062/episode: learned aiming is weak.
- **USE is learned and correctly edge-triggered** (290 verified door openings). It is not
  the bottleneck. The real problem is knowing *where* to go.
- **Stuck/loop rate 0.203** (32-episode eval) with 299 unique ray signatures/episode: motion
  exists but is mostly aimless. On one fixture, mean speed fell to 0.5 units/step with
  constant distance 128 — wedged against geometry.

## 7. Reward balance (root cause #2)

Measured novelty contribution: `unique_ray_sigs ≈ 520` per 525-step episode × 0.02 =
**10.4 reward/episode**, essentially equal to `reward_exit = 10.0`. Meanwhile distance
shaping contributed at most ~0.5 and damage/time penalties were <1. The agent was therefore
optimising * Ray-signature novelty*, i.e. wandering, exactly the failure mode the master
prompt warns about. Corrected to 0.002 in the focused run; still 0 exits because of root
cause #1 and #2 below.

## 8. PPO stability

Logs show entropy falling 6.1 → 5.0 over 20k steps (focused run), policy loss oscillating
[−1.12, +0.80], value loss occasionally spiking (0.43, 0.77) but no divergence. Updates are
numerically stable; instability is not the primary cause. The high-entropy plateau is
consistent with learning almost nothing useful from the 512-combination action space.

## 9. Action masking / combinations (root cause #3)

`legalize_action` suppressed: BACK 215, STRAFE_R 220, TURN_R 681, USE 1,642, NEXT_WEAPON
1,736 presses. The 9-Bernoulli head almost never produced a clean, purposeful macro
(e.g. "forward+turn right"); legalization then arbitrarily kept the *first* of each
contradictory pair, creating a strong left-turn/strafe-right bias. **Fix:** a learned
categorical over 17 legal macro actions (`keystone/actions.py:MACRO_ACTIONS`), which keeps
decisions from trained weights but eliminates contradictory combos by construction.

## 10. Encoder

Features are finite, correctly shaped (1249 dims) and vary across observations; hidden-state
activity is 0.113 mean std (nonzero but modest). The encoder is not obviously broken. It
cannot, however, infer an exit the observation never exposes — shaping must supply the
gradient, which it failed to do because of root cause #1.

## 11. Distance-to-exit shaping

`find_exit` returned coordinates for the training maps, and `RewardShaper` used them, but the
coordinates were monsters. A second bug: the first formulation `k(d−γd')` paid a positive
reward for standing still (γd' baseline). Both fixed. The current formulation is
`k(d_prev−d_now)/128`, zero/negative for loops, with *verified* linedef exit coordinates.

## 12. Exploration reward

Quantified above: it rivalled exit and was the dominant learned incentive. Reduced 10×.

## 13. Death timing

Evaluation deaths occur by the end of episodes (final `privileged_state().dead`), with mean
damage 78.19. With the misidentified monster target this is expected: the agent was
navigating into a Demon. Generated movement/nav maps have no monsters and the oracle
verifies 12–26-step exits with 100 hp, so early curriculum death is avoidable.

## 14. Map learnability

Original fixtures are tiny and several have **no exit at all** (verified via exit-linedef
scan), while others hide the exit behind unverified geometry. `training/mapgen.py` now
generates maps from known-good fixture geometry, and `training/verify_maps.py` drives an
oracle (privileged steering, training-only) to prove each objective achievable. 15/16
generated maps are verified reachable in 12–25 steps (one key variant excluded); the
verified set is used for the curriculum, and a separate held-out set is used for evaluation.

## Priority order for recovery

1. Correct exit target (done).
2. Discrete macro action head (done).
3. Verified reachable curriculum maps (done).
4. Rebalance novelty vs exit/progression (done).
5. Retrain simple baseline, measure, then add memory components one at a time.
