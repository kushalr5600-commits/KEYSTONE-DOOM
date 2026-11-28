# OPEN_SOURCE_COMBAT_MODEL_AUDIT

## Scope and safety

- **Phase**: research/compatibility audit only. **No training, no integration, no promotion, no submission artifact changes.**
- Preserved: frozen champion, `main`, `final-submission`, DoorPolicyNet candidate, existing submission weights/inference.
- Audit workspace: `C:\Users\kkr\AppData\Local\Temp\opencode\combat_audit` (cloned 2026-11-27, shallow clones).
- Objective: identify best already-trained open-source Doom/VizDoom combat model that could improve KEYSTONE-DOOM without jeopardizing verified DoorPolicyNet candidate.

## KEYSTONE current interface (compatibility target)

| Item | Current KEYSTONE value |
|---|---|
| Observation | Official dict: `rgb` (render buffer), 64 rays/ray hit types/ray distances, vitals, event scalars/features via `FeatureBuilder` (no privileged map/enemy coords at inference) |
| Action | 9-dim discrete `np.int8` action from `macro_action`/`legalize_action` (contract-compliant MultiDiscrete-style buttons) |
| Frame skip | 4 (evaluation/tick semantics used in project) |
| Runtime | CPU 1-core validator budget, <40ms p99 target; init <60s; no network; no training imports |
| Existing nav backbone | `find_return_11.pth.tar` from 3d_control_deep_rl already in champion |
| DoorPolicyNet | Verified isolated candidate; any combat experiment must not modify/regress it |

## Repositories inspected (priority order)

1. Arnold — <https://github.com/glample/Arnold>
2. stable-baselines-agent — <https://github.com/ViZDoomBot/stable-baselines-agent>
3. ContinuousDoom — <https://github.com/bebeal/ContinuousDoom>
4. 3d_control_deep_rl — <https://github.com/edbeeching/3d_control_deep_rl>
5. FreeDoom_Reinforcement_Learning_Game — <https://github.com/1marieCurie/FreeDoom_Reinforcement_Learning_Game>
6. rt-doom — <https://github.com/RelativeDB/rt-doom>

Additional search: web search returned no useful additional pretrained combat checkpoint candidates; no extra repos cloned.

## Per-candidate audit

### 1) Arnold

- **Links**: repo <https://github.com/glample/Arnold>; local `Arnold/README.md`, `src/model/dqn/base.py`, `src/model/utils.py`, `src/doom/actions.py`, `src/doom/game.py`.
- **Model availability**: **Pretrained weights present in repo** under `pretrained/`. No auth. Downloadable by git clone.
  - `deathmatch_shotgun.pth` — 52,192,720 bytes — SHA256 `8A88BCB373A22D4D2CED5BC6A8421403DE54E719ED9621C61CC4E27972005696`
  - `vizdoom_2017_track1.pth` — 52,196,840 bytes — SHA256 `08A59E88FB7C69D27F8326F7B724B6EF876758AA74E4D71686DB637D37C8D36F`
  - `vizdoom_2017_track2.pth` — 52,184,528 bytes — SHA256 `574A93670224A425EA095DA0C106755D4E84CB1901EC529790B809C4BD592F94`
  - `defend_the_center.pth` — 9,859,181 bytes — SHA256 `25A866493CB0D20621A7E743A5C9D6EEF14A6ABFD07E2179020E1F22DC42B6DD`
  - `health_gathering.pth` — 9,765,249 bytes (hash not computed in audit)
- **Architecture**: PyTorch DQN/DRQN family. `DQNModuleBase` builds CNN (`build_CNN_network`), optional game-variable embeddings, optional game-feature head, linear Q head; supports dueling (`proj_action_scores` + `proj_state_values`). Recurrent variants supported by framework (`get_recurrent_module` supports rnn/gru/lstm). Feedforward/recurrent input handling in base.
- **Input modality**: screen buffer (RGB/CRCGCB configurable in game; `n_fm` channels, `height x width`), optional game variables (health, armor, ammo, weapons, selected weapon/ammo, position), optional labels/game features. Not the same as KEYSTONE 64-ray+vitals+RGB contract.
- **Output/action**: `ActionBuilder` creates discrete/continuous action sets; supports action combinations, continuous deltas, crouch; deathmatch/combat includes attack/weapon selection/movement/turning. Not directly KEYSTONE's 9-button macro action encoding.
- **Training scenario**: defend_the_center, health_gathering, deathmatch_shotgun, ViZDoom 2017 track1/track2. Combat-relevant (deathmatch/track).
- **Documented results**: README states won ViZDoom AI Competition 2017 track1/track2; pretrained models provided for visualization/play.
- **License**: **NO LICENSE FILE found in repo**. README has no license section. `raw.githubusercontent.com/.../LICENSE` returned 404. **License unclear; treat as all-rights-reserved/blocked until clarified.** Weight license also unclear.
- **Compliance**: No obvious privileged enemy coords in inference path; uses game variables (health/ammo/weapons/positions in some configs) — positions/privileged vars would be non-compliant if used at inference. No hardcoded aim tree obvious, but action space and training assumptions differ. No network dependency in model file itself. Size ~50MB acceptable.
- **KEYSTONE compatibility**: **C — transferable weights/behavior source only**. Requires license clearance, obs adapter (map KEYSTONE rays/vitals/RGB to Arnold inputs or distill), action remap to 9-button contract, removal of privileged/game-variable dependencies, and careful compliance audit.
- **Risks**: license blocker; action/obs mismatch; may rely on game variables unavailable/undesired; old codebase (Python2/3 era) integration cost.

### 2) stable-baselines-agent (ViZDoomBot)

- **Links**: <https://github.com/ViZDoomBot/stable-baselines-agent>
- **Model availability**: Many `.zip` model files present under `archive_models/` and scenario dirs (git clone showed LFS/filtering activity; 198MB payload). Example: `archive_models/deathmatch/saved_model.zip`, `curiosity_saved_model.zip`. Downloadable via git/LFS; accessibility depends on LFS. No simple auth noted, but LFS required.
- **Architecture**: Stable-Baselines v2 (TF1) PPO2/A2C style; CNN/LSTM policy variants (`common/policies.py`, `larger_policies.py`, `augmented_ppo2.py`). Grayscale frame stack.
- **Input**: `process_frame` to 120x120x1 grayscale; history_length=4; optional attention crop; extra game-variable features in constants (`HEALTH`, `ARMOR`, ammo, weapons, selected weapon). Frame skip 4.
- **Action**: discrete expanded atomic+combo actions via `make_expand_action_f`; deathmatch constants include 26 actions covering move/turn/attack/weapon select/speed combos.
- **Training scenario**: deathmatch/CIG/flatmap/defend scenarios. Combat-relevant.
- **License**: **NO LICENSE FILE found**. Unclear/blocked.
- **Compliance/compatibility**: **D**. TF1/SB1 obsolete; framework dependency heavy; obs includes extra privileged-like game vars; action space mismatch; license unclear. Not worthwhile for hackathon deadline.

### 3) ContinuousDoom

- **Links**: <https://github.com/bebeal/ContinuousDoom>
- **Model availability**: **No in-repo weights**. README links external Google Drive weights and Colab. Not verifiable/in-repo.
- **Architecture**: PPO actor-critic with continuous (mouse) + categorical (keyboard) outputs; `actor_critic.py`, `ppo.py`.
- **Input**: VizDoomGym wrapper; RGB/frame stack (configurable). 
- **Action**: continuous mouse delta + keyboard categorical — **incompatible** with KEYSTONE discrete 9-button action contract.
- **Scenario**: Defend The Center mainly; not deathmatch combat model.
- **License**: **NO LICENSE FILE in repo**. External weights license unknown.
- **Compatibility**: **D — incompatible**. Continuous control and missing licensed in-repo weights.

### 4) 3d_control_deep_rl

- **Links**: <https://github.com/edbeeching/3d_control_deep_rl>
- **Model availability**: Pretrained `.pth.tar` checkpoints in `3dcdrl/saved_models/` (many nav/find_return/item maze checkpoints). Downloadable in repo.
- **Architecture**: CNN + GRU A2C (`models.py`: Conv3/ReLU stack -> linear -> GRUCell -> policy logits + value). RGB 64x112x3 input (`CNNPolicy` test shape). Discrete actions (scenario dependent).
- **Scenario**: navigation/labyrinth/find_return/item collection; not combat.
- **License**: **MIT LICENSE present**. Checkpoint provenance appears repo-internal, but verify redistribution terms before any new use.
- **Compliance**: clean nav model; no combat. 
- **Compatibility**: **D for combat improvement; already leveraged**. `find_return_11.pth.tar` is already used by KEYSTONE champion backbone. No new combat capability.

### 5) FreeDoom_Reinforcement_Learning_Game

- **Links**: <https://github.com/1marieCurie/FreeDoom_Reinforcement_Learning_Game>
- **Model availability**: weights present in repo root. No auth.
  - `doom_ppo_v1_3.zip` — 20,416,799 bytes — SHA256 `5D911967B27D20DEAFE0124329F1C416012AF1DFB3E27C97F7067CFE3C12BAB3` (SB3 zip: `data`, `policy.pth`, `policy.optimizer.pth`, `pytorch_variables.pth`, `_stable_baselines3_version`, `system_info.txt`)
  - `arnolddqn.zip` — 27,707,313 bytes — SHA256 `899BCE659B6C336E7CF4D7828D8AD3E1219FE077D4F5B884A61B28CF651798E4`
- **Architecture**: PPO approach uses Gymnasium + SB3-compatible zip; env gives `Box(0,255,(84,84,3),uint8)` observation and `Discrete(8)` actions: forward, backward, left, right, turn left, turn right, attack, forward+attack. Arnold/DQN variant described in README as Dueling DRQN with LSTM; zip format needs inspection before use.
- **Scenario**: deathmatch/combat. README reports PPO 50% success in constrained arena; Arnold DRQN 16.8 mean kills/episode, K/D 10.68 in larger deathmatch setup.
- **License**: **NO LICENSE FILE found**. README states authors/contact but no reuse license. **Unclear/blocked.**
- **Compliance**: obs/action mismatch with KEYSTONE; uses raw RGB 84x84; no ray/vitals/ray-hit interface. No privileged coords obvious, but training reward shaping and env constraints differ. SB3 dependency manageable but adaptation nontrivial.
- **Compatibility**: **C — possible source after license clearance**. Could be distilled into KEYSTONE action space/obs, but effort high and license blocker present.

### 6) rt-doom

- **Links**: <https://github.com/RelativeDB/rt-doom>
- **Model availability**: **No pretrained Doom-tuned weights in repo**. `README` explicitly says "No model weights are included in this repo"; base RT-J downloads from Hugging Face at runtime. Repo contains demo datasets (`.jsonl.gz`) and training scripts.
- **Architecture**: Relational Transformer (RT-J) agent, fine-tunes existing model; text/relational state encoding, not KEYSTONE RGB/ray/vitals.
- **License**: MIT repo license; RT-J/base model licenses separate.
- **Compatibility**: **D — not an already-trained drop-in combat model**. Heavy runtime, different interface, no ready tuned checkpoint in repo.

## Compatibility matrix

| Candidate | Weights ready | Combat | License | Obs compatible | Action compatible | Runtime fit | Compliance path | Overall |
|---|---|---|---|---|---|---|---|---|
| Arnold deathmatch/track | Yes, in-repo | Yes | **Unclear (no LICENSE)** | Partial (screen+vars) | Partial (discrete/combos) | Maybe | Needs audit/adaptation | **C (technical), blocked by license** |
| stable-baselines-agent | Yes, in-repo/LFS | Yes | **Unclear** | Low (gray stack+vars) | Partial | Poor (TF1/SB1) | Difficult | D |
| ContinuousDoom | External only | Limited | **Unclear** | Low | No (continuous) | Poor | Poor | D |
| 3d_control_deep_rl | Yes | No (nav) | MIT | Medium (RGB) | Partial | Good | Good | D for combat (already used) |
| FreeDoom RL Game | Yes, in-repo | Yes | **Unclear** | Medium (RGB 84x84) | Partial (Discrete 8) | Maybe (SB3) | Needs audit/adaptation | **C, blocked by license** |
| rt-doom | No tuned weights | Research | MIT code, base separate | No | No | Poor | N/A | D |

## Top 3 candidates (by requested criteria)

1. **Arnold — `deathmatch_shotgun.pth` / `vizdoom_2017_track1.pth`**  
   - Real pretrained weights: strong (in-repo). Combat relevance: strong. License certainty: **low/blocker**. Architecture compatibility: medium-low. Runtime: feasible in PyTorch. Adaptation: high. Compliance: possible after audit. Measurable-improvement probability: medium *if license cleared and distilled properly*.
2. **FreeDoom_Reinforcement_Learning_Game — `doom_ppo_v1_3.zip` / `arnolddqn.zip`**  
   - Weights: in-repo. Combat relevance: good. License certainty: **low/blocker**. Compatibility: medium-low (SB3/RGB/Discrete8). Adaptation: high. Compliance: possible after audit. Probability: medium-low.
3. **3d_control_deep_rl**  
   - License/compatibility excellent, but **not combat** and already integrated as nav backbone; no expected combat gain. Ranked third only on reuse safety, not combat impact.

## Recommendation (one candidate for next isolated experiment)

**Recommended candidate (conditional): Arnold `pretrained/deathmatch_shotgun.pth`** — *only after license/weight-rights clearance*.

**Why**: it is the only inspected candidate with in-repo, combat/deathmatch-relevant, PyTorch pretrained weights and documented competition-grade behavior. It is technically the best source for a **distillation/behavior-cloning teacher** or offline action prior in an isolated experiment.

**Blocking conditions before any experiment**:
1. Obtain explicit license/permission for repo code and pretrained weights (current repo has no LICENSE). 
2. Verify weight provenance and redistribution rights.
3. Confirm no privileged inputs are used in extracted teacher behavior; record only permitted observations.
4. Keep experiment isolated from `main/final-submission/champion/DoorPolicyNet`.

**If license cannot be cleared quickly**: do **not** use Arnold/FreeDoom/stable-baselines-agent weights. Fall back to training a small combat policy from scratch in the existing KEYSTONE RL infra (isolated), using only in-house/generated data and compliant observations.

## Explicit integration plan (NOT executed)

For the conditional Arnold candidate, if and only if license is cleared:

1. **Isolation**: create new worktree/branch `combat-audit-experiment` from current integrate baseline; do not touch `main/final-submission/submission`.
2. **Weight audit**: record hash, source commit, license proof; store outside submission until approved.
3. **Teacher wrapper**: run Arnold model in isolated offline evaluator to produce action logits/trajectories from **permitted observations only**; do not import Arnold training/runtime into submission.
4. **Adapter/distillation**: map Arnold observation needs to KEYSTONE-available obs where compliant; otherwise distill teacher outputs into existing TransferKeyStoneNet-compatible student using logged RGB/ray/vitals episodes.
5. **Action mapping**: map to 9-button contract via learned head or offline relabeling; no hand-coded combat aiming rules.
6. **Eval gate**: matched battery vs frozen champion and DoorPolicyNet; require no regression on doors/keys/reg/gauntlet/timeouts/deaths/damage; measure combat kills/damage dealt/exit rate.
7. **Compliance/validator**: package only student weights/code; no Arnold runtime dependency; pass `validate.py`, clean checkout, Docker/Linux.
8. **Promotion**: only if all gates pass and evidence supports improvement; otherwise discard.

## Deliverables produced

- `OPEN_SOURCE_COMBAT_MODEL_AUDIT.md` (this file)
- `combat_model_compatibility.csv`
- No code/model integration, no training, no promotion.

## Evidence / notes

- Cloned repos inspected locally at `C:\Users\kkr\AppData\Local\Temp\opencode\combat_audit`.
- License checks: `3d_control_deep_rl/LICENSE` MIT; `rt-doom/LICENSE` MIT; Arnold/stable-baselines-agent/ContinuousDoom/FreeDoom had **no LICENSE file** in repo root.
- Hashes recorded above for inspected in-repo weight files.
- This audit is compatibility/reuse research, not legal advice; obtain explicit permission before using unclear-license weights.
