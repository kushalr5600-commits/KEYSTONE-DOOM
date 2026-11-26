# Research and license audit

Verification date: 2026-11-26. Sources: GitHub REST API (`license` field + tree listings),
raw LICENSE/README/source files from each repository, arXiv metadata, and the local template
clone. **No reference code was copied into this repository.** Where a license could not be
verified from a LICENSE file it is flagged, not guessed.

## Target interface compatibility baseline

Official contract (`docs/CONTRACT.md`, `env/doom_env/spaces.py:73-95`): Dict observation with
`rgb` (HUD hidden), `depth`, 17-dim `vitals`, 64 ego-centric rays, 32 masked entities;
`MultiBinary(9)` actions held 4 tics; no global position, no engine handle at evaluation,
no network, read-only repo, 2 GB VRAM / 8 GB RAM / 1 CPU thread, ViZDoom 1.3.3 + FreeDoom2.
Any reference relying on `POSITION_*`, `ANGLE`, automap, HUD keycard pixels or game variables
cannot be deployed as-is.

## Per-reference findings

### 1. Official starter — https://github.com/Reinforce-SST/devforge-doom-template

- **License: NONE FOUND.** GitHub API `"license": null`; no LICENSE/NOTICE at root; raw
  `LICENSE` → 404; local clone has none. Do **not** claim MIT or any other grant. Organizers'
  intent was not verified.
- Useful files: `env/doom_env/` (contract as code, sensors, wrapper, limits, remote protocol),
  `validate.py`, `docs/`, `submission/agent.py`, `training/`, `maps/two_rooms.wad`.
- Dependencies: Python ≥3.13, `vizdoom==1.3.3`, `gymnasium>=1.0`, `numpy>=2.0`, editable
  `doom-env`, pytest (dev).
- Compatibility: exact — it *is* the interface. Reuse the wrapper/contract/validator as
  provided; we wrote all agent, memory and training code.
- Attribution: cite repository URL as competition foundation; no copyright licence visible.
- Reason not to copy more: no explicit licence, and copying scaffolding supplies no novelty.

### 2. https://github.com/edbeeching/3d_control_deep_rl

- **License: MIT**, confirmed from `LICENSE`: `Copyright (c) 2019 Ed Beeching`, standard MIT
  text. GitHub API agrees (`spdx_id: MIT`).
- Useful ideas: `3dcdrl/models.py` (CNN+GRUCell policy, orthogonal init), `multi_env.py`
  (GPU-batched A2C), `generate_scenario.py` + `scenario_generation/*` (procedural labyrinth /
  find-and-return WADs with train/test splits). "Find and return" is conceptually close to
  our key→door→exit revisit hypothesis.
- Dependencies (their `requirements.txt`): `torch==0.4.1`, `vizdoom==1.1.4`, `numpy==1.15.4`,
  `opencv-python==3.4.3.18`, `omgifol`, `matplotlib`, `moviepy`, `visdom`, `pyzmq`, Python 3.5+.
- Compatibility: low. Returns RGB `(3,64,112)` only and reads `POSITION_*`/`ANGLE` game
  variables; custom discrete actions; no rays/vitals/entities or Gymnasium MultiBinary(9).
- Reuse vs rewrite: we re-derive the *idea* of procedural training maps and a compact GRU
  policy; env, observation pipeline, action mapping and dependencies are rewritten.
- Attribution if reused: retain MIT copyright + permission notice; cite `beeching2020baselines`.
- **Rejected for code reuse:** stack is ~2019 (torch 0.4.1 / vizdoom 1.1.4 / Python 3.5),
  incompatible with Python 3.13 / vizdoom 1.3.3 and with the privileged-free evaluation obs.

### 3. https://github.com/callumhay/vizdoom_ppo_rnd

- **License: MIT**, confirmed from `LICENSE`: `Copyright (c) 2022 Callum Hay`. GitHub API agrees.
- Useful ideas: `doom_ppo_rnd.py` (PPO+RND), `rnd_network.py`, `sd_encoder.py`,
  `doom_reward_vars.py`, README notes on reward normalisation (normalise intrinsic, not
  extrinsic; extrinsic advantage double weighting; `--reward-i-coeff` 0.01), network sizing
  (LSTM input:hidden ≈ 4:1). Most relevant: the README explicitly reports the LSTM **fails
  to link far-apart key→door dependencies** — the exact failure mode KEYSTONE targets.
- Dependencies: Python 3.10.8, torch 1.13 + CUDA 11.7, gym 0.21, vizdoom 1.1.13, numpy 1.23,
  tensorboard, optional wandb; 20 envs on a 12 GB 3080, ~6M steps / ~8 h per level.
- Compatibility: partial/needs rewrite. CNN uses RGB 80×60 + label buffer; embeddings use
  `POSITION_*`, `ANGLE`, health, ammo and **cropped HUD keycard pixels** (HUD hidden at eval);
  RND novelty uses position/angle one-hot. Actions include SPEED and a custom mapping.
- Reuse vs rewrite: PPO/RND structure and normalisation heuristics (ideas only); we rewrite
  networks, env wiring, key representation (`vitals[8:14]`), action adapter.
- Attribution if reused: retain MIT notice; upstream credits OpenAI (PPO/RND), CleanRL,
  `shakenes/vizdoomgym` — check each before carrying code.
- **Rejected for code reuse:** privileged inputs throughout, obsolete pins, 12 GB-scale
  assumptions incompatible with 2 GB VRAM / 1 CPU thread.

### 4. https://github.com/vwxyzjn/cleanrl

- **License: MIT per LICENSE file** (`Copyright (c) 2019 CleanRL developers`; `pyproject.toml`
  declares `license = "MIT"`), **but GitHub API reports `NOASSERTION`** because LICENSE appends
  third-party blocks. Those blocks include procgen starter kit (Apache-2.0 boilerplate),
  TD3/SAC/DQN/SpinUp/Stable-Baselines3 notices (MIT/BSD) and other Apache-2.0 text.
- Useful ideas: `cleanrl/ppo_atari_lstm.py` (the file callumhay based their LSTM PPO on),
  GAE/loss patterns, hyperparameter conventions, vectorised rollouts.
- Dependencies: Python `>=3.8,<3.11`, torch 2.4.1, gym 0.23.1, gymnasium 0.29.1, tensorboard,
  wandb, tyro, rich, moviepy, pygame, huggingface-hub. No ViZDoom dict support.
- Reuse vs rewrite: algorithmic patterns only; our PPO loop (`training/ppo.py`) is written
  from scratch for `Dict` obs, `MultiBinary(9)` and our event-memory state replay.
- Attribution if any file copied: retain the CleanRL MIT notice **and** all appended
  third-party notices; the metadata discrepancy may flag automated compliance checks.
- **Rejected for code reuse:** Python-version conflict, no DOOM interface, large multi-licence
  attribution surface, and it supplies generic algorithm plumbing, not the novelty.

### 5. Episodic curiosity — https://arxiv.org/abs/1810.02274 + https://github.com/google-research/episodic-curiosity

- **Paper licence:** arXiv metadata lists `nonexclusive-distrib/1.0/`, a *preprint distribution*
  licence, **not** a software licence. Cite the paper (`Savinov2019_EC`), do not treat it as
  code permission.
- **Repo licence: Apache 2.0**, confirmed from `LICENSE` (`Apache License / Version 2.0,
  January 2004`; full §4 redistribution terms) and `setup.py` (`license='Apache 2.0'`,
  `Copyright 2019 Google LLC`). No root NOTICE file was seen in the tree listing (not
  independently re-verified).
- **Bundled `third_party/` licences:** `baselines` MIT (OpenAI 2017, Google 2018), `gym` MIT,
  `keras_resnet` MIT, **`dmlab` GPL v2** (Google 2016/2019). Do **not** import
  `third_party/dmlab` — GPL-2.0 would contaminate a derivative work.
- Useful ideas: `episodic_curiosity/episodic_memory.py` (buffer), `r_network.py` (learned
  reachability from observation pairs with step-distance labels), `curiosity_env_wrapper.py`
  ("novel if unreachable from memory within k steps" → bonus + insert; memory wiped per
  episode), R-network data-generation curriculum. Paper: ICLR 2019, ViZDoom/DMLab/MuJoCo,
  beats ICM on navigation and addresses noisy-TV "couch potato" failure.
- Dependencies: Python 2.7, Linux only, TensorFlow-GPU ≥1.12, Keras, gym ≥0.10.9, absl-py,
  gin-config, dill, pytype; DMLab built from source; 32–80 GB RAM, ~5 GB GPU, days of training.
- Compatibility: very low (pixel DMLab actions, privileged grid oracle, no ViZDoom wrapper).
- Reuse vs rewrite: algorithm *design* (episodic buffer, reachability-gated insertion,
  episode reset) may inspire a documented ablation; all code rewritten in PyTorch.
- Attribution if reused: Apache 2.0 requires licence copy, marking modified files, retaining
  notices and reproducing any NOTICE contents.
- **Rejected for code reuse:** unusable legacy stack, huge resource envelope, GPL-2.0 inside
  the tree, privileged oracle.

### 6. Neural Map — https://arxiv.org/abs/1702.08360

- **Paper licence:** arXiv `nonexclusive-distrib/1.0/`; no software licence on the page.
  Authors Emilio Parisotto, Ruslan Salakhutdinov (ICLR 2018).
- **Code: NOT VERIFIED.** No official repository linked from arXiv or author pages. A
  secondary site mentions only a third-party, work-in-progress Keras implementation
  (unaudited, unlicensed) — we did not locate or use it.
- Useful ideas (paper only): structured 2D feature-map memory `C×H×W`, write restricted to the
  agent's location, global conv read + context read, `[read, context, write]` policy input,
  modified A3C, DOOM maze evaluation and reported generalisation to unseen environments.
- Compatibility: a Neural Map needs **metric position** for write addressing, which our
  contract explicitly withholds. An ego-centric analogue is a design task, not a port.
- Attribution: cite `Parisotto & Salakhutdinov, ICLR 2018`; no code-licence obligation.
- **Rejected for code reuse:** no verified official code; arXiv licence is not a code licence;
  privileged-position dependency conflicts with evaluation.

## Prior-art assessment of the novelty claim

- **Episodic/progression memory + novelty is pre-empted:** Savinov et al. (ICLR 2019) already
  use episodic memory, selective insertion and novelty bonuses in ViZDoom; `SPTM` adds graph
  episodic memory for navigation (not audited here).
- **Structured long-horizon memory in DOOM-like mazes is pre-empted:** Neural Map (ICLR 2018)
  and the callumhay README's own note about LSTM failure on distant key→door dependencies.
- **Confidence-aware goal conditioning is pre-empted in adjacent literature:** search snippets
  (titles/abstracts only, full texts **not** verified) include RISC (confidence-based goal
  switching), search-inspired subgoal selection, discriminability-based goal selection and
  "SEA" confidence-map goal selection. These defeat a broad "first confidence-aware goal
  conditioning" claim.
- **Defensible claim (only after measured ablations):** the *specific integration* of
  progression-event-tagged memory (keys, locked doors, interaction outcomes under the
  64-ray + vitals + entities interface), confidence-gated retrieval, and learned goal
  conditioning under the 2 GB VRAM / 1-CPU budget. Phrase as "a confidence-gated
  progression-memory architecture for ego-centric-only DOOM". **Do not** claim "first
  episodic memory for DOOM", "first confidence-aware goal selection", or invention of
  PPO/GRU/attention/curiosity.

## Attribution obligations actually incurred by this repository

- Official starter: URL citation as competition foundation (no visible licence).
- No third-party source files were copied; our code is original to this repository.
- Python libraries used at runtime (`torch`, `numpy`, `gymnasium`, `vizdoom`) are supplied by
  the official base image; no copies are vendored.
- Literature cited: Savinov et al. 2019, Parisotto & Salakhutdinov 2018, Beeching et al. 2020,
  Hay 2022 (idea-level references only).

## Not verified / caveats

- Exact commit SHAs for references 2–6 were not recorded (default branches and last-push dates
  only: edbeeching `master` 2022-12-08, callumhay `main` 2023-01-13, cleanrl `master`
  2026-04-20, episodic-curiosity `master` 2020-10-02).
- Neural Map official code could not be found; absence is not proof none exists.
- CleanRL `NOASSERTION` vs MIT text is an observed metadata discrepancy, not resolved here.
- Per-file licences inside `episodic-curiosity/third_party/` beyond the four listed were not
  individually audited.
- Prior-art items found by search snippets only; full texts not read.
