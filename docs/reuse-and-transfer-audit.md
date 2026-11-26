# Reuse and transfer audit — trained DOOM agents, checkpoints and demonstration data

Verification date: 2026-11-26. Scope: READ-ONLY research for KEYSTONE-DOOM (Yugaantar DevForge Track 1).
Methods: `gh api` (repo metadata, recursive git trees, raw file contents, release/tag lists),
Hugging Face Hub REST API (`/api/models`, `/api/datasets`, raw README files), `websearch`/`webfetch`
(arXiv metadata + full HTML, PyTorch docs), and an HTTP HEAD probe of a GitHub LFS media endpoint.
**No external code or weights were copied into this repository; no other file was edited.**

## 0. Baseline and verification conventions

Target contract (`docs/CONTRACT.md`, `docs/architecture.md`): `Dict` observation with `rgb`
`(120,160,3)` HUD-hidden, `depth` `(120,160)`, `vitals` `(17,)`, `rays` (64 distances + 64
hit types), `visible_entities` (32×5 features + types + mask); `MultiBinary(9)` buttons held
4 tics; Python 3.13, ViZDoom 1.3.3 + FreeDoom2; 2 GB VRAM / 1 CPU thread; repository read-only;
**no network at evaluation**. Our deployed graph is a numpy feature builder (1249 floats, RGB
unused by default) → GRU(128) → 16-slot event memory → confidence gate → 7-goal head → 9
independent Bernoulli logits + critic (~314k parameters, 1.3 MB checkpoint, p50 2.6 ms/decision).

Conventions used below:

- **"Weights found"** means an actual downloadable binary was located (committed blob, Hub file,
  or a git-LFS media URL confirmed served). A git-LFS *pointer* alone is explicitly labelled.
- **"none stated"** for a Hugging Face card means no `license` field/tag was present in the
  Hub metadata — it is **not** a claim that no licence exists, and it is **not** a grant.
- Existence of a checkpoint does **not** imply it can be loaded into our graph. Observation,
  action-space and framework mismatches are recorded per candidate.
- Exact GitHub commit SHAs were **not** recorded for any candidate (default branch tips only).
  Internal contents of `.zip` / `.pth` archives were **not** loaded/inspected; format statements
  come from the repositories' own docs and are flagged where unverified.

## 1. Candidate findings

### 1.1 `edbeeching/3d_control_deep_rl` — committed weights (best-evidenced artefact)

- **Source:** https://github.com/edbeeching/3d_control_deep_rl (paper arXiv:1904.01806,
  https://arxiv.org/abs/1904.01806). Default branch `master`, last push 2022-12 (prior audit).
- **Licence:** **MIT**, exact text in root `LICENSE` (`Copyright (c) 2019 Ed Beeching`);
  GitHub API `"license": {"spdx_id": "MIT", ...}`. Paper is arXiv preprint licence; cite
  `beeching2019budget` (the prior audit wrote `beeching2020baselines`, verify before use).
- **Code availability:** yes — `3dcdrl/models.py`, `multi_env.py`, `train_agent.py`,
  `create_rollout_videos.py`, scenario generators and 4 benchmark families.
- **Weights: FOUND — 16 committed `.pth.tar` files, ~1.59 MB each (~25.4 MB total).**
  Exact URLs (`https://raw.githubusercontent.com/edbeeching/3d_control_deep_rl/master/saved_models/<name>`),
  sizes in bytes:
  `2item_checkpoint_0198658048.pth.tar` (1,590,107), `4item…` (1,590,304), `6item…` (1,590,280),
  `8item…` (1,590,270), `find_return_7…` (1,590,191), `find_return_9…` (1,590,227),
  `find_return_11…` (1,590,097), `find_return_13_checkpoint_0190466048.pth.tar` (1,590,152),
  `labyrinth_7…` (1,590,232), `labyrinth_9…` (1,590,171), `labyrinth_11…` (1,590,187),
  `labyrinth_13…` (1,590,254), `two_col_p1…` (1,590,219), `two_col_p3…` (1,590,287),
  `two_col_p5…` (1,590,337), `two_col_p7…` (1,590,272).
  Note: README examples say `3dcdrl/saved_models/…`, but the verified tree path is root
  `saved_models/` — the README paths are stale. No releases or tags exist.
- **Architecture:** `CNNPolicy` = conv head `Conv2d(3,16,8,stride4) → Conv2d(16,32,4,stride2)
  → Conv2d(32,16,3,stride1)` (ReLU), flatten → `Linear(·,128)` → `GRUCell(128,128)` →
  separate `Linear(128,1)` critic and `Linear(128,num_actions)` categorical actor; orthogonal
  init. Input `(3,64,112)` RGB normalised ×1/255; 1-frame input, frame-skip 4. Algorithm:
  GPU-batched A2C with RMSprop, GAE optional, entropy 0.001.
- **Observation / action / env:** RGB only, resized 64×112. Custom ViZDoom labyrinth, find-and-return,
  k-item and two-color WADs. `--limit_actions` gives discrete `{F, L, R, F+L, F+R}` (5 actions);
  otherwise a discrete set derived from the scenario. Their `POSITION_*`/`ANGLE` game variables
  are available in training but are *not* in the policy input in `models.py` (the prior audit's
  "reads POSITION_*" statement applies to other files in the repo, not the policy forward pass).
- **Documented performance:** README/paper report mid-range-hardware training in **<24 h**,
  12,000 env fps, 5–10 M frames to convergence, 64 held-out test mazes per scenario; results are
  PNG figures (`results/*.png`), not machine-readable numbers — **exact scores not verified**.
- **Inference requirements:** very light (conv+GRU, <2 MB) — comfortably inside 2 GB VRAM / 1 CPU.
  Needs `torch` (original pin 0.4.1) and ViZDoom 1.1.4 for the *original* loader; weights are a
  plain `torch.save` dict and can be read by a modern torch with `weights_only` care.
- **DevForge compatibility:** low-to-partial. RGB input is 64×112 (ours 160×120) but both are
  HUD-hidden screen buffers — resizing is feasible. Action head is **categorical over ≤5 buttons**,
  incompatible with our 9 independent Bernoullis. No rays/vitals/entities. The GRU hidden size
  (128) coincidentally equals ours, but its input is conv features, not our 128-d `z`.
- **Adaptation difficulty:** moderate for the visual encoder, high for end-to-end policy use.
- **Compliance risk:** low (MIT — retain copyright + permission notice, cite paper).
- **Capability gain:** medium for encoder init; low for direct policy transfer. Most relevant
  concept is the **find-and-return / labyrinth memory benchmark**, directly analogous to our
  key→door→exit hypothesis.

### 1.2 `callumhay/vizdoom_ppo_rnd` — no committed weights

- **Source:** https://github.com/callumhay/vizdoom_ppo_rnd. `main`, last push 2023-01-13.
- **Licence:** **MIT**, raw `LICENSE` decoded: `MIT License / Copyright (c) 2022 Callum Hay`;
  GitHub API agrees. No releases, no tags.
- **Code availability:** yes — 34 files (`doom_ppo_rnd.py`, `rnd_network.py`, `sd_encoder.py`,
  `doom_reward_vars.py`, `doom_gym_wrappers.py`, `vizdoomgym/`).
- **Weights: NOT FOUND.** Full recursive tree has **no** `saved_models`, `.pth`, `.pt`, `.tar`,
  `.pkl` or checkpoint file. README says checkpoints are written at runtime to
  `runs/<gym_env_id>/<unique_run_dir>` and loaded via `--model <path>` — i.e. local training
  artefacts only, nothing downloadable from the repo.
- **Architecture:** CNN encoder over RGB (80×60) **plus label buffer**; embeddings of `POSITION_*`,
  `ANGLE`, health, ammo and **cropped HUD keycard pixels**; RND target/predictor (conv + FC) over
  RGB + keycard pixels + position/angle one-hot; LSTM fed `[CNN latent, position, keycard pixels,
  game-variable embeddings]`; actor + critic heads. PPO + RND, intrinsic coefficient 0.01.
- **Documented performance:** README reports beating Doom2 MAP01 and Doom E1M2 in ~6 M global
  steps (~8 h on a 12 GB RTX 3080 with 20 envs) and near-perfect play on basic ViZDoom configs
  in <1 M steps/level. Importantly it reports the **LSTM fails to link far-apart key→door
  dependencies** — the exact failure mode KEYSTONE targets.
- **Inference requirements:** PyTorch 1.13/CUDA 11.7, gym 0.21, ViZDoom 1.1.13, Python 3.10.8;
  20-env/12 GB training envelope. A trained checkpoint would be small at inference but is
  unavailable.
- **DevForge compatibility:** very low — privileged position/angle and HUD keycard pixels are
  unavailable at evaluation; custom action mapping includes SPEED; obsolete pins.
- **Adaptation difficulty:** n/a (no weights). **Compliance risk:** low (MIT, attribution to
  OpenAI PPO/RND and CleanRL if code were carried; none was).
- **Capability gain:** none from weights; high as a **negative result / design reference**
  (reward-normalisation heuristics and the documented LSTM long-horizon failure).

### 1.3 `ViZDoomBot/stable-baselines-agent` — LFS-hosted weights, no licence

- **Source:** https://github.com/ViZDoomBot/stable-baselines-agent. `main`, last push 2021-05-05.
- **Licence:** **NONE FOUND** — GitHub API `license` is null and root tree has no LICENSE/NOTICE.
  Do not claim any grant.
- **Code availability:** yes — Stable-Baselines (TF1) agents with custom networks/wrappers,
  including `my_way_home_lstm_curiosity` (RND), `simpler_my_way_home_lstm_icm*` (ICM variants),
  `simpler_deathmatch_lstm`.
- **Weights: FOUND via git LFS (availability verified).** `archive_models/*/saved_model.zip`
  and `…/curiosity_saved_model.zip` are 132–133-byte **LFS pointers** in the git tree; for
  `archive_models/deathmatch/saved_model.zip` the pointer declares SHA-256
  `40e3e8c5…bb98e`, **21,212,246 bytes**, and an HTTP HEAD to
  `https://media.githubusercontent.com/media/ViZDoomBot/stable-baselines-agent/main/archive_models/deathmatch/saved_model.zip`
  returned **HTTP 200, `Content-Type: application/zip`, `Content-Length: 21212246`**. The other
  ~13 pointers in `archive_models/` were **not individually probed** (unverified). Archive
  contents were not opened; SB3 parameter names/format are per repo docs, not independently loaded.
- **Architecture / obs / action:** Stable-Baselines (v2, TensorFlow ≤1.15) PPO/A2C with
  `nature_cnn` or larger CNN, optional LSTM and optionally RND/ICM curiosity; pixel observations
  at custom resolutions; ViZDoom discrete action subsets per scenario.
- **Documented performance:** team report/website only; no numeric evaluation in the repo tree
  inspected — **not verified**.
- **Inference requirements:** TF 1.x + stable-baselines + CUDA 10/cuDNN 7.6 (README warning),
  Python of that era. Directly unusable under Python 3.13.
- **DevForge compatibility:** very low (TF1 stack, discrete actions, pixel-only, no contract keys).
- **Adaptation difficulty:** high. **Compliance risk:** high (no licence).
- **Capability gain:** low; mainly a curiosity/LSTM reference.

### 1.4 `arnaudstiegler/gameNgen-repro` — committed SB3 PPO checkpoint + demo datasets

- **Source:** https://github.com/arnaudstiegler/gameNgen-repro. `main`, last push 2025-05-19.
- **Licence:** **Apache-2.0** (GitHub API `spdx_id: Apache-2.0`). Hugging Face derivative
  artefacts have their own licence fields (see below).
- **Code availability:** yes — `ViZDoomPPO/train_ppo_parallel.py`, `load_model_generate_dataset.py`.
- **Weights: FOUND (committed, not LFS).** `ViZDoomPPO/logs/models/deathmatch_simple/best_model.zip`,
  **18,468,026 bytes**, URL
  https://raw.githubusercontent.com/arnaudstiegler/gameNgen-repro/main/ViZDoomPPO/logs/models/deathmatch_simple/best_model.zip.
  SB3 `.zip` format per repo docs; not loaded here. HF also hosts diffusion world-model
  checkpoints `arnaudstiegler/gameNgen-baseline-20ksteps` (CreativeML Open RAIL-M, ~104 GB total,
  859M params) and `sd-model-gameNgen-60ksteps` — **world models, not policies, and
  RAIL-M-use-restricted; not recommended**.
- **Architecture / obs / action:** Stable-Baselines3 PPO (`ViZDoomPPO/`), deathmatch-simple,
  CNN over pixels, SB3 discrete actions. Generation pipeline extracts frames, actions, health,
  step ids at frame-skip 4.
- **Documented performance:** used only to generate training episodes for world-model work;
  no policy score verified.
- **Inference requirements:** SB3 (PyTorch), ViZDoom; a modern SB3 can load the zip, but the
  observation/action wrappers are scenario-specific.
- **DevForge compatibility:** low (pixel-only, discrete, no dict keys).
- **Adaptation difficulty:** high for policy; moderate for using the dataset to pretrain an encoder.
- **Compliance risk:** low for the Apache repo; **dataset licences not stated** (see 1.9).
- **Capability gain:** low-medium (demo data more valuable than the checkpoint).

### 1.5 Hugging Face Sample Factory APPO models (`edbeeching/*`, `rl_course_vizdoom_*`, many forks)

- **Source examples:**
  https://huggingface.co/edbeeching/doom_health_gathering_supreme_2222 ,
  https://huggingface.co/edbeeching/doom_deathmatch_bots_2222 ,
  https://huggingface.co/rohit0128/rl_course_vizdoom_health_gathering_supreme ,
  https://huggingface.co/g-kurtish/rl_course_vizdoom_doom_defend_the_center ,
  https://huggingface.co/execbat/rl_course_vizdoom_doom_defend_the_line ,
  https://huggingface.co/KraTUZen/Vizdoom-Doom-Agent .
  Training code: Sample Factory (https://github.com/alex-petrenko/sample-factory), Apache-2.0
  (not re-verified here).
- **Licence:** **NONE STATED on the model cards** — `cardData` has no `license` key and no
  `license:*` tag for the `edbeeching/*` and `rl_course_*` repos checked. Treat as no verified
  grant. `nouli-ai` (1.7) and `fabiononato` (1.6) are the exceptions with explicit licences.
- **Code availability:** training framework public; per-model `cfg.json`, `sf_log.txt`, `README.md`.
- **Weights: FOUND (Hub files verified in `siblings`).**
  - `edbeeching/doom_health_gathering_supreme_2222`: `checkpoint_p0/best_000466273_3819708416_reward_63.056.pth`,
    `checkpoint_000538763_4413546496.pth`, `checkpoint_000539850_4422451200.pth`; repo storage
    137 MB; self-reported mean reward **67.00 ± 0.00** (unverified).
  - `edbeeching/doom_deathmatch_bots_2222`: best reward 93.067 plus 2 checkpoints; 141 MB;
    reported mean reward `nan` (unverified).
  - `execbat/rl_course_vizdoom_doom_defend_the_line`: three `.pth` LFS files of **34,928,614 /
    34,929,028 / 34,929,028 bytes** (pointer SHAs verified in a commit view); direct download via
    `https://huggingface.co/execbat/rl_course_vizdoom_doom_defend_the_line/resolve/main/checkpoint_p0/<file>`.
  - `g-kurtish/rl_course_vizdoom_doom_defend_the_center`: 117 MB including `checkpoint_p0/`.
  - `KraTUZen/Vizdoom-Doom-Agent`: card claims `model.pkl`, but actual verified files are Sample
    Factory `checkpoint_p0/checkpoint_000000003_12288.pth` and `checkpoint_000000005_20480.pth`
    (**card/file-list inconsistency — flag**); self-reported mean reward only **5.80 ± 0.46**.
  - `rohit0128/rl_course_vizdoom_health_gathering_supreme`: mean reward 18.50 ± 1.50.
  - Many near-identical student uploads exist (`ThomasSimonini/`, `frangiral/`, `aj555/`, …),
    all tagged `sample-factory`.
- **Architecture / obs / action:** Sample Factory APPO (asynchronous PPO), CNN feature extractor
  over stacked RGB frames plus game-variable vector (`HEALTH` etc.), discrete per-scenario action
  sets (e.g. health-gathering has 3 buttons: turn left/right, forward). No dict obs, no rays.
- **Documented performance:** model-card self-reports above, all `"verified": false`.
- **Inference requirements:** Sample Factory env registration/wrappers; Python of the SF2 era;
  GPU-trained but CPU inference feasible for the small actor. Weights are not standalone
  `state_dict`s in our format.
- **DevForge compatibility:** very low (scenario-specific pixel policies, discrete actions,
  no contract fields).
- **Adaptation difficulty:** high; **compliance risk:** medium (no licence stated).
- **Capability gain:** low as policies; they are useful only as **offline teachers** for combat/
  survival priors in matching standard scenarios (1.5 offers no licensed shortcut).

### 1.6 `fabiononato/doomfly-rl` — licensed, downloadable, distilled Doom policies

- **Source:** https://huggingface.co/fabiononato/doomfly-rl (code: https://github.com/nonatofabio/doomfly-rl).
- **Licence:** **MIT** — Hub `cardData.license: "mit"` and tag `license:mit`; GitHub repo MIT.
  Connectome data derived from `fernandofernandes/fly-connectome-49k`, **CC BY 4.0** (attribution
  obligation if the connectome file is redistributed).
- **Code availability:** yes (`doomfly.surgery.load_flynet`, `doomfly.evaluate`, launch scripts).
- **Weights: FOUND (verified file list, `model.safetensors`):**
  `malecns49k_v2_final/model.safetensors` (48.2 M params),
  `malecns49k_shuffled_s0/model.safetensors` (48.2 M),
  `malecns49k_noconn/model.safetensors` (27.7 M), plus `connectome_malecns49k.npz`;
  download via `https://huggingface.co/fabiononato/doomfly-rl/resolve/main/<path>`.
  Total repo storage 1.14 GB; exact per-file byte sizes **not verified** (Hub `siblings` omits sizes).
- **Architecture / obs / action:** `FlyNet`: 4-frame stack of **72×96 grayscale** Doom frames →
  conv stem → sparse recurrent "connectome" layer (49 k neurons / 9 M edges, unrolled 5 steps,
  frozen topology and signs, one learned gain per synapse; `noconn` control omits it) → 512-wide
  readout → **22 discrete actions** with per-scenario legality mask + 64-bin value head.
  Trained by distilling five per-scenario SB3 PPO teachers for 60 k steps on one L40S.
- **Documented performance (10 episodes, mean ± std return):**
  `v2_final` basic 78.2±7.5, defend_the_center 18.6±1.8, health_gathering 1728±616,
  deadly_corridor 2280.5±2.4, defend_the_line 21.0±5.9; controls within 1 std — the paper/story
  explicitly says the wiring adds **no measurable benefit**.
- **Inference requirements:** PyTorch + safetensors + connectome `.npz`; CPU-capable but 27–48 M
  params, 4×72×96 input; no network at eval if files are vendored.
- **DevForge compatibility:** very low (grayscale stack, 22-action categorical, standard scenarios).
- **Adaptation difficulty:** high. **Compliance risk:** low (MIT + CC BY attribution if connectome used).
- **Capability gain:** low as a policy; medium as a **distillation teacher** whose conv stem is
  pretrained on Doom pixels (better domain match than ImageNet).

### 1.7 `nouli-ai/nouli-doom` — small structured-state PPO (distillation student)

- **Source:** https://huggingface.co/nouli-ai/nouli-doom.
- **Licence:** **Apache-2.0**, with actual `LICENSE`, `NOTICE`, `licenses/Doomfly-MIT.txt` and
  `provenance.json` in the repo (the best-documented licence chain found).
- **Code availability:** yes (`inference.py`, `native_backend.py`, `state.py`, `serve.py`, `prepare.py`).
- **Weights: FOUND:** `policy.pt` (repo total 2.1 MB), URL
  https://huggingface.co/nouli-ai/nouli-doom/resolve/v1.0.0/policy.pt.
- **Architecture / obs / action:** PPO MLP, **23,435 parameters**, input **42-float structured
  state**, output **54 legal joint actions**; Deadly Corridor at skill 5. Student trained by
  imitation (DAgger) from `fabiononato/doomfly-rl` `malecns49k_noconn` plus simulator RL.
- **Documented performance:** illustrative recorded comparison with Jev 1.14.0 (armor collected,
  survived) — explicitly *not* a general ranking; no benchmark table.
- **Inference requirements:** Python 3.12, custom Nouli CLI/workflow server; CPU.
- **DevForge compatibility:** very low — completely different structured observation and joint
  action space, despite both being "Doom".
- **Adaptation difficulty:** high. **Compliance risk:** low (Apache + notices).
- **Capability gain:** low; useful only as a worked example of **policy distillation into a small
  contract-shaped student**, which is the one transfer pattern that fits our budget.

### 1.8 `kuds/rl-doom`, `JihangLi1121/Doom-rl`, CleanRL, SB3 Zoo, episodic curiosity — negative results

- **`kuds/rl-doom`** (https://github.com/kuds/rl-doom): **MIT**; code yes; README lists a
  `checkpoints/` artefact directory, but the recursive tree contains **no weights**
  (`src/rl_doom/models.py` and a test only) → **weights NOT FOUND**. PPO/DDQN + recurrent PPO
  (`sb3_contrib.RecurrentPPO`, `CnnLstmPolicy`, 256-d LSTM), standard ViZDoom scenarios.
- **`JihangLi1121/Doom-rl`** (https://github.com/JihangLi1121/Doom-rl): **licence NONE FOUND**;
  `models/` is gitignored → **weights NOT FOUND**. SB3 PPO with reward shaping.
- **CleanRL** (https://github.com/vwxyzjn/cleanrl): MIT text / GitHub `NOASSERTION` (see prior
  audit); GitHub code search `repo:vwxyzjn/cleanrl vizdoom` returned **0 hits** → **no Doom
  checkpoints**. Its `ppo_atari_lstm.py` was an algorithm ancestor of 1.2; no artefacts.
- **Stable-Baselines3 Zoo** (https://github.com/DLR-RM/rl-baselines3-zoo): code search for
  `doom` returned **0 hits**; `hyperparams/` contains no Doom entry → **no official SB3 Zoo Doom
  models**. The SB3 "Doom" policies found are community uploads (1.5) or third-party repos (1.4).
- **`google-research/episodic-curiosity`** (https://github.com/google-research/episodic-curiosity):
  Apache 2.0 (prior audit); re-checked releases/tags and tree — no checkpoints, only
  `episodic_curiosity/keras_checkpoint.py` helper → **weights NOT FOUND**.

### 1.9 Demonstration datasets and imitation-learning artefacts

- **`p-doom/doom-dataset`** (https://huggingface.co/datasets/p-doom/doom-dataset):
  **CC0-1.0** (`cardData.license: cc0-1.0`) — cleanest licence found. **10 M frames at 60×80
  with actions**, ArrayRecord train/val/test, ~72.6 GB storage (700+ shards). Intended for
  world-model training (Jasmine). Useful for *offline* encoder pretraining or imitation
  warm-start on a subset; action labels are ViZDoom defaults and must be remapped to
  `MultiBinary(9)`. Compute/compute-storage cost is far beyond a sprint.
- **`arnaudstiegler/vizdoom-5-episodes-skipframe-4-lvl5`** — 187 MB, 10,212 examples; features
  `episode_id, frames(binary), actions(int64), health, step_ids`; and
  **`…/vizdoom-500-episodes-skipframe-4-lvl5`** — ~48 GB, 2,470,229 examples. URLs:
  https://huggingface.co/datasets/arnaudstiegler/vizdoom-5-episodes-skipframe-4-lvl5 and
  https://huggingface.co/datasets/arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5 .
  **Licence: NONE STATED** on either dataset card (README contains only `dataset_info`) —
  the parent GitHub repo is Apache-2.0, but that does not automatically cover the datasets.
  Actions are the generating SB3 PPO agent's discrete actions; no key/door progression labels.
- **`Behavioural Cloning in VizDoom`** (Spick et al., https://arxiv.org/abs/2401.03993):
  paper HTML states **CC BY-NC-ND 4.0** (non-commercial + no derivatives). Data: 41 human
  deathmatches, 95 trajectories, ~541k frame–action pairs at 640×480, RGB+depth+label
  (5-channel, 256×192), continuous mouse deltas + binary keys. Architecture: CNN → ConvLSTM
  (4 layers) → per-action MLP heads, 15-frame exponential skip. Best agent ~11 kills / 1462
  damage in 3 min, on par with average human. **No public code, dataset or checkpoint link was
  found** on the arXiv abstract or full HTML → **weights/data NOT FOUND**. The licence alone
  blocks derivative reuse.
- **`Deep RL and IL based on VizDoom`** (ACM 10.1145/3573428.3573729): abstract/HTML reviewed;
  human-in-the-loop dueling-DQN; **no verified code/weights link** → not actionable.
- **Generic visual encoder:** `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
  — 11.7 M params, **44.7 MB** checkpoint, acc@1 69.758, `min_size` 1×1 (so **160×120 is
  accepted** after preprocessing), source https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html .
  Torchvision code is BSD-3-Clause; PyTorch docs warn pretrained weights "may have their own
  licenses or terms" — verify before bundling. It is a domain-mismatched, relatively heavy
  encoder for a 1-CPU/40 ms-per-decision budget; a small Doom-pretrained conv stem (1.1/1.6)
  is a better size/performance trade.

## 2. Ranked table

Scores are qualitative: **Gain** = expected capability gain if used as-is or after light
adaptation; **Compat** = compatibility with the DevForge contract; **Effort** = engineering
effort (Low is better); **Compute** = compute cost of exploitation (Low is better);
**Legal** = clarity/usability of the licence (High is better). Sorted by practical value for
this project, not by raw model strength.

| Rank | Candidate | Weights? | Gain | Compat | Effort | Compute | Legal | Practical verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `edbeeching/3d_control_deep_rl` (MIT, 16×1.59 MB) | **Yes, verified** | Med | Partial | Med | Low | High (MIT) | Only clean, small, downloadable Doom conv+GRU checkpoints. Reuse **encoder weights** (resize 160×120→64×112) and GRU *design*; retrain action head. |
| 2 | `fabiononato/doomfly-rl` (MIT, safetensors) | **Yes, verified** | Med (teacher) | Low | High | Med | High (MIT + CC BY connectome) | Best-licensed **distillation teacher** on Doom pixels; 27–48 M params and 22-action head make direct deployment impractical. |
| 3 | `p-doom/doom-dataset` (CC0, 10 M frames) | Data, verified | Med (offline) | Low | High | Very high (72 GB) | Very high (CC0) | Clean licence for offline encoder/world-model pretraining; too large and wrong action format for a sprint. |
| 4 | `arnaudstiegler/vizdoom-5-episodes…` (187 MB) | Data, verified | Low–Med | Low | High | Med | **Unknown / none stated** | Small enough to sample; licence must be resolved before use. |
| 5 | `nouli-ai/nouli-doom` (Apache-2.0, `policy.pt`) | **Yes, verified** | Low | Very low | High | Low | Very high | Best-documented licence chain; demonstrates small student distillation, but obs/action incompatible. |
| 6 | `arnaudstiegler/gameNgen-repro` SB3 PPO zip (Apache-2.0) | **Yes, verified** | Low | Low | High | Low | High | Usable as SB3-format teacher only; the accompanying world models are RAIL-M-restricted and huge. |
| 7 | HF Sample Factory APPO (`edbeeching/*`, `rl_course_*`) | **Yes, verified** | Low | Very low | High | Med | **Unverified (none stated)** | Technically downloadable, legally clouded and scenario-locked; do not rely on. |
| 8 | `ViZDoomBot/stable-baselines-agent` (no licence) | Yes (LFS, 1 URL HTTP-200 verified) | Low | Very low | Very high | Low | **Very low** | TF1/SB2 stack plus no licence — reject. |
| 9 | `KraTUZen/Vizdoom-Doom-Agent`, `kuds/rl-doom`, `JihangLi1121/Doom-rl` | Partial / none | Low | Very low | High | Low | Unverified / none / MIT | Card inconsistencies or missing weights; no actionable artefact. |
| 10 | `callumhay/vizdoom_ppo_rnd` (MIT) | **No** | 0 (weights) | Very low | n/a | n/a | High | Design/negative-result value only; the documented LSTM key→door failure supports our novelty claim. |
| 11 | Episodic curiosity / CleanRL / SB3 Zoo / BC paper | **No** | 0 | Very low | n/a | n/a | Mixed (Apache/MIT/NC-ND) | No public Doom checkpoint; the BC paper's CC BY-NC-ND blocks derivatives. |
| 12 | torchvision ResNet18 ImageNet (BSD-3 code) | **Yes** (44.7 MB) | Low–Med (generic) | Low | Med | Med | Medium (weights terms) | Works at 160×120 but domain-mismatched and slow on 1 CPU; prefer Doom-pretrained small encoder. |

## 3. Verdict: can any external checkpoint be transferred into our architecture in a short sprint?

**No — not end-to-end.** Every verified Doom checkpoint was trained against a different
observation representation (stacked/grayscale pixels, privileged position/angle, HUD keycards,
or 42-float structured state) and a different action head (categorical over 3–22 discrete
actions, or 54 joint actions) from our `Dict` obs + `MultiBinary(9)` + GRU/event-memory graph.
Loading weights into `keystone/model.py` would require rewriting the feature builder, the
recurrent input dimension, the actor and the critic — i.e. training from scratch with extra
bookkeeping. The legal picture reinforces this: only three artefact families have a verified
licence (MIT edbeeching, MIT doomfly, Apache nouli, CC0 p-doom), and two of those are
scenario-specific teachers. No release/tag of a "drop-in DevForge agent" exists.

What **is** realistically transferable within a sprint:

1. **Encoder — yes, partially.** `edbeeching`'s 3-conv stem (16/32/16 filters, 8/4/2/3 kernels,
   ~1.59 MB) is the best candidate: pretrained on Doom RGB at 64×112 and easily adapted to our
   HUD-hidden 160×120 `rgb` by resize or by adjusting the first stride. Keep it frozen initially,
   feed its pooled output into our existing `z` path, and compare against our current
   structured-only baseline. Doomfly's conv stem is grayscale/4-stack and needs a channel
   adaptation; ImageNet ResNet18 is technically valid at 160×120 but unlikely to beat a tiny
   Doom-trained stem under 1 CPU / 40 ms.
2. **Recurrent core — architecture only, not weights.** Our GRU(128) matches `edbeeching`'s
   hidden size, but the input differs (their conv features vs our 128-d `z`), so their GRU
   tensors cannot be loaded. Their orthogonal init and A2C/GRU design, plus callumhay's LSTM
   sizing heuristic, can inform our training recipe; nothing is copied.
3. **Action head — no.** All external heads are categorical single-choice or multi-discrete;
   ours emits 9 independent Bernoullis with edge-trigger legalisation. Must be retrained.
   The only transferable item is a *soft prior* (e.g. distill "turn toward enemy" from a
   teacher into a training-only auxiliary loss), never a copied head.
4. **Distillation teacher — yes, the most promising pattern.** Use `edbeeching`'s
   find-return/labyrinth checkpoints or `fabiononato/doomfly-rl` (`malecns49k_noconn`, MIT) as
   an **offline teacher during training only** to provide action/value targets or representation
   alignment on matched scenarios; `nouli-ai/nouli-doom` is a worked example of exactly this
   student/teacher relationship. No teacher, weights or network calls are needed at evaluation,
   so the no-network/read-only constraints remain satisfied. Caveats: teacher observations must
   be reconstructed from our contract obs (resize RGB; we lack HUD keycards and absolute pose),
   and discrete teacher actions must be mapped to our 9 buttons. Budget this as an experiment,
   not a guaranteed shortcut.
5. **Demonstration data — supplementary.** `p-doom/doom-dataset` (CC0) or the 187 MB gameNgen
   sample could pretrain a visual encoder offline; licence must first be resolved for the latter.
   Neither provides key/door progression labels, so neither directly teaches KEYSTONE's target
   behaviour.

**Recommended sprint decision:** keep the current from-scratch architecture; optionally run one
time-boxed ablation that initialises only the visual encoder from
`edbeeching/.../labyrinth_9_checkpoint_0198658048.pth.tar` (or one `find_return_*` checkpoint)
and measure held-out progression performance against the current baseline. Do **not** vendor any
"none stated"/"none found" artefact, do not touch the RAIL-M world models or the CC BY-NC-ND
imitation paper, and attribute MIT/Apache/CC0 sources if any of their artefacts are used.

## 4. Unverified / caveats register

- Checkpoint archive *contents* (SB3 `.zip`, Sample Factory `.pth`, `pth.tar` key names) were not
  opened; compatibility claims are inferred from source code and README docs.
- Hub `siblings` do not expose per-file byte sizes for `fabiononato/doomfly-rl`, `nouli-ai/nouli-doom`
  or the Sample Factory models; only repo-level storage totals were verified.
- Only one of ViZDoomBot's LFS pointers was confirmed served (HTTP 200); the rest are pointers only.
- Documented scores from READMEs/model cards are self-reported (`"verified": false`) or presented
  as figures; no independent reproduction was performed.
- No exact commit SHAs were recorded for GitHub repositories (branch tips only); Hub SHAs are
  available from the API if needed.
- No licence was inferred from a repository's purpose or author. Where none was found, it is
  reported as "none found"/"none stated", never as permissive.
- `callumhay`'s README path `runs/<gym_env_id>/<unique_run_dir>` describes *local training output*,
  not a published artefact; absence of committed weights was verified against the full tree.
