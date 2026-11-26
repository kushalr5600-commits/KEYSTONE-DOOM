# Isolated open-source transfer experiment (branch `transfer-research`)

Primary working tree was NOT modified. All files here are isolated under `transfer/`.

## Source audit and licence inventory

| Source | URL | Licence (verified) | Weights actually found | Used? |
|---|---|---|---|---|
| 3D Control Deep RL | https://github.com/edbeeching/3d_control_deep_rl | **MIT**, `Copyright (c) 2019 Ed Beeching` (full text saved at `transfer/licenses/EDBEECHING_LICENSE.txt`) | **16 committed `.pth.tar`, ~1.59 MB each** | **Yes** — 2 checkpoints downloaded and inspected |
| Arnold | https://github.com/glample/Arnold | GPL (per repo) | No usable checkpoint matching contract | No |
| Stable Baselines agent | https://github.com/ViZDoomBot/stable-baselines-agent | **NONE FOUND** | LFS TF1/SB2 zips (no licence) | Rejected |
| vanDamme | https://github.com/thatzopoulos/vanDamme | checked in prior audit | no compatible weights | No |
| HF KraTUZen Vizdoom-Doom-Agent | https://huggingface.co/KraTUZen/Vizdoom-Doom-Agent | none stated | Sample Factory checkpoints, card inconsistent | Rejected |
| NaVDoom | https://github.com/agiantwhale/NavDoom | MIT | no weights; maze generator with goal game-vars (training-only idea) | Idea only |
| fabiononato/doomfly-rl | https://huggingface.co/fabiononato/doomfly-rl | MIT | safetensors, 27–48 M params, 22-action | Rejected (too large/incompatible) |

Structured-input donors were searched via GitHub API (`vizdoom navigation`, `vizdoom curiosity`).
No structured-observation agent with downloadable weights matching the DevForge contract was
found. The winner is therefore a **pixel-encoder transfer**, retained only because the quick
empirical test proved a large gain (below), as permitted by the addendum.

## Checkpoint inventory (downloaded and inspected)

Exact raw URLs via GitHub API `Accept: application/vnd.github.raw`:

- `saved_models/find_return_11_checkpoint_0198658048.pth.tar` — 1,590,097 bytes,
  SHA of content not separately recorded; state dict keys verified below.
- `saved_models/labyrinth_9_checkpoint_0198658048.pth.tar` — 1,590,171 bytes.

State dict (both): `conv_head.0.weight (16,3,8,8)`, `conv_head.2.weight (32,16,4,4)`,
`conv_head.4.weight (16,32,3,3)`, `linear1.weight (128,640)`, `gru.weight_ih/hh (384,128)`,
`critic_linear (1,128)`, `dist.linear (5,128)`. Source policy: 64×112 RGB → conv(16/32/16)
→ FC(128) → GRU(128) → 5-action categorical; training scenario find-return / labyrinth.

## Compatibility matrix

| Component | Donor | DevForge | Transfer decision |
|---|---|---|---|
| Observation | RGB 64×112 only | Dict: rgb 160×120, depth, 17 vitals, 64 rays, 32 entities | Resize RGB 160×120 → 64×112; keep our full structured encoder |
| Action | 5-button categorical | MultiBinary(9), our 17 legal macros | No donor action head reused |
| Recurrent core | GRU(128) | GRU(128) | Same size but different input; weights NOT reused (architecture reference only) |
| Visual encoder | 3-conv + FC(128) | none by default | **Weights reused** in isolated additive adapter |
| Training env | custom WADs | official wrapper | Donor env irrelevant; teacher only for demos |

## Transfer implementation

`transfer/cnn_adapter.py`:
- `PretrainedConvEncoder` loads the 8 conv tensors and `linear1` verbatim from the MIT
  checkpoint (shape-asserted), freezes them.
- `TransferKeyStoneNet(KeyStoneNet)` overrides `encode()` as `z = base_encode(parts) +
  visual_proj(conv(rgb))`; `visual_proj` is trained from zero. Every existing KEYSTONE layer
  keeps its shape, so the protected baseline2 checkpoint loads with zero missing keys.

`transfer/make_demos_rgb.py` regenerates oracle demonstrations with RGB frames (90 episodes,
6,384 transitions, 90/90 training exits). `transfer/train_transfer.py` behavior-clones from
baseline2 for 4 epochs with the conv stem frozen. `transfer/eval_transfer.py` evaluates on
the same held-out maps/seeds as the main experiment ledger.

**Bug found and fixed during the experiment:** the first RGB batch was pre-permuted, so the
adapter silently skipped the visual branch and both donor checkpoints produced identical
models. After fixing (RGB kept as `(B,H,W,3)`), `visual_proj` weights are non-zero and the
two donors produce distinct policies. Both results are reported below.

## Measured held-out performance (identical maps/seeds, deterministic argmax)

8 episodes × 6 verified holdout maps = 48 episodes:

| Model | Exits | Exit rate | Notes |
|---|---:|---:|---|
| `baseline2` (protected reference, same harness) | 16/48 | 0.333 | 0 deaths; stuck ≈0.85–0.99 |
| `transfer_bc` (BC only, visual branch accidentally skipped) | 32/48 | 0.667 | improvement came from BC, NOT transfer |
| **find_return_11 + visual adapter** | **40/48** | **0.833** | 8/8 on 5 of 6 maps; 0/8 on key map |
| **labyrinth_9 + visual adapter** | **40/48** | **0.833** | same pattern; different weights |

Per-map for `find_return_11`: move_0 8/8, move_1 8/8, nav_0 8/8, nav_1 8/8,
combat_0 8/8, key_1 0/8. Inference p50 ≈1.0 ms, p99 ≈2.1 ms (budget 40 ms).
A 16-episode confirmation run is in progress; results will be appended to
`transfer/transfer_results.csv` and summarised here when complete.

**Honest interpretation:** the 0.833 rate is measured only on our simple verified holdout
maps. It does NOT demonstrate Tier 4 competence, and key-door-exit progression still fails
(0/8 on the key map). The gain is consistent across two independent donor checkpoints, which
suggests the pretrained visual stem (not a single checkpoint's quirk) helps navigation.

## Integration instructions (for the lead agent, not applied here)

1. Copy `transfer/cnn_adapter.py` into the target tree.
2. Copy `transfer/weights/transfer_bc.pt` (or `transfer_lab.pt`) and the corresponding donor
   `.pth.tar` into the submission package; preserve `transfer/licenses/EDBEECHING_LICENSE.txt`
   and cite Ed Beeching 2019 (MIT).
3. Replace `submission/agent.py` with the isolated `transfer/agent_transfer.py` pattern:
   instantiate `TransferKeyStoneNet`, pass `observation["rgb"]` as `(B,H,W,3)`, use the same
   macro action legality. No network, no privileged input, no map access at inference.
4. Re-run `python validate.py` and the held-out evaluation. Only promote if the full held-out
   suite confirms the gain.

## Compliance

- Donor licence MIT verified and text preserved; no GPL/no-licence artefact used.
- Teacher uses privileged map data only to label training observations; student input is the
  official observation dict; teacher is absent from the deployed path.
- No hidden evaluation maps or evaluation seeds used in training (holdout seeds 1234+ only at
  evaluation; training seeds 9000+).
- Model ~1.3 MB checkpoint + 1.59 MB donor weights; p50 ≈1 ms on 1 thread — inside the
  2 GB / 40 ms budget.
