# Open-source data reuse & sample-efficient training audit

Branch: `data-reuse` (from `score-max`). Frozen champion on `main` untouched.

## 1. Candidate audit table (verified, not assumed)

| Candidate | Source | Existence | Licence | Artifacts verified | Storage | Compatibility | Verdict |
|---|---|---|---|---|---:|---|---|
| GameWam ViZDoom APPO | `Yunncheng/gamewam-vizdoom` (HF) | yes | **Apache-2.0** (README + API) | 50k episodes, 9-D action vectors, 640×480 video; parquet files ~80 KB each | **1.88 TB full**; 4 episodes = 88 MB | Action order `[shoot,fwd,back,SL,SR,speed,TL,TR,turn_delta]` mappable to our 9 buttons (USE/none absent); obs = RGB + privileged x/y/angle/health, no rays/vitals/entities | **Convertible, sampled only** — representation experiment run; no direct policy gain |
| VAGO Doom MultiVec 31k | `VAGOsolutions/SauerkrautLM-Doom-MultiVec-31k` (HF) | yes | **Apache-2.0** | 31,645 frames; ASCII 40×25 + depth bins + 4 soft action scores; single 1.3 GB arrow file | 1.3 GB | ASCII/depth input, 4 combat actions; no rays/entities; dataset-server sample endpoint failed | **Reject for this agent** — incompatible observation and no compatible entity/rays; cannot inspect without full download |
| DoomFrameDataset | `brahmandam/DoomFrameDataset` (HF) | yes (GitHub repo 404) | **no licence tag** | preview 79k samples, RGB + 6-D action vector + reward/value; full 2.4M samples | 68–73 GB | 6-action head, no rays/vitals; licence absent | **Reject** — unlicensed and incompatible |
| p-doom/doom-dataset | HF `p-doom/doom-dataset` (GitHub 404) | yes | CC0-1.0 | 72.6 GB, 717 files; no inspected action labels | 72 GB | Unknown label format; huge | **Reject** — cost/value unjustified |
| Arnold | `glample/Arnold` (GitHub) | yes | **none found** | 5 pretrained `.pth` DQN weights (deathmatch, defend_the_center, health_gathering, 2017 tracks) | 175 MB repo | Pixel DQN, different action space | **Reject** — no licence = no reuse grant |
| DoomRL | `IgorKolodziej/DoomRL` (GitHub) | yes | none found | small course project, no weights | 3.8 MB | Roguelike RL, unrelated | **Reject** |
| SauerkrautLM-Doom-1.3M | HF model | 401 (gated/unavailable) | claimed Apache by dataset card | none verified | — | — | **Reject** — unavailable |
| edbeeching/3d_control_deep_rl | GitHub | yes | MIT (verified) | 16 checkpoints | 1.59 MB each | Pixel CNN+GRU | **Already reused** in champion |

## 2. Small-sample inspection actually performed
- Downloaded 4 GameWam `defend_the_center` episodes (4 parquet + 4 mp4, ~88 MB) after
  confirming single-file sizes (parquet ~20 KB, mp4 ~20–23 MB).
- Schema: `observation.state` 4 floats (POSITION_X/Y/ANGLE/HEALTH), `action` 9 floats, video
  640×480 H.264. Sampled every 4th frame → **955 frames**; action histogram
  `[41 stand, 0 fwd, 460 turn_left, 165 turn_right, 112 attack, 140 attack_left, 37 attack_right]`.
- Mapped canonical actions to our macro labels; observation mismatch (no rays/vitals/entities)
  means only a visual-representation transfer is plausible, not behaviour cloning into the
  full KEYSTONE encoder.

## 3. Training efficiency baseline (measured earlier, documented in GPU_REPORT)
- BC/transfer (CPU, 5,560 samples): 6,342–7,234 samples/s, ~0.8 s/epoch.
- GPU RTX 5060 sm_120: 330–3,873 samples/s (batch 256–8192), 1,772 MB peak VRAM — slower
  for this tiny model.
- PPO 20k steps on CPU ≈ 600 s; GPU 256 steps 33 s vs CPU 14.6 s.
- Conclusion: environment interaction, not neural optimisation, dominates cost. Reusing
  955 external frames saved <0.2 s of equivalent BC compute — negligible.

## 4. Experiment run (sample-efficient representation transfer)
1. Trained a 3-conv visual stem (same architecture as the MIT encoder) on 955 GameWam frames
   to predict mapped combat actions: 4 epochs, 3,796–10,170 samples/s, loss 2.05→1.50,
   accuracy 0.482 (majority-class plateau).
2. Swapped **only** `pretrained_conv.conv_head.*` and `linear1.*` into a copy of the champion
   (`transfer/weights/combat_conv_candidate.pt`), leaving all other weights unchanged.

| Candidate | Combat maps | Regression suite | Kills | Decision |
|---|---:|---:|---:|---|
| Frozen champion | 24/24 | 64/64 | 0 | reference |
| Combat-conv swap | 24/24 | **40/64** | 0 | **rejected** (no gain, regression) |

## 5. Compliance / attribution
- GameWam Apache-2.0 acknowledged; only 4 episodes downloaded and used for an offline
  representation experiment; videos/samples are not committed (too large).
- No external weights were copied into the deployed submission; champion SHA256
  `7e0a8d02…b644d7` unchanged.
- No hidden maps/evaluator state used; training-only privileged labels from the external
  dataset were never fed to inference.

## 6. Recommendation
Do not integrate any audited external dataset. The MIT 3D-Control encoder already in the
champion remains the only proven transfer. The dominant remaining bottlenecks (multi-door
navigation and zero kills) are **behavioural/action-selection** problems, not representation
problems, and 955–31k externally-labelled frames in incompatible scenarios do not address
them. Future reuse should target compatible rays/vitals/entity trajectories or a complete
teacher for our own maps, not more pixel-only data.
