# Final evidence-backed report — `progression-transfer` branch

## Champion (frozen)

- Checkpoint: `transfer/weights/champion_frozen.pt`, SHA256
  `7e0a8d022a3e8dce04090100211a7a2151792b15e30c69b10aa516aaa0b644d7` (1,326,425 bytes).
- Config: `configs/baseline_macro.json` (17 legal macro actions; reactive backbone; MIT
  pretrained conv stem adapter; progression-balanced BC).
- Integrated into this branch's `submission/agent.py` only; primary tree untouched.
- **Official validator PASSED** (3 Windows-only warnings): repo 11.0 MB, init 1.7 s, first
  act 7 ms, p50 2.7 ms, p99 6.6 ms (budget 40 ms).

## Controlled comparisons (held-out maps, deterministic, identical seeds/harness)

| Candidate | Exit rate | Progression maps | Notes |
|---|---:|---:|---|
| baseline2 (protected reference) | 32/96 = 0.333 | 0/16 | original corrected baseline |
| transfer_find (visual only) | 80/96 = 0.833 | 0/16 | nav/combat gain; no key-door |
| no_visual_bal (balanced USE only) | 40/64 = 0.625 | 24/24 | solves keys but loses move/nav |
| **champion (visual + balanced USE)** | **128/128 = 1.000** | 48/48 | no regression; 0 deaths/damage |

The 16-episode champion run covered 8 held-out maps: 2 move, 2 nav, 1 combat, 3 key
(including two independently generated unseen key variants). All 128 episodes exited with
zero deaths, zero damage; mean steps 12–25 depending on map; stuck rate 0.17 on key maps and
0.77–0.88 on move/nav maps (the policy repeats a macro while translating, which is not a
hardware stall). One warm-up p99 spike of 71 ms occurred in the first episode; steady-state
p99 ~1.5–4 ms.

## Progression phase metrics (24 episodes, `transfer/progression_metrics.csv`)

- Key acquisition: 24/24; locked-door encounter: 24/24; exit: 24/24.
- USE presses: 8.0/episode; observation-derived INTERACT_OK events 1.33/episode,
  INTERACT_FAIL 4.67/episode, crude "USE without OK" 6.67/episode.
- **Interpretation:** the 46%-USE balanced dataset did produce frequent USE, but it is
  context-effective (every episode unlocks and exits in 24 steps, zero damage/deaths). The
  repeated presses are partly an artefact of the event tagger (an OK is only tagged on the
  transition where the locked-door ray disappears). It is not a catastrophic exploit, but a
  lower USE ratio or DAgger on failure states could refine it.

## Full-gauntlet-style test

An oracle-verified held-out map combining key, locked door and a monster
(`maps/holdout/eval_gauntlet_2.wad`, oracle exit in 201 steps) was generated. Champion:
**8/8 exits**, zero deaths/damage, 22 steps/episode, p50 1.1 ms. Visual-only transfer:
**0/8** (stuck at the locked door with key held). This is still a simplified proxy, **not**
the official hidden Tier 4 suite.

## Ablation evidence for the research question

- Structured observation learning + pretrained visual encoder (MIT edbeeching weights):
  improved nav/combat from 0.333 to 0.833 but did not solve progression (0/16 key maps).
- Progression-aware demonstration balancing (46% USE): solved key-door-exit (24/24) but
  degraded move/nav when used without visual transfer (0.625 overall).
- Combination achieved 1.000 on the held-out suite, showing the two contributions are
  complementary in this setting. **This is a competition-specific integration result, not a
  claim that PPO/GRU/memory/curiosity were invented.**
- Memory components of the original KEYSTONE architecture are present in the codebase but
  the frozen champion uses the reactive backbone; no claim is made that memory improved this
  particular benchmark.

## GPU acceleration (see `transfer/GPU_REPORT.md`)

- RTX 5060 Laptop 8 GB, sm_120; cu126 lacked kernels; CUDA 13 wheel (`torch 2.14.1+cu130`)
  works in an isolated venv. Centralized `--device` added to BC, PPO and transfer training;
  CPU fallback verified.
- CUDA benchmarks: transfer BC best 1,389–3,873 samples/s (batch 8192, 1,772 MB VRAM) vs CPU
  6,342–7,234; plain BC GPU 4,146 vs CPU 35,658; PPO 256 steps GPU 33.1 s (73.5 MB VRAM)
  vs CPU 14.6 s. Tiny networks favour CPU; GPU support retained for larger models.
- GPU-trained checkpoint loaded and ran correctly with the system CPU-only torch; deployment
  remains CPU-compatible and inside the 2 GB / 40 ms budget.

## Compliance and attribution

- Official contract preserved; no evaluator change, hidden maps, privileged inference input,
  network access or scripted route in the deployed agent.
- Training oracle used privileged map data only to label permitted observations; not deployed.
- MIT licence of `edbeeching/3d_control_deep_rl` verified and preserved
  (`transfer/licenses/EDBEECHING_LICENSE.txt`, Copyright 2019 Ed Beeching).
- Held-out maps and seeds were never used for training.

## Unverified

Official hidden Tier 4 maps and score; Linux/Docker validation; VRAM measurement at official
evaluation; broad generalization beyond our generated map families.
