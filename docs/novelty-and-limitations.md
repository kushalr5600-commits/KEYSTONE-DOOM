# Novelty claim and limitations

## Defensible USP (scientifically cautious wording)

**KEYSTONE-DOOM** integrates a compact, learned, fixed-capacity progression-event memory with
confidence-gated retrieval and learned goal conditioning, operating **only** on the official
ego-centric observation interface (64 rays, vitals, masked entity set; optional depth) under
the competition's 1-CPU / 2 GB-VRAM inference budget. The hypothesis is that remembering
progression evidence (locked-door sightings, key acquisition, interaction outcomes) and
weighting it by a learned confidence signal helps a goal-conditioned policy backtrack and
complete key→door→exit maps better than a recurrent policy without such memory.

## What we do NOT claim

- We did **not** invent PPO, GRU/LSTM, attention, episodic memory, curiosity, neural maps,
  hierarchical control, or confidence-aware goal selection. All are established prior art.
- We do **not** claim scientific novelty for "memory in DOOM" or "confidence-aware goals".
  Savinov et al. (ICLR 2019) already combine episodic memory, selective insertion and
  novelty bonuses in ViZDoom; Neural Map (ICLR 2018) uses structured spatial memory in a
  DOOM maze; callumhay's PPO+RND repository documentation explicitly reports LSTM failure on
  distant key→door dependencies. Adjacent literature (titles/abstracts only, full texts not
  verified) also covers confidence-based goal switching and subgoal selection.
- We do **not** claim any leaderboard placement, score, or Tier 4 completion that has not
  been measured. Results live in `experiments/results.csv` and the final report.

See `docs/research-and-license-audit.md` for sources and licence verification.

## Why the integration is still worth evaluating

1. The official observation has **no absolute pose**, so classical spatial-map memory
   (e.g. Neural Map's location-indexed write) cannot be copied; our writes are keyed by
   content and event type derived from rays/vitals.
2. Long-horizon key→door→exit tasks are exactly where a plain recurrent policy is reported
   to fail; a progression-event memory is a targeted, testable remedy.
3. Confidence gating lets the policy discount stale/contradicted evidence rather than
   treating all remembered events as certain.
4. The whole agent stays within a tiny parameter budget and p50 ≈ 2.6 ms/decision locally.

Evidence standard: claim improvement **only** if matched-seed ablations (§8 of the master
prompt, §7 of the project specification) show it. Otherwise report neutral/negative and
simplify.

## Known limitations

- **Not a metric/topological map.** The memory is a slot-attention dictionary; "topological"
  refers to room-transition *evidence*, not a reconstructed graph or coordinates. No
  hand-written route planner exists (and would be illegal).
- **Confidence is learned, not calibrated.** It scales retrieval but is not a probability;
  stale-evidence handling is soft decay, not logical invalidation.
- **Vision unused by default.** RGB is excluded; depth downsampling exists but is off because
  profiling showed >2× step cost. This may hurt on maps where texture/colour cues matter.
- **Sparse exit reward on large levels.** Freedoom2 maps are used as training scaffolding;
  a 2100-tick cap may end episodes before an untrained agent explores them, so progression
  credit may be limited.
- **Short training budget.** The sprint is ~5 hours on CPU-only local hardware; the 30k-step
  run is a feasibility result, not convergence. More compute should improve all variants.
- **Hidden scoring.** Exact coefficients and tier weights are undisclosed; our shaping is a
  proxy aligned to the documented ordering, not an optimiser of the true score.
- **Local hardware gaps.** No NVIDIA GPU here: VRAM and Docker-sandbox checks are **not
  verified locally**; we report validator results from CPU mode and mark GPU checks as
  pending. The official `--docker` run is the definitive check on a GPU host.
- **Windows console/CPU pinning.** `validate.py` could not pin to one CPU or read `/proc`
  RAM on Windows; those validator lines are warnings, not passes. We report them honestly.

## Failure analysis to watch

- USE held across steps → door bouncing. Mitigated by edge legality, not guaranteed.
- Key acquired but never used → memory logs KEY but no UNLOCK; shaping rewards key once,
  exit dominates, so this should not become a farming strategy.
- Loops / spinning → observation-novelty saturates quickly; per-step cost discourages it.
- Stuck in corners → recurrence + memory give no hard guarantee; measured via stuck-rate
  metric in evaluation.
- Death avoidance → damage penalty is small (0.005/unit) vs exit (10.0), so it should not
  produce pathological passivity.
