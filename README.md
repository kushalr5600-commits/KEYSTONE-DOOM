# KEYSTONE-DOOM

**Key-Event Structured Topological Memory for Full-Gauntlet DOOM** — Yugaantar 2026 DevForge,
Track 1 ("Can It Play DOOM?"). Repository created from the official
[DevForge DOOM template](https://github.com/Reinforce-SST/devforge-doom-template).

A compact learned agent that keeps a fixed-capacity memory of progression-relevant events
(locked doors, keys, items, enemies, interaction outcomes), retrieves them with a
confidence-gated attention read, and conditions a learned goal head and action policy on the
result. It uses **only** the official ego-centric observation: no engine handle, no global
position, no hidden map data, no scripted route.

## Status

- [x] Public repo from official template; contract/rules/scoring audited with source citations
      (`docs/official-spec-summary.md`)
- [x] Research + licence audit (`docs/research-and-license-audit.md`)
- [x] Architecture, memory, PPO trainer, deterministic submission agent implemented
- [x] Unit tests: 9/9 passing; validator PASSED on baseline and trained agent (CPU mode)
- [ ] Final training run and held-out evaluation results in `experiments/results.csv`
- [ ] Docker/GPU validator run (not verifiable on this CPU-only Windows host)

Never interpret unchecked boxes as completed work; exact measured results are in
`experiments/results.csv` and the final report.

## Quickstart

Python ≥3.13 (official base image) with `uv`, or a local Python with the template deps:

```bash
uv sync                       # installs vizdoom, gymnasium and the editable env/ wrapper
uv run python validate.py     # official local validator
uv run python -m pytest -q    # unit tests (including KEYSTONE tests)
```

Local Windows setup used during development (Python 3.12 + pip):

```bash
pip install "vizdoom==1.3.3" "gymnasium>=1.0" numpy scipy opencv-python-headless torch pytest
$env:PYTHONPATH = (Resolve-Path env).Path    # PowerShell: expose the official env/ wrapper
python -m training.train --steps 30000 --out submission/weights
python -m training.evaluate --weights submission/weights/keystone.pt --tag final
python validate.py
```

Recommended definitive check on a GPU/Linux host: `python validate.py --docker`
(or build the image with `docker build -t devforge-doom .` first).

## Layout

```
keystone/              architecture shared by training and inference
  config.py            hyperparameters + ablation switches (serialized with checkpoints)
  features.py          observation -> fixed 1249-dim vector + 16-dim evidence + event tag
  memory.py            fixed-capacity differentiable event memory (confidence, age, attention)
  model.py             encoders, GRU, memory read/write, goal head, actor/critic
  actions.py           contract button-validity enforcement (shared)
training/              our training pipeline (judges read it)
  ppo.py               PPO + GAE + sequence replay through recurrence/memory
  train.py             curriculum: basics -> doors -> mixed -> gauntlet
  evaluate.py          deterministic held-out evaluation -> experiments/results.csv
configs/               ablation configs
submission/agent.py    official DoomAgent entry point (do not change its API)
submission/weights/    trained weights (Git LFS)
docs/                  contract summary, licence audit, architecture, experiments, demo
tests/                 official wrapper tests + tests/test_keystone.py
experiments/           JSONL training logs and results.csv
```

## Reproducing the submission

1. `python -m training.train --steps 30000 --out submission/weights` — writes
   `submission/weights/keystone.pt` and `config.json`.
2. `python -m training.evaluate --weights submission/weights/keystone.pt --episodes 8` — appends
   matched metrics to `experiments/results.csv`.
3. `python -m pytest -q` and `python validate.py` — both must pass before pushing.
4. Commit weights with Git LFS (`git lfs track` rules are already in `.gitattributes`), push,
   and verify `git lfs ls-files` lists `submission/weights/keystone.pt`.

## Compliance summary

- Decisions come from trained weights; no if/else action trees, A*/Dijkstra/BFS, `atan2`
  aimbots, unmodified SB3 policies or privileged inference input.
- `legalize_action` only enforces documented button semantics (USE/weapon edge triggers,
  mutually exclusive movement), identically in train and inference.
- Training-time reward shaping uses `DoomEnv.privileged_state()` as labels only; the deployed
  agent never sees it. Intrinsic reward ≤0.02/step vs exit 10.0.
- No network, no engine access, no repo writes at evaluation; repository mounted read-only.
- Model: ~314k parameters, checkpoint ~1.3 MB; local p50 ≈2.6 ms, p99 ≈4.0 ms per decision
  (budget 40 ms). VRAM/RAM limits: see `docs/official-spec-summary.md`.

## Documentation

| Document | Contents |
|---|---|
| `docs/official-spec-summary.md` | exact contract, tiers, scoring terms (weights undisclosed), limits, rule→test map |
| `docs/research-and-license-audit.md` | six references, verified licences, reuse decisions, prior art |
| `docs/architecture.md` | dataflow, memory schema, boundary compliance, resource profile |
| `docs/experiment-plan.md` | curriculum, reward, benchmark protocol, ablation matrix |
| `docs/novelty-and-limitations.md` | defensible USP, prior art, honest limitations |
| `docs/demo-script.md` | pitch, demo sequence, judge Q&A |
| `experiments/results.csv` | measured held-out results |

## Underlying template

The official starter's contract files (`docs/CONTRACT.md`, `EVALUATION.md`, `RULES.md`,
`SUBMISSION.md`), `env/`, `validate.py`, `maps/` and `tests/fixtures/` are preserved. See the
template [README](https://github.com/Reinforce-SST/devforge-doom-template) for base-image and
platform notes. The starter repository has no visible software licence; we cite it as the
competition foundation and do not claim a licence grant.
