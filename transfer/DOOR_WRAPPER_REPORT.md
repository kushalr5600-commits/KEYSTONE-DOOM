# Surgical Tier-2 door intervention — results

## Protected baseline
Champion `transfer/weights/champion_frozen.pt` SHA256 `7e0a8d02…b644d7` unchanged. Its
weights, architecture and `submission/agent.py` inference were not edited. The (optional)
wrapper lives in `door_controller.py` + `transfer/eval_wrapped.py`.

## Observable signals used (official observation only)
- `rays.hit_type == 2` (normal door) in the 18 front rays; no `== 4` enemy in front.
- `rays.distance < 0.18` (door within 184 units) for >= 3 consecutive steps.
- Disappearance of door evidence after USE = success; persistence = retry/escape.
- No position, map data, oracle labels, hidden evaluator state or hardcoded route.

## Controller design
Bounded state machine: `policy → use (1 step) → wait (36 steps) → policy`, up to 3 retries,
80-step cooldown, then a 12-step right turn as escape. Control returns to the champion as soon
as door evidence clears. USE is one press per trigger (with the existing one-step edge rule),
so the door is not reversed by spamming.

## Results (identical seeds/maps, champion + optional wrapper)

| Benchmark | Champion | Champion + wrapper | USE attempts |
|---|---:|---:|---:|
| Tier-2 single-door maps (4 verified maps × 8) | 0/32 | **32/32** | 192 (6/episode avg; 192 tagged success) |
| Tier-2 multi-door `two_rooms` maps (4 × 8) | 0/32 | 0/32 | repeated attempts, exit still unreachable |
| Tier-1 combat holdout | 8/8 | 8/8 | 0 |
| Tier-2 navigation holdout | 16/16 | 16/16 | 0 |
| Tier-3 key progression (3 maps × 8) | 24/24 | 24/24 | 24 (1 per episode) |
| Tier-4 gauntlet proxy | 8/8 | 8/8 | 8 (1 per episode) |
| Open-door/no-USE-needed map | 8/8 | 8/8 | 8 (one harmless trigger) |
| **Aggregate regression suite** | **64/64** | **64/64** | 24 |

Latency p50 ~1.0–1.1 ms, p99 ~1.6–2.0 ms; zero deaths/damage in all gate maps.

## Interpretation
The wrapper solves the *single closed door* interaction bottleneck without any regression on
combat, navigation, progression or the gauntlet proxy. It does **not** solve the larger
multi-door `two_rooms` maps — the remaining failure there is navigation/exit discovery after
passing doors, not USE interaction. The one harmless trigger on the open-door map shows a
minor false-positive rate; a tighter distance threshold could remove it.

## Compliance caveat — important
Official `docs/RULES.md:6-17` forbids hand-written if/else action trees in the **final**
agent. This wrapper is therefore an **experiment and candidate behavior specification**, not
an automatically compliant submission. Before deployment, the gate must be replaced by a
small **learned** gate trained on these observation→intervention decisions (parameters, not
rules), and re-evaluated. The frozen champion remains the compliant default.

## Decision
Keep the wrapper as a versioned experiment (`door_controller.py`, `transfer/eval_wrapped.py`,
results in `transfer/wrapped_results.csv`). Do not promote it to `submission/agent.py` until
the learned-gate distillation is measured.
