# Checkpoint migration and reuse

We keep old checkpoints rather than discarding them. `training/migrate.py` copies every
tensor whose name **and shape** match between configurations and reinitialises only the
incompatible layers. Every migration records `copied_layers` and `skipped_layers` inside the
new checkpoint so the provenance is auditable.

## Legacy Bernoulli → macro categorical

Legacy `submission/weights/keystone.pt` (30,208 steps, full KEYSTONE with memory) used nine
independent Bernoulli logits. The macro baseline uses a categorical over 17 legal button
combinations. Migration result:

```
python -m training.migrate --src submission/weights/keystone.pt \
    --config configs/m5_full.json --out experiments/migrated_full/keystone.pt
copied 46 tensors, skipped 2
  skipped actor.2.weight (9, 128)
  skipped actor.2.bias (9,)
```

Preserved: ray/vitals/depth/entity encoders, GRU, event-memory write/read parameters,
goal head and embeddings, confidence/risk head, critic, actor hidden layer. Reinitialised:
only the actor output layer. The migrated checkpoint is stored at
`experiments/migrated_full/keystone.pt` and is **not** assumed better — it must be evaluated
against random init on the same held-out maps.

## Compatibility rules

- A tensor is copied only if name and shape match exactly.
- Config JSON is rewritten to the target architecture; the source config is preserved in
  the source checkpoint.
- `step` is reset to 0 in the migrated checkpoint so training logs are not misleading.
- `migrated_from`, `copied_layers`, `skipped_layers` are stored for provenance.

## External checkpoints

No external checkpoint has been integrated yet. See `docs/reuse-and-transfer-audit.md` for
the verified audit. If an external encoder is later found compatible, it can be imported
with the same name/shape rule or used as a distillation teacher (training only).
