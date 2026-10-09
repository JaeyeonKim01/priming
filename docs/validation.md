# Validation

The simplified workflows are `pretrain.py`, `posttrain.py`, and `eval.py`.
The model, data, losses, metrics, and checkpoint helpers have one shared source
tree. CPU comparisons read an original checkout selected by
`PRIMING_REFERENCE_ROOT`; tests disable bytecode writes to that checkout.

```bash
PRIMING_REFERENCE_ROOT=~/priming_old PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest discover -s tests -q
python -B -m unittest discover -s scripts/tests -q
python scripts/check_portability.py --reference-root ~/priming_old
python scripts/check_public_tree.py
```

Comparisons cover the original model architecture/tensor schema, reconstruction
and perceptual losses, B/L seed and accumulation behavior, OT/energy/Laplacian
losses, transport plans, diagnostics, gradients, class/reference/noise sampling,
optimizer warmup, EMA, checkpoint counters, and exact resumed updates.

Encoder checks use compact real MAE/ConvNeXt architectures and complete Inception
networks with synthetic local weights, plus test factories for heavyweight timm
models. They check preprocessing, precision, feature identities, input-gradient
replay, architecture schemas, and local asset guards.

Scheduled evaluation checks execute the original online/EMA draw with synthetic
features and compare exact moments, metric values, image ordering, RNG, native
FP32 Inception, and distributed scalar rounding. Standalone evaluation checks
execute the original padded sampling loop and compare image bytes, official
Inception outputs, feature arrays, RNG, reference-dtype FID arithmetic, and FDr5.

The portability check copies source into a temporary checkout, checks all CLI
help/config paths, runs CPU tests, and checks imports without local assets.
The public-tree scan checks source text for known private infrastructure paths;
it does not scan Git history or binary metadata.

## GPU validation

CPU equivalence is not full GPU reproduction. Compiled GPU trajectories,
released heavyweight encoder weights, all-rank recovery, and 50,000-image metrics
must be verified on an allocation with prepared assets. The current renewal was
checked in a CPU-only session; it does not claim new full training runs or paper
metrics.

Use `--smoke-steps` for bounded fresh training checks. Compare original and
renewed runs using identical hardware, ranks, seeds, batches, weights, and
asset versions, including the next update after a checkpoint resume. Run a
1,000-image evaluation pilot before the full protocol.
