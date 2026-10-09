# Validation

## Follow the complete ImageNet workflow

Run from the repository root in the [shared environment](setup.md), after
accepting the dataset terms. These are **full training commands**: 160 epochs
of pretraining, then 25,000 post-training updates. For short startup checks,
use the stage guides' `--smoke-steps` option and separate output directories.

```bash
hf auth login
for group in dataset pretraining posttraining conversion evaluation; do
  python scripts/download_assets.py --group "$group"
done
python -m pip install -r old/environment/requirements-conversion.txt
python scripts/convert_ot_encoders.py

torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml --output outputs/pretrain/my_b
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/my_b \
  --init-checkpoint outputs/pretrain/my_b/checkpoints/latest.pth

torchrun --standalone --nproc_per_node=4 eval.py \
  --model priming_b_160 --checkpoint outputs/posttrain/my_b/checkpoints/step_025000.pth \
  --num-images 50000 --seeds 1 2 3 --batch 32 --output-dir outputs/eval/my_b
```

The converter creates new files and refuses to overwrite existing converted
encoders; run it once after preparing the source weights. To use released
pretrained weights instead, download `checkpoints/pretraining/priming_b_160.pth`
using the [checkpoint guide](checkpoints.md), skip pretraining, and omit
`--init-checkpoint`. For L, use `l160_perceptual.yaml`, `l160.yaml`, and
`--model priming_l_160`, with distinct output paths.

These commands assume assets under the repository defaults. The
[encoder path table](assets.md#encoder-configuration-and-paths) and
[storage guide](setup.md#assets-outside-the-checkout) explain external paths.
Training requires eight ranks; follow your scheduler's multi-node torchrun
launch procedure if those GPUs span nodes.

## CPU and source checks

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

## ImageNet refactor audit

See the [October 9 audit](imagenet_refactor_audit.md) for the comparison against
`443db44`, completed checks, and changes affecting existing workflows.

## GPU validation

CPU equivalence is not full GPU reproduction. Compiled GPU trajectories,
released heavyweight encoder weights, all-rank recovery, and 50,000-image metrics
must be verified on an allocation with prepared assets. The refactor was initially checked in a CPU-only session. The
[October 9 audit](imagenet_refactor_audit.md) subsequently reproduced all 12
Priming paper-protocol evaluations; full training remains a separate scope.
Eight-H100 checks also passed B/L perceptual pretraining and both B
post-training initialization routes, but L post-training ran out of memory.
The follow-up on eight H200s passed both L initialization routes and B
no-perceptual pretraining. Original L production also used H200s.
See the audit's [training table](imagenet_refactor_audit.md#eight-gpu-training-checks)
for completed checks and untested routes.

Use `--smoke-steps` for bounded fresh training checks. Compare original and
renewed runs using identical hardware, ranks, seeds, batches, weights, and
asset versions, including the next update after a checkpoint resume. Run a
1,000-image evaluation pilot before the full protocol.
