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

## Check the training setup before a full run

`--check-config` validates recipe settings only; it does not check local assets
or run training. Use the GPU check below to verify that training can start.

After preparing the same data and encoders, run two updates of each stage in
new output directories. Two updates exercise the nonzero post-training learning
rate after its first, zero-LR warmup update:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml \
  --output outputs/pretrain/check_b --smoke-steps 2
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml \
  --init-checkpoint outputs/pretrain/check_b/checkpoints/latest.pth \
  --output outputs/posttrain/check_b --smoke-steps 2
```

The first command creates the checkpoint used by the second; no released
generator checkpoint is needed for this route. To check released initialization,
download the matching pretrained checkpoint and omit `--init-checkpoint` in a
separate post-training run. For L, use `l160_perceptual.yaml`, `l160.yaml`, distinct
output paths, and eight H200s. On a multi-node allocation, replace the standalone
launcher with the [Slurm example](setup.md#training-across-multiple-nodes).

Pretraining writes `checkpoints/latest.pth`; the post-training check writes
`checkpoints/step_000002.pth` and `checkpoints/latest.json`. A successful check
establishes startup, updates, and saving; it does not establish convergence or
exercise the scheduled evaluation at update 1,250. Start full training in new
output directories without `--smoke-steps`.

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

## GPU validation

CPU checks do not establish GPU training trajectories or full reproduction.
Use `--smoke-steps` for bounded training checks, and verify checkpoint recovery
on your GPU setup. Compare runs using identical hardware, ranks, seeds, batches,
weights, and asset versions, including the next update after a checkpoint resume.
Run a 1,000-image evaluation pilot before the full 50,000-image protocol in the
[evaluation guide](eval.md).
