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

### Current-checkout startup check — October 11

The current training source at `7fc7667` passed five fresh checks on eight H200
GPUs (four nodes, two GPUs each), using the production recipes with compilation
and BF16 enabled. Each stage completed two optimizer updates on real ImageNet:

| Workflow | Result | Peak allocated memory per GPU |
|---|---|---|
| B perceptual pretraining → its own checkpoint → B OT post-training | Passed both stages | 35.70 / 47.87 GiB |
| L perceptual pretraining → its own checkpoint → L OT post-training | Passed both stages | 59.78 / 81.32 GiB |
| Released B-160 pretrained checkpoint → B OT post-training | Passed | 47.87 GiB |

All eight ranks completed each stage. Saved model, optimizer, and EMA tensors
were finite; post-training inherited the source EMA counters and changed online
weights after the second update. The pinned environment checks, all eight recipe
checks, and all 66 CPU/reference and asset-helper tests passed, including in a
relocated source-only copy.

The encoder instructions were checked against the download manifest, converter,
and runtime loaders. Both ResNet-MAE and ConvNeXt were converted again into a
new output directory; strict parameter loading and the published whole-file
SHA256 checks passed. The setup guide now spells out the optional MessagePack
dependency for external storage, and the asset guide distinguishes matching,
SIM, and diagnostic encoders and the assets required at training startup.

This pass reused the pinned environment and prepared data/assets from October 9;
it did not repeat a fresh installation or full dataset download. It establishes
B/L perceptual pretraining and OT post-training startup, updates, and saving.
It does not cover full-training convergence, scheduled 50,000-image evaluation,
GPU resume equivalence, or GPU runs of SIM/energy/Laplacian ablations.
See the [machine-readable readiness record](readiness_20261011.json) for exact
scope, checkpoint checks, and memory measurements, and the audit below for the
earlier fresh-installation and download checks.

### Fresh training workflow audit — October 9

A separate checkout and newly created pinned Conda environment passed the
setup script, dependency checks, CPU/reference suite, and source-only
portability checks. Training encoder inputs, diagnostic weights/statistics,
dataset metadata/indices/calibration, and a released B pretrained checkpoint
were downloaded again through the public preparation tools. The 517 large
ImageNet Arrow shards reused the previously downloaded public release, with
sizes checked again; their class indices and calibration were revalidated.

This exposed a conversion packaging issue: the fresh MAE archive differed only
in PyTorch's non-tensor `serialization_id`. SHA256 comparisons matched all
119 other archive records. The converter now pins that metadata and its ZIP
CRCs, then requires the **unchanged whole-file release SHA256** before publishing
the file. Fresh MAE and ConvNeXt conversions both passed. Changed tensor data
remain rejected; the training and evaluation implementations are unchanged.

Using that new environment and newly converted encoders, eight H200 GPUs
completed two updates and saved-state audits for all five checks:

- B perceptual pretraining → its own checkpoint → B post-training.
- L perceptual pretraining → its own checkpoint → L post-training.
- Downloaded released B-160 pretraining checkpoint → B post-training.

All eight ranks completed each stage. Saved weights, optimizer and EMA states
were finite; post-training inherited the source EMA counters and changed online
weights. GPU execution used offline asset loading. These checks do not cover
full-training convergence, scheduled 50,000-image training evaluation, or GPU
resume equivalence. Prior B no-perceptual and released-L initialization checks
are in the [refactor audit](imagenet_refactor_audit.md).

See the [machine-readable workflow record](training_workflow_audit.json) for
environment versions, public-access checks, release identities, and per-case
results. ImageNet still requires accepting its dataset terms and logging in.

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
