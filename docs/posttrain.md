# Distributional post-training

Prepare [packed ImageNet, class indices, frozen calibration, and encoders](assets.md).
The default source is the pinned 160-epoch B/L pretraining checkpoint.

The feature encoder settings are `matching.stack` and `data.encoder_cache` in
[`configs/posttrain/_base.yaml`](../configs/posttrain/_base.yaml).
`--encoder-cache PATH` selects the directory containing `mae_encoder.pt`,
`convnext_encoder.pt`, `calibration.json`, and the accompanying metadata/indices.
See the [encoder configuration/path table](assets.md#encoder-configuration-and-paths)
for SIM, offline weight caches, and the separate evaluation assets.

[`data/imagenet.py`](../data/imagenet.py) owns ImageNet real-reference loading
and checkpointable class/reference sampling.

```bash
python posttrain.py --config configs/posttrain/b160.yaml --check-config
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/b160
```

Recipes: `b160.yaml`, `l160.yaml`, `b160_sim.yaml`, `b160_energy.yaml`, and
`b160_laplacian.yaml`. They inherit common settings from
[`_base.yaml`](../configs/posttrain/_base.yaml). Full-feature recipes use the calibrated ResNet-MAE and
ConvNeXt outputs; SIM uses SigLIP, ViT-MAE, and Inception. Energy and Laplacian
change the discrepancy while retaining the full feature stack.

Each update selects 16 distinct global classes, with 64 generated samples,
64 independent generated references, and 128 real images per class. Frozen
features are computed online. Only the live generated population receives
gradients; encoder replay runs in microbatches of eight. Class-local losses
use the original scalar and parameter-gradient SUM conventions. OT retains
its fixed bandwidths, epsilon mixture, per-feature convergence, and coefficient
five. The matcher owns neither datasets nor generator/optimizer state.

Post-training starts fresh AdamW (LR 1e-5, betas 0.9/0.95, no weight decay,
gradient clipping one). The zero-based warmup starts at LR zero and reaches
1e-5 at update 125. The inherited pretraining EMA shadow/count continues with
decay 0.999. The horizon is 25,000 completed updates, with global batch 1,024,
BF16 generator execution, and eight ranks.

B post-training has been checked on H100 80 GB GPUs. Use eight H200s for L
post-training to match the original hardware: L-80/L-160 production runs peaked
above 85 GiB allocated per GPU, exceeding H100 80 GB capacity.

## Optional faster encoder execution

`b160_fast.yaml` and `l160_fast.yaml` retain the baseline objective, populations,
random streams, precision of the generator and ConvNeXt, optimizer, and evaluation
schedule. They change only frozen encoder execution:

- `matching.microbatch: 32` (the baseline remains eight).
- `matching.mae_bf16_channels_last: true` stores MAE convolution weights once in
  BF16/channels-last; normalization parameters remain FP32.
- `matching.compile_encoders: true` compiles family encodes and gradient replay
  with static shapes and Inductor layout optimization disabled. A dispatch-mode
  measurement uses eager execution without disabling later compilation.

These settings change floating-point rounding; they are not bit-identical to the
baseline. The original encoder files, calibration digest, and verification remain
unchanged. Optimized checkpoints record execution settings and reject incompatible
resumes. Start each configuration in a new output directory. Compilation adds
startup cost, including after resume; compare warmed updates separately.

For a bounded benchmark with update timings and allocated/reserved GPU peaks per
rank (saved under `OUTPUT/benchmark/`):

```bash
torchrun --standalone --nproc_per_node=8 scripts/benchmark_posttrain.py --steps 130 -- \
  --config configs/posttrain/b160_fast.yaml --output outputs/benchmark/b160_fast
```

Use the multi-node launcher from [setup](setup.md) when eight GPUs span nodes.
Benchmark baseline and fast settings on identical hardware, CPU allocations and
data storage. The wrapper always limits execution to at most 200 updates per
invocation; it does not launch full production training. Short runs establish
execution and recovery, not final FID/FDr5 reproduction.

## Initialization and resume

Omitting `--init-checkpoint` uses the released pretrained B-160 or L-160 file
selected by the recipe and verifies its release checksum. To use another
released pretrained checkpoint, including B/L-80, pass its path explicitly.
Choose `b160.yaml` for B weights or `l160.yaml` for L weights; the recipe name
selects post-training settings, not a minimum pretraining duration.

```bash
# Your compatible perceptual pretraining checkpoint; fresh post-training.
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/custom_b \
  --init-checkpoint outputs/pretrain/b160/checkpoints/latest.pth

# Exact continuation of the latest renewed post-training checkpoint.
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/custom_b --resume
```

Custom initialization accepts a **full compatible Priming pretraining
checkpoint at any saved step**. It must contain `model`, the EMA 0.9999 shadow
and counter in `model_ema`, and consistent `step` / `current_step` fields.
Tensor names and shapes must match the selected B/L architecture. B also
requires the perceptual pretraining `contract`; a supplied contract must match
the model/class settings and enable perceptual loss. Checkpoints written by
the corresponding perceptual `pretrain.py` recipe satisfy this format.
Weights-only exports, arbitrary external architectures, and the packaged
no-perceptual B checkpoints are not supported initialization formats.

For L, replace both recipe/model paths in the example with their L equivalents.
Initialization loads online weights, inherits EMA, and starts a fresh AdamW
optimizer; it does not resume the pretraining optimizer or its update counter.
Use a new output directory for each initialization run.

`--resume PATH` selects a checkpoint explicitly. Resume validates the recipe,
shared source, packed-data metadata, calibrated encoder identity, world size,
EMA origin, counters, all-rank main RNG, and private class/reference/noise
streams. It does not require the original initialization file. Historical
production training states are available in the original checkout; renewed
training recovery uses the explicit renewed checkpoint contract.

`--data-path`, `--encoder-cache`, `--hf-home`, and `--torch-home` select prepared
local assets. `--smoke-steps N` caps updates for that invocation and saves its
state. Scheduled evaluation remains active at any milestone reached.

## Outputs and scheduled evaluation

Checkpoints use completed-update names, for example `checkpoints/step_001250.pth`,
with `checkpoints/latest.json` selecting the latest save. `training.jsonl`
records diagnostics; `run.json` binds the run identities.

Every 1,250 updates, both online and EMA weights receive 50,000-image draws for
FDr6, native Inception Score, and k-nearest-neighbor precision/recall. This
training-time protocol includes ConvNeXt and the ADM reference. It is separate
from [official final FID/FDr5](eval.md).

The checkpoint is saved before evaluation. One atomic `evaluation/step_001250.json`
record is written after both draws succeed. An interrupted draw is retried
before the next update on resume; complete records are reused only when their
checkpoint hash and protocol match. Evaluation restores Python, NumPy, and
all-device Torch RNG even after failure.

USR1, `PREEMPT_FLAG_FILE`, or an output `STOP_REQUESTED` file requests a
checkpointed stop at an update boundary.
