# Reconstruction pretraining

Train Priming-B with or without perceptual loss, or Priming-L with perceptual
loss, using the production reconstruction recipes below.

| Recipe | Runtime | Auxiliary loss | Suggested execution |
|---|---|---|---|
| `b160_perceptual` | B production snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 8 H200 GPUs (validated) |
| `b160_no_perceptual` | Same B snapshot | Entire auxiliary branch disabled | 8 H200 GPUs (validated) |
| `l160_perceptual` | Retained L code snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 8 H200 GPUs (validated) |

All three use ImageNet-256, fixed corruption t=0.8, adaptive pixel MSE,
label dropout 0, seed 1, Muon LR 0.001 with a constant schedule, BF16,
EMA 0.9999, and global batch 1,024: eight ranks × microbatch 64 × accumulation 2.
There are 1,251 updates per epoch and 200,160 updates over 160 epochs.
Checkpoints save each epoch and retain 20-epoch milestones, including
`step_0100080.pth` (80 epochs) and `step_0200160.pth` (160 epochs).
These two intermediate/final checkpoints supply the corresponding post-training
recipes; separate 80-epoch configs are unnecessary. No L no-perceptual recipe
is included.

B and L preserve their original backend and resume behavior. See
[reproducibility notes](provenance/README.md#historical-training-details) when
comparing a new run with the released checkpoints.

## Files and dependencies

- `configs/`: three complete recipes plus a site-specific asset-path example.
- `runtime/{b,l}/`: trainers and their transitive local Python dependencies,
  including model registry imports, local DistMatch, and the TVM JVP dependency
  with its license. Disabled endpoint-OT code is retained because the original
  generic trainer imports it; these recipes do not enable endpoint OT.
- `provenance/`: sanitized source recipes, runtime hashes, and reproducibility notes.
- `scripts/`: bundle preparation, torchrun/Slurm launch support, and CPU checks.
- `tests/`: packaging checks. The original B recovery tests are also retained
  unchanged inside its runtime.

Create and activate the shared `priming` environment using
[the setup guide](../docs/setup.md). The pinned environment includes the CPU
verification dependencies. All paths resolve from the `Priming/` repository.

Large assets remain external:

1. Packed ImageNet training data at `DATA_PATH/imagenet-latents-images/`
   (1,281,167 decoded 256×256 RGB images). B additionally uses its `val/`
   subdirectory (50,000 images) for fixed diagnostic panels.
2. Perceptual recipes: the original LPIPS `vgg.pth` calibration file,
   torchvision's `vgg16-397923af.pth` in `TORCH_HOME/hub/checkpoints/`, and the
   Hugging Face cache for `facebook/convnextv2-base-22k-224` in `HF_HOME/hub/`.

The no-perceptual recipe does not require or construct those encoders.
The exact processed dataset and pinned perceptual weights can be prepared with:

```bash
# Accept the dataset's ImageNet terms on HF first; see the asset guide.
hf auth login
python scripts/download_assets.py --group dataset
python scripts/download_assets.py --group pretraining
```

The supplied `configs/site.json` uses `data/imagenet`, `assets/huggingface`,
`assets/torch`, and `assets/lpips/vgg.pth`, all relative to the repository.
See [asset layout](../docs/assets.md). No parent OST source is required.

## Check and prepare

From `Priming`, using an environment with the dependencies installed:

```bash
python pretraining/scripts/verify.py --cpu
python -m pytest -q pretraining/tests

python pretraining/scripts/prepare.py \
  --recipe b160_perceptual \
  --bundle outputs/pretraining/b160_bundle \
  --output outputs/pretraining/b160_run
```

Choose `b160_no_perceptual` or `l160_perceptual` for the other recipes.
Both bundle and output paths must be new, separate directories. Preparation
creates only the bundle; it does not initialize a model or submit a job.
Optional `--site`, `--data-path`, `--hf-home`, `--torch-home`, and
`--lpips-weights` select asset locations. Output/logging paths are resolved
without modifying the checked-in scientific recipes. W&B is disabled in the
release launcher.

The bundle freezes effective config and runtime hashes, data metadata hashes,
Arrow shard filenames/sizes/mtimes, and perceptual asset hashes. It does not
hash all ImageNet image bytes. LPIPS is linked into the bundle's working
directory; encoder caches are read offline. Preparation of a different local
dataset establishes its identity but does not prove equivalence to the
historical image bytes.

## Launch when training is desired

On one eight-GPU machine:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretraining/scripts/train.py --bundle outputs/pretraining/b160_bundle
```

On a Slurm cluster with H200 GPUs, submit from `Priming`:

```bash
export PRETRAINING_ROOT="$PWD/pretraining"
export PYTHON="$(command -v python)"
sbatch pretraining/scripts/run.sbatch "$PWD/outputs/pretraining/b160_bundle"
# To use multiple nodes, keep the total at eight ranks:
sbatch --nodes=2 --gpus-per-node=4 \
  pretraining/scripts/run.sbatch "$PWD/outputs/pretraining/l160_bundle"
```

The default is one node with eight GPUs. Add your cluster's partition/account
flags if required. Hardware/topology changes do not imply bitwise equivalence;
use the original microbatch, global batch, precision, and runtime settings.

The Slurm script requests a checkpoint on USR1 and waits for the ranks to
finish saving. A preempted segment exits with code 75. To continue in a new
allocation, reuse the same bundle and add `RESUME=1` to the `sbatch` environment,
or `--resume` to the torchrun entrypoint. There is no automatic submission or
requeue loop. L retains its historical rank-zero optimizer restore, so an L
restart is **not** claimed to reproduce an uninterrupted all-rank trajectory.
B validates and restores its complete all-rank state. Fresh bundles have new
manifest identities; they do not transparently resume original historical B
checkpoints.

Runtime files and prepared bundle configs must remain unchanged during a run.
Changing them requires a new bundle and output. Large checkpoints stay in the
explicit output directory. The B trainer retains its fixed diagnostic
panels; pretraining does not launch a terminal 50k evaluation.

## Post-training handoff

The default post-training reproduction uses the **released 160-epoch online
weights** and pins their SHA256. To use your own perceptual B/L checkpoint,
pass `--init-checkpoint` when preparing a standalone post-training job:

```bash
python training/scripts/prepare.py --arm b160 \
  --init-checkpoint outputs/pretraining/b160_run/checkpoints/step_0100080.pth \
  --run-name my_b_posttraining --job outputs/posttraining_allocations/my_b
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job outputs/posttraining_allocations/my_b
```

The example uses the 80-epoch milestone; any saved pretraining step is accepted
if its model and EMA are compatible. `b160` and `l160` select the packaged
post-training recipe, not a required source epoch. Use `--arm l160` for L.
Prepare the [post-training assets](../docs/assets.md#post-training-encoders-and-ot-calibration)
first; with an explicit source, downloading the released pretrained generator
is unnecessary.

Preparation records and verifies your source's SHA256, model schema, and EMA
counter. Post-training starts from online weights, inherits the EMA shadow and
counter, and uses a fresh optimizer and post-training counter. B checkpoints
must retain the perceptual pretraining contract. See
[source requirements and recovery](../training/README.md#start-from-your-own-pretraining-checkpoint).
Use [the final evaluator](../eval/README.md) with `--checkpoint` after the new
post-training run completes its 25,000 updates.
