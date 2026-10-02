# Reconstruction pretraining

Train Priming-B with or without perceptual loss, or Priming-L with perceptual
loss, using the production reconstruction recipes below.

| Recipe | Runtime | Auxiliary loss | Suggested execution |
|---|---|---|---|
| `b160_perceptual` | B production snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 8 H100/H200 GPUs |
| `b160_no_perceptual` | Same B snapshot | Entire auxiliary branch disabled | 8 H100/H200 GPUs |
| `l160_perceptual` | Retained L code snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 8 H100/H200 GPUs |

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
- `provenance/`: original configs/arguments, source hashes, historical evidence,
  environment versions, and links from released models to pretraining sources.
- `scripts/`: bundle preparation, torchrun/Slurm launch support, and CPU checks.
- `tests/`: packaging checks. The original B recovery tests are also retained
  unchanged inside its runtime.

Create and activate the shared `priming` environment using
[the setup guide](../docs/setup.md). The pinned environment includes the CPU
verification dependencies. All paths resolve from the `priming/` repository.

Large assets remain external:

1. Packed ImageNet training data at `DATA_PATH/imagenet-latents-images/`
   (1,281,167 decoded 256×256 RGB images). B additionally uses its `val/`
   subdirectory (50,000 images) for fixed diagnostic panels.
2. Perceptual recipes: the original LPIPS `vgg.pth` calibration file,
   torchvision's `vgg16-397923af.pth` in `TORCH_HOME/hub/checkpoints/`, and the
   Hugging Face cache for `facebook/convnextv2-base-22k-224` in `HF_HOME/hub/`.

The no-perceptual recipe does not require or construct those encoders.
The supplied `configs/site.json` uses `data/imagenet`, `assets/huggingface`,
`assets/torch`, and `assets/lpips/vgg.pth`, all relative to the repository.
See [asset layout](../docs/assets.md). No parent OST source is required.

## Check and prepare

From `priming`, using an environment with the dependencies installed:

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

On a Slurm cluster with H100/H200 GPUs, submit from `priming`:

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
explicit output directory. The B trainer retains its original fixed diagnostic
panels; no terminal 50k evaluation jobs are attached by this migration.

## Post-training handoff

Use the selected 80/160-epoch checkpoint's **online generator weights** with
a fresh post-training optimizer. Preserve the existing post-training EMA
loading convention described in [the OT guide](../training/README.md).
This migration does not change the existing OT reproduction package or its
historical checkpoint identities. Final release evaluation remains in
[eval](../eval/README.md).
