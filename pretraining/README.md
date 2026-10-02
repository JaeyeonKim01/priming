# Reconstruction pretraining

This package migrates the production B and L reconstruction recipes and the
B no-perceptual ablation. It does not launch training or claim a new numerical
reproduction. All migrated runtime files are unchanged copies of retained
source files; preparation and launch support live separately in `scripts/`.

| Recipe | Runtime | Auxiliary loss | Production execution |
|---|---|---|---|
| `b160_perceptual` | B production snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 4 nodes × 2 H200s |
| `b160_no_perceptual` | Same B snapshot | Entire auxiliary branch disabled | 4 nodes × 2 H200s |
| `l160_perceptual` | Retained L code snapshot | LPIPS 0.4 + ConvNeXt 0.1 | 2 nodes × 4 H200s |

All three use ImageNet-256, fixed corruption t=0.8, adaptive pixel MSE,
label dropout 0, seed 1, Muon LR 0.001 with a constant schedule, BF16,
EMA 0.9999, and global batch 1,024: eight ranks × microbatch 64 × accumulation 2.
There are 1,251 updates per epoch and 200,160 updates over 160 epochs.
Checkpoints save each epoch and retain 20-epoch milestones, including
`step_0100080.pth` (80 epochs) and `step_0200160.pth` (160 epochs).
These two intermediate/final checkpoints supply the corresponding post-training
recipes; separate 80-epoch configs are unnecessary. No L no-perceptual recipe
is included.

## Historical identity

The four released checkpoints trace to these pretraining runs:

- B-80/B-160: `OST_B_t08_e160_perceptual_seed1_20260912`.
- L-80/L-160: `OST_L_aux_unif_t08_38545843`.

The B trainer, deterministic crop implementation, and dependencies match the
retained September 12 production manifest. B explicitly uses math SDPA,
deterministic cuDNN/algorithms, a native CPU bicubic crop backward, stateless
flip/noise/crop seeds, and complete all-rank Muon/RNG recovery.

The L recipe comes from the **original executed arguments**, not the run
directory's subsequently overwritten 320-epoch config. Its original training
ran August 12–14, 2026 (Eastern time), with a restart at epoch 110. The 80-epoch
source was saved August 13 at 16:37; the 160-epoch source August 14 at 14:12.
The separate 320-epoch continuation started August 16.

`runtime/l` is the unchanged September 10 retained L-study code snapshot.
It is not an independently hash-verified snapshot of the August pretraining
allocation. Original arguments and logs verify the L recipe and checkpoint
lineage; they do not record which SDPA kernel was selected. The retained code
uses the original nondeterministic backend settings, ordinary RNG/dataloader
streams, and CUDA crop backward. Its checkpoint format stores rank-zero
optimizer state rather than B's complete all-rank state. These limitations are
preserved and documented, not silently replaced by the B implementation.

The perceptual thresholds are also preserved: B 0.8 and L 1.0. Both include
all examples at fixed t=0.8. The underlying objectives, model factories, and
optimizer code have not been refactored or unified.

See [production provenance](provenance/README.md),
[release lineage](provenance/release_lineage.json), and
[migration validation](reports/validation.json).

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

Use Python 3.11 with [requirements.txt](requirements.txt). The local `fdloss`
environment already provides these versions. CPU verification additionally
requires `pytest`; see [requirements-dev.txt](requirements-dev.txt).

Large assets remain external:

1. Packed ImageNet training data at `DATA_PATH/imagenet-latents-images/`
   (1,281,167 decoded 256×256 RGB images). B additionally uses its `val/`
   subdirectory (50,000 images) for fixed diagnostic panels.
2. Perceptual recipes: the original LPIPS `vgg.pth` calibration file,
   torchvision's `vgg16-397923af.pth` in `TORCH_HOME/hub/checkpoints/`, and the
   Hugging Face cache for `facebook/convnextv2-base-22k-224` in `HF_HOME/hub/`.

The no-perceptual recipe does not require or construct those encoders.
The supplied `configs/site.json` records this cluster's existing locations;
change it or use preparation flags on another machine. No parent `OST` source
imports or checkpoint links are needed to run the migrated Python code.
The site example still points to externally stored assets in the original
workspace; move those assets or supply new paths when moving the package.

## Check and prepare

From `OST_public`, using an environment with the dependencies installed:

```bash
python pretraining/scripts/verify.py --cpu
python -m pytest -q pretraining/tests

python pretraining/scripts/prepare.py \
  --recipe b160_perceptual \
  --bundle /scratch/my_pretraining/b160_bundle \
  --output /scratch/my_pretraining/b160_run
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
  pretraining/scripts/train.py --bundle /scratch/my_pretraining/b160_bundle
```

For the historical multi-node H200 layout on Kempner, from `OST_public`:

```bash
export PRETRAINING_ROOT="$PWD/pretraining"
export PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
# B recipes:
sbatch --nodes=4 --gpus-per-node=2 \
  pretraining/scripts/run.sbatch /scratch/my_pretraining/b160_bundle
# L recipe, after preparing its own bundle:
sbatch --nodes=2 --gpus-per-node=4 \
  pretraining/scripts/run.sbatch /scratch/my_pretraining/l160_bundle
```

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
