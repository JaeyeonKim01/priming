# Data and external assets

The eight generator checkpoints are hosted on
[Hugging Face](https://huggingface.co/jaykim1102/priming); see the
[download commands](checkpoints.md). Source code is self-contained.
Large datasets, checkpoints, encoder weights,
and statistics are separate assets and are ignored by Git. Their default
locations are inside the `Priming` repository:

```text
priming/
  data/imagenet/imagenet-latents-images/  # packed Hugging Face dataset
  data/prc/VIRTUAL_imagenet256_labeled/   # precision/recall reference
  data/prc/VIRTUAL_imagenet256_labeled_inception.pt
  checkpoints/                          # four post-trained release checkpoints
  checkpoints/pretraining/              # four pretrained checkpoints
  assets/lpips/vgg.pth
  assets/huggingface/hub/                # full Hub cache, including refs/snapshots/blobs
  assets/torch/hub/checkpoints/          # VGG16 and Inception
  assets/fid_stats/                      # reference .npz statistics
  assets/ot/representations/             # converted OT encoders, labels, calibration
  assets/ot/features/                    # OT feature assets and calibrations/
  assets/baselines/checkpoints/          # pMF, iMF, JiT weights
  assets/baselines/comparison/           # W-Flow, converted Drifting, VAE
  outputs/                              # new runs, bundles, evaluation artifacts
```

Use real directories when exporting a portable copy. Symlinks to external
storage are supported locally, but their targets must be copied or mounted on
the destination machine. Neither the code nor default configs reference an
original user's home directory.

## Reconstruction pretraining

Obtain ImageNet using your own authorized access, and supply the packed
`imagenet-latents-images` dataset: 1,281,167 training images, decoded RGB at
256×256, with its original Arrow shards, `dataset_info.json`, and `state.json`.
B also requires the 50,000-image `val/` split used by its diagnostic panels.
This format is a Hugging Face dataset saved to disk, not an ImageFolder tree.
Preparation validates layout, metadata and shard identities. A newly repacked
dataset is not claimed to have the original byte identity or ordering.

Perceptual runs additionally need:

- `assets/lpips/vgg.pth`: original LPIPS calibration (MD5
  `d507d7349b931f0638a25a48a722f98a`).
- `assets/torch/hub/checkpoints/vgg16-397923af.pth`: torchvision VGG16 weights.
- Complete Hub cache for `facebook/convnextv2-base-22k-224`, including the
  `refs/main` revision and corresponding snapshot.

The B no-perceptual recipe does not load these encoders. Pretraining preparation
hashes the selected perceptual assets and uses them offline thereafter.

## Distributional post-training

Place the 160-epoch pretrained weights at the paths in
[pretraining_manifest.json](../checkpoints/pretraining_manifest.json).
[training/assets.json](../training/assets.json) records repository-relative
locations, sizes and SHA256 for the retained production assets. Copy the whole
OT cache and Hub cache trees, preserving relative symlinks and snapshots, not
just individual weight blobs. The OT stack is `convnext_full_mae`.

```bash
python training/scripts/verify.py --assets
```

The original converted encoders, dataset index/calibration files, and reference
statistics are needed for exact production comparisons. They are not yet a
published downloadable asset bundle. The repository supplies their identities
and paths; it does not pretend that downloading a generic encoder reconstructs
these calibrated assets. After receiving the asset export, copy its `assets/`
and `data/` trees into `Priming/`. Checkpoint download instructions are
[separate](checkpoints.md).

## Evaluation

Priming evaluation needs only the selected post-trained generator, frozen
FDr⁵ encoder caches/statistics, and the official Inception weights/reference;
it does not require ImageNet training shards. FDr⁵ uses the original DINOv2,
MAE, CLIP, SigLIP and Inception components. Keep their timm Hub revisions from
the supplied cache and checksums from `training/assets.json`.

The official FID reference `assets/fid_stats/jit_in256_stats.npz` must have
SHA256 `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
It must not be substituted with the training-time BF16 statistics.

Baselines additionally use `pMF-L_256.pth`, `pMF-H_256.pth`, `iMF-XL.pth`, and
`JiT-H.pth` in `assets/baselines/checkpoints`; their checksums are recorded in
[the baseline adapter](../eval/baseline_loaders.py). W-Flow uses
`assets/baselines/comparison/wflow/checkpoints/latent_sota_XL_ot/state_00180000.pt`
and the VAE in `assets/baselines/comparison/vae`. Drifting uses
`assets/baselines/comparison/drifting/models/gen/jax/pixel_{B,L}_sota/converted_ema.pt`.
iMF uses the original cached VAE selected by its included upstream sampler.
All corresponding sampler and metric code is included in `eval/vendor` and
`eval/runtime`; a second source checkout is not required.
