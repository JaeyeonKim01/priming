# Data and external assets

Run from `Priming` with the [pinned environment](setup.md) active. Keep the
checkout on a large disk, or pass `--root /large-disk/Priming` to the download
helper and configure the asset paths as described in [setup](setup.md).

| Operation | Required assets |
|---|---|
| Sampling | One [generator checkpoint](checkpoints.md) |
| FID/FDr5 evaluation | Generator, evaluation encoders, precomputed reference statistics |
| Reconstruction pretraining | Packed ImageNet; perceptual recipes also need LPIPS, VGG16, ConvNeXt |
| OT post-training | Packed ImageNet with indices/calibration, converted MAE/ConvNeXt, pretrained generator, training diagnostic assets |

The [encoder manifest](../releases/encoders.json) pins upstream revisions,
filenames, sizes, and SHA256. The helper verifies every downloaded file and
prepares the offline Hub cache. Existing files or cache revisions that differ
from the release are rejected.

## Packed ImageNet dataset

Dataset repository: [jaykim1102/priming](https://huggingface.co/datasets/jaykim1102/priming).
This is separate from the generator's **model** repository with the same name.
The [release manifest](../releases/dataset.json) pins the complete dataset
snapshot and checksums. The helper downloads that verified revision.

Accept the ImageNet terms on the dataset page once. Access uses automatic
approval, with no manual approval queue. Authenticate the download client:

```bash
hf auth login
python scripts/download_assets.py --group dataset
```

Allow approximately **259 GB** for data and index files, plus weights and
training outputs. The release preserves the original Arrow shards and row
ordering: **1,281,167 train and 50,000 validation images**, RGB 256x256.
No image preprocessing, repacking, or latent encoding is needed.

```text
data/imagenet/imagenet-latents-images/
  dataset_info.json
  state.json
  data-*.arrow
  val/
    dataset_info.json
    state.json
    data-*.arrow
assets/ot/representations/
  dataset_labels.npy
  class_order.npy
  class_counts.npy
  calibration.json
  downloads.json
  converted_manifest.json
```

Use `datasets.load_from_disk` on the training directory or its `val/`
subdirectory, not `load_dataset` or an ImageFolder loader. Despite the retained
directory name, the data contain images, not latent tensors. To check every
local file without downloading:

```bash
python scripts/download_assets.py --group dataset --verify-only
```

The [ImageNet Terms of Access](https://image-net.org/download.php) apply;
see the [dataset card](huggingface-dataset-card.md) for attribution.
[HF documents](https://huggingface.co/docs/hub/datasets-gated) browser acceptance
and authenticated downloads.

## Reconstruction pretraining encoders

```bash
python scripts/download_assets.py --group pretraining
```

No weight conversion is needed:

| Component | Pinned source |
|---|---|
| LPIPS VGG calibration | LPIPS v0.1 `vgg.pth`; MD5 `d507d7349b931f0638a25a48a722f98a` |
| VGG16 | torchvision `vgg16-397923af.pth` |
| ConvNeXt V2 | `facebook/convnextv2-base-22k-224`, revision `758ff0922dc09136abb55774e7f8b1e1bd0dc344` |

Exact filenames and SHA256 are in the encoder manifest. The B no-perceptual
recipe does not require these encoders. Continue with the
[pretraining guide](../pretraining/README.md).

## Post-training encoders and OT calibration

First download the dataset. For default reproduction, also download the
[160-epoch pretrained generator](checkpoints.md). With
[`--init-checkpoint`](../training/README.md#start-from-your-own-pretraining-checkpoint),
use your own compatible pretraining checkpoint instead.
Then prepare diagnostic encoders/statistics and the original conversion inputs:

```bash
python scripts/download_assets.py --group posttraining
python scripts/download_assets.py --group conversion
python -m pip install -r environment/requirements-conversion.txt
python scripts/convert_ot_encoders.py
```

Conversion runs on CPU; allow sufficient RAM (64 GB recommended) and about
20 GB for source/converted weights. The optional MessagePack package is only
needed for conversion. The converter verifies source hashes, requires strict
parameter loading, verifies the output hashes, and refuses existing outputs.

- **Pixel MAE:** `Goodeat/drifting`, revision
  `1a5afa9fc22c3beefe06e4689e84d3e6500b5983`, files under
  `models/mae/jax/mae_pixel_640/`. Map its EMA parameter tree to the included
  PyTorch encoder, including Dense/kernel layouts, and exclude the unused
  decoder/classifier. This pixel MAE differs from the evaluation timm ViT MAE.
- **ConvNeXt:** the same pinned Facebook source as above, using the original
  `pytorch_model.bin`. Map parameter names to the included encoder and exclude
  the classification head.
- **Runtime behavior:** the full-MAE encoder preserves Flax GroupNorm semantics
  with epsilon `1e-6`. MAE uses BF16 compute and ConvNeXt uses FP32. Keep these
  implementations with the frozen calibration.

The dataset includes **class indices and frozen OT calibration**; reuse them
for the released experiment. Calibration contains 97 feature bandwidths,
weights, dimensions, 2,000 selected training rows, and the encoder source hash.
Selection uses a seed-0 PyTorch permutation of stable class-sorted training
rows, without generator samples or evaluation scores.

```bash
python scripts/prepare_ot_indices.py \
  --dataset-dir data/imagenet/imagenet-latents-images \
  --output-dir assets/ot/representations --verify-only \
  --calibration assets/ot/representations/calibration.json
python training/scripts/verify.py --assets --arm b160
```

Select `--arm l160` for L. Omitting `--arm` verifies every packaged recipe and
therefore requires both B-160 and L-160 pretrained generator checkpoints.
For a custom initialization, `prepare.py --init-checkpoint PATH` performs asset
verification together with validation of your source; the default released-source
verification command above is unnecessary.

To independently rebuild the three index files, omit `--verify-only` and
choose a new `--output-dir`. Read labels in dataset order, stable-sort by label
for `class_order.npy`, and count each class for `class_counts.npy`. Changing
images or row ordering requires new calibration; the index tool does not
recalibrate a changed dataset.

Public provenance JSON has portable paths. The asset verifier accepts the
explicitly recorded original and published metadata identities; numerical
calibration and weights are unchanged. Resume checkpoints record the metadata
digest, so do not swap metadata editions during a run.

The post-training group also supplies precomputed Inception reference features
for training-time precision/recall diagnostics, avoiding another reference-image
download. SIM calibration is included in the source tree and reuses the
downloaded timm/Inception encoders. Continue with the
[training guide](../training/README.md).

## Evaluation: download existing reference statistics

```bash
python scripts/download_assets.py --group evaluation
```

**No ImageNet dataset, OT calibration, or reference-statistics computation is
required for released FID/FDr5 evaluation.** Download the frozen evaluation
encoders and precomputed real-image statistics. The evaluator generates images,
measures their features, and compares their moments with the supplied references.

| Component | Reference in `assets/fid_stats/` |
|---|---|
| Official FID | `jit_in256_stats.npz` |
| FDr5 Inception/ADM | `guided_diffusion_stats.npz` |
| FDr5 DINOv2 | `vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz` |
| FDr5 MAE | `vit_large_patch16_224_mae_in256_t224_stats.npz` |
| FDr5 SigLIP | `vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz` |
| FDr5 CLIP | `vit_large_patch14_clip_224_openai_in256_t256_stats.npz` |

The official FID reference is from the pinned
[JiT release](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/fid_stats/jit_in256_stats.npz),
SHA256 `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
Frozen FDr5 statistics are hosted alongside the Priming generator weights.
These are distinct references; do not substitute one for the other.
FDr5 excludes ConvNeXt; its additional statistic is used by training diagnostics.

Encoder model IDs and exact revisions are in the [manifest](../releases/encoders.json).
The helper prepares the pinned snapshots and `refs/main` for offline loading.
Reference files hosted in the Priming model repository require the same read
access as [generator checkpoints](checkpoints.md), independently of dataset access.
See [evaluation commands](../eval/README.md) for the sampling/precision protocol
and a 1,000-image pilot.

## Baseline generators

Baseline evaluation additionally requires the respective released generator
and, where applicable, VAE. These are separate from shared evaluation encoders:
`pMF-L_256.pth`, `pMF-H_256.pth`, `iMF-XL.pth`, and `JiT-H.pth` belong in
`assets/baselines/checkpoints/`; identities are recorded in
[the baseline adapter](../eval/baseline_loaders.py). W-Flow uses
`assets/baselines/comparison/wflow/checkpoints/latent_sota_XL_ot/state_00180000.pt`
and `assets/baselines/comparison/vae`. Converted Drifting uses
`assets/baselines/comparison/drifting/models/gen/jax/pixel_{B,L}_sota/converted_ema.pt`.
The asset helper covers Priming data and shared encoders/references; it does
not download these baseline generator checkpoints.
