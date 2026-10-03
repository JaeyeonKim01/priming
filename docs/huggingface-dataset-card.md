---
license: other
license_name: imagenet-terms-of-access
license_link: https://image-net.org/download.php
task_categories:
  - image-classification
language:
  - en
size_categories:
  - 1M<n<10M
extra_gated_heading: ImageNet terms of access
extra_gated_description: "This dataset contains processed ImageNet images. Read the ImageNet Terms of Access at https://image-net.org/download.php before requesting access. Access is automatically approved after acceptance."
extra_gated_fields:
  I have read and agree to the ImageNet Terms of Access: checkbox
---

# Priming: packed ImageNet-256 training data

The exact packed image dataset used by Priming, with 1,281,167 training
images and 50,000 validation images. Images are embedded RGB 256x256 images;
fields are `id`, `image`, and `label` (class IDs 0–999). The dataset name
`imagenet-latents-images` is retained for loader compatibility; this release
contains images, not precomputed latent tensors.

ImageNet's original terms apply. This is a processed research release, not
an ImageNet-endorsed distribution or a new license for the underlying images.
Users must accept the [ImageNet Terms of Access](https://image-net.org/download.php).
Credit: Deng et al., *ImageNet: A Large-Scale Hierarchical Image Database*,
CVPR 2009; Russakovsky et al., *ImageNet Large Scale Visual Recognition
Challenge*, IJCV 2015.

## Use with Priming

Get the code and environment from
[Priming](https://github.com/JaeyeonKim01/one_step_priming).
Accept the access terms once on this page, then authenticate your machine:

```bash
hf auth login
hf download jaykim1102/priming --repo-type dataset --local-dir /large-disk/Priming
```

The download preserves the repository-relative `data/` and `assets/` layout.
Use the revision and SHA256 manifest linked in the code's data setup guide
for the pinned release. Downloads require approximately 259 GB of storage.

Load the packed data with `datasets.load_from_disk`, not `load_dataset`:

```python
from datasets import load_from_disk
train = load_from_disk('/large-disk/Priming/data/imagenet/imagenet-latents-images')
validation = load_from_disk('/large-disk/Priming/data/imagenet/imagenet-latents-images/val')
```

The Arrow shards and their row order are preserved. Matching class indices
and OT calibration are included under `assets/ot/representations/`.
Operational paths were removed from the encoder provenance JSON files;
image bytes, labels, encoder weights, and calibration values are unchanged.
`dataset_manifest.json` identifies every released file by size and SHA256.
Do not reorder rows without rebuilding the class indices and calibration.

Generator checkpoints and evaluation assets are separate from this gated
dataset. Sampling and the released FID/FDr5 evaluation do not require this
training dataset.
