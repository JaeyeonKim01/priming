---
language: en
library_name: pytorch
tags:
  - image-generation
  - imagenet
  - priming-models
---
# Priming Models

ImageNet-256 class-conditional B/L checkpoints after 80/160 epochs of
reconstruction pretraining, before and after 25,000 distributional
post-training updates.

Files under `checkpoints/pretraining/` contain original full pretraining
checkpoints with online and EMA 0.9999 weights. Files directly under
`checkpoints/` contain full post-training checkpoints; evaluation uses their
EMA 0.999 weights. Post-training initializes from the pretraining online weights.
These are original PyTorch training checkpoints, including optimizer state,
not standalone Diffusers pipelines or safetensors exports.

The two JSON manifests identify all eight files with sizes and SHA256 hashes.
See [the code and setup guide](https://github.com/JaeyeonKim01/one_step_priming)
for model definitions, evaluation, training recipes, and reproduction records.

## Evaluation reference assets

`assets/fid_stats/` contains the frozen FDr5 reference statistics and the extra
ConvNeXt statistic used during training diagnostics. The precomputed Inception
precision/recall reference features are under `data/prc/`. The code's
`scripts/download_assets.py` downloads the required subset, pins upstream
evaluation encoders, and fetches the separate official JiT FID reference.
Users evaluating Priming do not need to recompute real-image statistics.

The processed ImageNet training release is in the separate
[dataset repository](https://huggingface.co/datasets/jaykim1102/priming), with
ImageNet access terms and matching class indices/OT calibration. Neither
sampling nor final FID/FDr5 evaluation requires that training dataset.
