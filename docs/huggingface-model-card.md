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
