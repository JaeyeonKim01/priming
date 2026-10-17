# Faster post-training and reproduction

The recommended `b160_fast.yaml` and `l160_fast.yaml` recipes accelerate frozen
encoder execution. Both completed 25,000 post-training updates from the released
160-epoch pretrained online weights, followed by the official three-seed
FID/FDr⁵ evaluation. Pretraining and inference use the existing implementation.

## Versions

- `main` recommends the fast B/L OT recipes. The original `b160.yaml` and
  `l160.yaml` remain available with their existing execution settings.
- [paper-reproduction](https://github.com/JaeyeonKim01/priming/tree/paper-reproduction)
  preserves the original public tree at `94fcb43c802a8f543f28b85385d9aea8e3063ea9`.
  The annotated [v1.0-paper](https://github.com/JaeyeonKim01/priming/tree/v1.0-paper)
  tag identifies the same commit for reproducibility.
- The validated optimized implementation is
  [`b91f408`](https://github.com/JaeyeonKim01/priming/commit/b91f408).
  Release documentation adds no changes to that runtime implementation.

Existing checkpoint downloads and the README's paper-results table continue to
refer to the released checkpoints. The new training results below are separate
checkpoints, identified by SHA-256 in the accompanying evidence.

## Execution changes

| Setting | Original | Fast |
|---|---|---|
| Frozen encoder microbatch | 8 | 32 |
| MAE convolution weight storage | Original storage; BF16 forward execution | BF16, channels-last, converted once after loading |
| MAE normalization parameters | FP32 | FP32 |
| ConvNeXt execution precision | FP32 | FP32 |
| Frozen encoder compilation | Eager | `torch.compile(fullgraph=True, dynamic=False, options={'layout_optimization': False})` |

Compilation covers family feature extraction and input-gradient replay. Profiling
dispatch modes use eager execution without disabling later compiled calls. The
generator was already compiled in the original recipe.

The objective, calibrated feature weights/bandwidths, Sinkhorn settings, class
populations, global batch 1,024, optimizer, EMA, training horizon, and evaluation
protocol are preserved. Original encoder assets and calibration are still
verified before configuring execution. SIM and the energy/Laplacian ablation
recipes retain their existing defaults; the full-run results here cover B/L OT.

These settings change floating-point rounding and are not bit-identical to the
original execution. Fast checkpoints record their execution settings and reject
incompatible resumes. Continue a run with its original recipe, code, and output
directory; start a new output directory when changing execution settings.

## Performance

Measurements use eight GPUs across two nodes, with the same hardware, CPU
allocation, and data storage for each model's original and fast runs. All runs
use PyTorch 2.6.0+cu124. Each benchmark performs 130 updates. Times below are
medians over updates 100–130, taking the slowest rank's time for each update.

| Model | GPUs | Original seconds/update | Fast seconds/update | Speedup | Peak allocated GiB/GPU, original → fast |
|---|---|---:|---:|---:|---:|
| B-160 | 8 × H100 80 GB | 5.649 | 1.830 | **3.09×** | 47.87 → 43.91 |
| L-160 | 8 × H200 | 5.641 | 1.918 | **2.94×** | 81.32 → 77.37 |

These are warmed training-update timings, excluding scheduled evaluation,
checkpoint writing, and startup. Compilation adds startup cost, including after
resume. The complete fast jobs, including scheduled evaluations, took 13h 34m
for B and 13h 54m for L; no matched original full-job wall-time comparison is
claimed. Keep using H200s for L training: its fast benchmark reserved 79.52 GiB
per GPU, and full L training was validated on H200s.

An intermediate configuration with microbatch 32 and BF16/channels-last MAE
weights, but without encoder compilation, took 2.805 s/update for B and 2.697
s/update for L (2.01× and 2.09×). That experiment combines microbatch and weight
layout changes; it does not isolate their individual contributions.

Run the [bounded benchmark](posttrain.md#faster-encoder-execution) on your own
hardware to compare performance.

## Final quality

Each entry evaluates one terminal EMA 0.999 checkpoint with 50,000 images per
sampling seed (1, 2, 3), four ranks, and batch 32. New-run values are mean ±
sample standard deviation. Three sampling seeds are not three training runs.
Both the released and new checkpoints use the same [official evaluator](eval.md).

| Model | Released FID ↓ | New FID ↓ | Released FDr⁵ ↓ | New FDr⁵ ↓ |
|---|---:|---:|---:|---:|
| B-160 | 1.3841 | **1.3837 ± 0.0103** | 11.2286 | 11.2937 ± 0.0200 |
| L-160 | 1.1871 | **1.1810 ± 0.0179** | 7.1048 | 7.1859 ± 0.0247 |

Final FID closely matches the released results. FDr⁵ is slightly higher
(+0.0652 for B, +0.0811 for L). These measurements support comparable final
quality with faster updates; they do not establish identical trajectories or
statistical equivalence across independent training runs.

## Validation and evidence

The bounded comparisons preserved main-noise draws and checkpoint RNG/stream
states on all eight ranks. First parameter-gradient relative L2 differences
were 0.000905 for B and 0.001459 for L. After 130 updates, parameter relative L2
differences were 0.000245 and 0.000462. A separate fixed-input comparison found
image-gradient relative error against FP32 of about 0.02244 for both original
and fast execution.

Resume validation restored model, optimizer, EMA, RNG, and stream state exactly
on all eight ranks, then completed ten further updates. This restoration check
does not assert bit-identical continuation against a separate uninterrupted GPU
run. Both full training jobs completed all 25,000 updates and 20 scheduled
evaluations. All 53 recorded runtime file hashes match the validated code.

[Machine-readable evidence](results/posttrain-performance-20261017.json) includes
benchmark measurements, per-seed final metrics, checkpoint hashes, runtime and
evaluation source hashes, and numerical/recovery checks. Infrastructure paths
and job identifiers are omitted. The
[release validation record](results/posttrain-release-validation-20261017.json)
records passing CLI/configuration, CPU regression/reference-comparison, asset
helper, shell syntax, and source-relocation checks. Follow
[validation](validation.md) to run these checks and GPU validation yourself.
