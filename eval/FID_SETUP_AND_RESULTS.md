# ImageNet FID setup and latest verified results

Snapshot: evaluation batch `20260925`, completed September 25, 2026 EDT / September 26 UTC. **11 models, 33/33 validated sampling replicates, 1,650,000 generated images.** This is the current reference for the reevaluated ImageNet headline models; older dashboard tables and ablation curves retain their historical protocols.

## Latest results

Each value is the mean ± **sample standard deviation** (`ddof=1`) of seeds 1, 2, 3. Each seed independently generates 50,000 class-balanced ImageNet-256 images (50/class). These are sampling seeds of one fixed checkpoint per model, **not training seeds**. FID is computed separately per seed; no pooled 150k-image FID and no best-seed selection. EFLOPs are previously audited training compute, not evaluation cost.

Source of truth: [full-precision paper CSV](paper_results.csv), [per-seed CSV](reports/per_seed.csv), [validation report](reports/validation.json).

| Model | FID ↓, mean ± SD | FDr⁵ ↓, mean ± SD | Training EFLOPs |
|---|---:|---:|---:|
| WF-XL/2 | 1.3275 ± 0.0137 | 11.0700 ± 0.0462 | 3744.43 |
| Drifting-B/16 | 1.7387 ± 0.0173 | 15.3034 ± 0.0399 | 1048.23 |
| Drifting-L/16 | 1.6149 ± 0.0205 | 12.5892 ± 0.0705 | 1476.39 |
| pMF-L/16 | 2.5544 ± 0.0152 | 10.8473 ± 0.0182 | 821.75 |
| pMF-H/16 | 2.1501 ± 0.0182 | 8.1446 ± 0.0058 | 1968.54 |
| JiT-H/16 | 1.8800 ± 0.0177 | 8.6927 ± 0.0528 | 839.00 |
| iMF-XL/2 | 1.7397 ± 0.0150 | 9.7454 ± 0.0382 | 3201.12 |
| Priming-B-80 | 1.4683 ± 0.0128 | 12.7089 ± 0.0188 | 85.75 |
| Priming-B-160 | 1.3841 ± 0.0130 | 11.2285 ± 0.0157 | 124.18 |
| Priming-L-80 | 1.2695 ± 0.0128 | 8.9430 ± 0.0134 | 151.29 |
| Priming-L-160 | 1.1871 ± 0.0147 | 7.1048 ± 0.0196 | 238.95 |

## FID contract to reuse

1. Generate 256×256 RGB images with each model's fixed checkpoint and published sampling settings below. Keep sample IDs 0–49,999 and exactly 50 images per class.
2. Convert sampler output to float and clamp to `[0,1]`; quantize with **`(images * 255).round().to(torch.uint8)`**. Do not truncate, evaluate unquantized floats, or add a different resize/crop pipeline.
3. Extract 2048-dimensional `inception-v3-compat` features using pinned **LTH14/torch-fidelity**, commit `bfe72b995473e4fee9351705d7b005f17442a4ca`. Feature extraction is **FP32**, autocast disabled, CUDA matmul TF32 and cuDNN TF32 both disabled.
4. Use upstream `fid_features_to_statistics` and `fid_statistics_to_metric` for generated mean/covariance and Fréchet distance. Match the real reference below exactly.
5. Report the arithmetic mean and sample SD across three separate 50k FIDs. Retain full precision in machine-readable artifacts and round only when presenting results.

| Dependency | Frozen location / identity |
|---|---|
| Official evaluator checkout | `eval/vendor/torch-fidelity` |
| Inception weights | `assets/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth` |
| Real ImageNet-256 reference | `assets/fid_stats/jit_in256_stats.npz` |
| Reference SHA256 | `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c` |
| Frozen local runtime | `eval/runtime` |

The earlier paired protocol audit (recorded in the historical archive) independently checked this direct-feature path against upstream PNG evaluation. Its single-seed diagnostic values are separate from this completed three-seed table. Changing the reference, image quantization, or feature precision changes the metric; do not treat an older local BF16/ADM FID as interchangeable.

## FDr⁵ is a separate metric

The same generated float images feed the frozen local DistMatch evaluator **before uint8 quantization**: BF16 feature extraction, FP64 moments, legacy ADM Inception reference, and no ConvNeXt. The composite is the arithmetic mean of five normalized distances:

`FDr5 = (FD_Inception/1.68 + FD_DINOv2/14.19 + FD_MAE/0.04 + FD_SigLIP/0.60 + FD_CLIP/5.60) / 5`.

**Do not substitute the new official FID into the Inception component of FDr⁵.** Raw distances and normalized components are preserved in each result JSON. Precision/recall and older ablations were not reevaluated in this 33-run batch.

## Sampling settings and reproducibility

| Model / evaluator key | CFG | Guidance interval | Sampling details |
|---|---:|---|---|
| WF-XL/2 / `wflow_xl` | **1.09** | embedded condition | Official DitGen, one step; paper `w=0.09` maps to condition `1+w=1.09`; SD-VAE decode |
| Drifting-B/L / `drifting_b`, `drifting_l` | 1.0 | embedded condition | One-step pixel generator; converted original JAX EMA weights in validated DitGen port, **not native JAX** |
| pMF-L / `pmf_l` | 7.0 | [0.2, 0.7] | Original one-step sampler; noise scale 1 |
| pMF-H / `pmf_h` | 7.0 | [0.2, 0.6] | Original one-step sampler; noise scale 2 |
| iMF-XL/2 / `imf_xl` | 8.0 | [0.42, 0.62] | Original one-step sampler and original VAE normalization |
| JiT-H / `jit_h` | 2.2 | [0.1, 1.0] | Original 50-step Heun, final Euler; noise scale 1, `t_eps=0.05`, `model_ema1` |
| Priming B/L, 80/160 / `priming_{b,l}_{80,160}` | 1.0 | none | Frozen one-step sampler; 25,000 OT updates; EMA 0.999; BF16 |

JiT's actual guided network-call count is **198** (49 Heun steps × 4 + final Euler × 2). Original-paper/table convention says 200 NFEs; the machine-readable CSV records 198. The FD-loss one-step JiT preset is for the post-trained model and must not be used for this baseline.

Generation permits CUDA matmul TF32 and cuDNN TF32, with cuDNN benchmarking enabled. Model-native BF16/FP32 behavior is retained; pMF/iMF use their official FP32/TF32 sampling paths, JiT BF16. These generation settings are distinct from the strict FP32/no-TF32 FID extractor. This is a common runtime, not bitwise reproduction of each paper's environment.

Seed streams:

- WF/Drifting: upstream batch-level `seed XOR batch_start`; batch 32. Batch starts are multiples of 32, keeping seeds 1/2/3 disjoint.
- pMF/iMF: original CPU BatchGenerator with `1000003 * seed + sample_index`. Small-seed XOR would mostly permute the same noises and was deliberately avoided.
- Priming/JiT: `1000003 * seed + rank`. For matching draws, preserve world size and batch: **4 GPUs, batch 32** for all except **JiT: 8 GPUs, batch 64**.

The retained `archive/evaluate.py` and `archive/evaluate_v2.py` are frozen; completed reports record source hashes. iMF uses the official MSE VAE wrapper with **eager FP32 decode** to avoid a Torch 2.6/diffusers 0.38 compiler error. Its weights and latent mean/std normalization are unchanged; do not replace that normalization with W-Flow's `/0.18215`. VAE revision: `31f26fdeee1355a5c34592e401dd41e45d25a493`.

## Portable execution

Use [README.md](README.md) for current commands and [the asset guide](../docs/assets.md)
for repository-relative paths. The retained numbers above describe the September
25 batch; [the September 30 reproduction](full_reproduction_20260930.md) is also
included. The [original report](archive/FID_SETUP_AND_RESULTS_20260925.md) retains
historical cluster paths and paper-build commands as provenance, not operational
instructions for a new checkout.
