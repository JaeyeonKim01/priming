# ImageNet FID setup and latest verified results

Snapshot: evaluation batch `20260925`, completed September 25, 2026 EDT / September 26 UTC. **11 models, 33/33 validated sampling replicates, 1,650,000 generated images.** This is the current reference for the reevaluated ImageNet headline models; older dashboard tables and ablation curves retain their historical protocols.

## Latest results

Each value is the mean ± **sample standard deviation** (`ddof=1`) of seeds 1, 2, 3. Each seed independently generates 50,000 class-balanced ImageNet-256 images (50/class). These are sampling seeds of one fixed checkpoint per model, **not training seeds**. FID is computed separately per seed; no pooled 150k-image FID and no best-seed selection. EFLOPs are previously audited training compute, not evaluation cost.

Source of truth: [full-precision paper CSV](../../final_paper/data/imagenet_three_seed_20260925.csv), [per-seed CSV](../../isolated_experiments/three_seed_eval_20260925/reports/per_seed.csv), [validation report](../../isolated_experiments/three_seed_eval_20260925/reports/validation.json).

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
| Official evaluator checkout | `isolated_experiments/fid_protocol_audit_20260925/vendor/torch-fidelity` |
| Inception weights | `tests/mmd/runs/torch_cache/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth` |
| Real ImageNet-256 reference | `/n/home10/jaykim/woobin/JiT/fid_stats/jit_in256_stats.npz` |
| Reference SHA256 | `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c` |
| Frozen local runtime | `tests/runs/ot_metric_climb_20260904/runtime` |

The [earlier paired protocol audit](../../isolated_experiments/fid_protocol_audit_20260925/reports/RESULTS.md) independently checked this direct-feature path against upstream PNG evaluation. Its single-seed diagnostic values are separate from this completed three-seed table. Changing the reference, image quantization, or feature precision changes the metric; do not treat an older local BF16/ADM FID as interchangeable.

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

Keep `evaluate.py` frozen for WF/Drifting and `evaluate_v2.py` frozen for the other models; completed reports record source hashes. iMF uses the official MSE VAE wrapper with **eager FP32 decode** to avoid a Torch 2.6/diffusers 0.38 compiler error. Its weights and latent mean/std normalization are unchanged; do not replace that normalization with W-Flow's `/0.18215`. VAE revision: `31f26fdeee1355a5c34592e401dd41e45d25a493`.

## Checkpoints, source versions, and environment

Exact per-model checkpoint paths, CFG, and compute provenance are in the [paper CSV](../../final_paper/data/imagenet_three_seed_20260925.csv); full per-seed metadata are in [reports](../../isolated_experiments/three_seed_eval_20260925/reports). Priming `step_0024999.pth` represents 25,000 completed updates. WF uses `state_00180000.pt` EMA.

- pMF official PyTorch branch: `990e81a84249dbd68a128accef67eb95621d10b1`.
- iMF official PyTorch branch: `04687983e821b3ad01f54f03dc44194a33c20c54`.
- JiT clean source archive: `cbc743a2ada5e9762697da2c83f8c4f8379e8c17`.
- Baseline checkpoint mirror: `jjiaweiyang/FD-Loss@88ce2bb4061b0dd58839a1827235ffcffa51eed2`, `checkpoints/base`. pMF-L/H and iMF files were verified byte-identical to original-author releases. [Verified checkpoint hashes](../../isolated_experiments/three_seed_eval_20260925/reports/checkpoint_hashes.json).
- Python executable: `/n/home10/jaykim/.conda/envs/fdloss/bin/python`.
- Python 3.11.15; torch 2.6.0+cu124; torchvision 0.21.0+cu124; NumPy 2.4.4; SciPy 1.17.1; timm 1.0.27; diffusers 0.38.0; transformers 5.12.1. [Full environment snapshot](../../isolated_experiments/three_seed_eval_20260925/reports/environment.json).
- Scheduler: `kempner_h100_priority`, account `kempner_sham_lab`. Ordinary jobs: one node × 4 H100, 200 GB RAM, 24 CPUs. JiT: two nodes × 4 H100 per seed, 200 GB and 24 CPUs per node.
- Wrappers set `PYTHONNOUSERSITE=1`, offline Hugging Face caches under `/n/netscratch/sham_lab/Lab/workdir/huggingface`, and `TORCH_HOME=tests/mmd/runs/torch_cache`. Preserve these paths or explicitly migrate and verify assets.

## Rebuild summaries without GPU sampling

From the repository root, using the recorded environment:

```bash
EVAL_PY=/n/home10/jaykim/.conda/envs/fdloss/bin/python
EVAL_DIR=isolated_experiments/three_seed_eval_20260925
"$EVAL_PY" "$EVAL_DIR/summarize.py"
"$EVAL_PY" "$EVAL_DIR/validate_results.py"
"$EVAL_PY" "$EVAL_DIR/compare_published.py"
"$EVAL_PY" "$EVAL_DIR/export_paper.py"
```

These rebuild generated reports, the paper CSV, and the exported LaTeX table. The validator also reads scratch features/IDs and moments; it requires those retained artifacts. It checks exact ID coverage, 50 images/class, rank counts, distinct seed streams, fixed checkpoints/CFG, evaluator/reference hashes, FDr⁵ aggregation, finite metrics, and scratch/report parity. The completed validation passed all 33 replicates.

## Sampling launch recipes and future reruns

These are **archived launch recipes**, not commands to overwrite the completed batch:

```bash
# WF/Drifting: use evaluate.py via run.sbatch.
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run.sbatch --model wflow_xl --num-images 50000 --seeds 1 2 3
# Replace wflow_xl with drifting_b or drifting_l as needed.

# Priming: evaluate_v2.py, 4 GPUs, batch 32.
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run_v2.sbatch --model priming_l_160 --num-images 50000 --seeds 1 2 3
# Other keys: priming_b_80, priming_b_160, priming_l_80.

# pMF/iMF wrapper: 1k pilot, then 50k for seeds 1/2/3.
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run_baseline.sbatch pmf_l
# Other keys: pmf_h, imf_xl.

# JiT: one 8-GPU job per seed.
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run_jit_8gpu.sbatch 1
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run_jit_8gpu.sbatch 2
sbatch --export=NIL isolated_experiments/three_seed_eval_20260925/run_jit_8gpu.sbatch 3
```

The evaluator intentionally refuses an existing completed result. For a **new batch**, create a new dated experiment directory and new scratch output namespace. Copy the necessary source/vendor snapshots, set the evaluator's hardcoded `OUTPUT` to the new scratch namespace, point the batch wrappers to the copied evaluator, and start with an empty new `reports/` directory. Audit hardcoded paths in validation/export scripts before using them for a new batch; the old paper export destination must not silently become the new batch's destination. Shared, verified checkpoint caches may be reused. There is no `--output-dir` CLI option in the frozen evaluators. Keep the original scripts, reports, and source hashes intact. Run a 1k smoke check first, then the 50k measurements; 1k pilots are excluded from headline summaries.

Retained scratch root: `/n/netscratch/sham_lab/Lab/jaykim/three_seed_eval_20260925/<model>/n50000_seed<seed>/`. It contains rank-level official Inception features, IDs, counts, byte-stream hashes, first-batch image bytes, generated statistics, and results. Full 50k image collections were **not** retained. Checkpoints and full feature arrays are not embedded in this Markdown or the small paper CSV; preserve/migrate those scratch assets if future independent verification is required.

## Comparison to original reports

The following values are transcribed from the completed [source-linked original-paper comparison](../../isolated_experiments/three_seed_eval_20260925/reports/PUBLISHED_COMPARISON.md), not a new literature search.

| Model | Old dashboard | Original paper | Official release | New mean ± SD | New − paper |
|---|---:|---:|---:|---:|---:|
| [wflow_xl](https://github.com/hanjq17/W-Flow) | 1.3477 | 1.29 | — | 1.3275 ± 0.0137 | +0.0375 |
| [drifting_b](https://github.com/lambertae/drifting) | 1.5470 | 1.76 | 1.73 | 1.7387 ± 0.0173 | -0.0213 |
| [drifting_l](https://github.com/lambertae/drifting) | 1.4260 | 1.61 | 1.62 | 1.6149 ± 0.0205 | +0.0049 |
| [pmf_l](https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1) | 2.7200 | 2.52 | 2.51 | 2.5544 ± 0.0152 | +0.0344 |
| [pmf_h](https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1) | 2.2900 | 2.22 | 2.15 | 2.1501 ± 0.0182 | -0.0699 |
| [jit_h](https://arxiv.org/html/2511.13720v1) | 1.9700 | 1.86 | — | 1.8800 ± 0.0177 | +0.0200 |
| [imf_xl](https://github.com/Lyy-iiis/imeanflow/tree/04687983e821b3ad01f54f03dc44194a33c20c54) | 1.8200 | 1.72 | 1.72 | 1.7397 ± 0.0150 | +0.0197 |

Drifting-B/L and pMF-H agree closely with official release results. WF still differs from the paper's 1.29; pMF-L, JiT-H and iMF also retain small differences. Do not claim exact reproduction of every paper or attribute all gaps to sampling seeds. Historical pMF-L 2.72, pMF-H 2.29, JiT-H 1.97 and iMF 1.82 are FD-loss re-evaluation values, not their original-paper results.

A separate WF seed-1 TF32 probe produced 1.31483031 with generation matmul TF32 disabled versus 1.31479429 enabled (difference +0.00003602). That does not explain the residual to 1.29. WF CFG remains **1.09**; no CFG sweep or seed selection was used. The diagnostic is excluded from the 33-run table.

## Paper integration status

- [ImageNet table and quantitative results](../../final_paper/onestep_newer/3_Experiment.tex) use the new means, rounded to two decimals. Priming-L-160: **FID 1.19, FDr⁵ 7.10**. Training EFLOPs are unchanged.
- [Figure 1 FID wide-v2 PDF](../../final_paper/onestep_newer/figures/figure1_fid_panel_v2_wide.pdf) uses the new means for overlapping reevaluated models. Existing **pMF-B and JiT-B points retain historical values** because they were not among the 11 models. The figure retains its original model selection; it does not display all 11 table rows. Source/point metadata: [JSON](../../final_paper/figures/figure1_fid_panel_v2_wide.json). Rebuild with `python final_paper/prepare_intro_v2_wide.py --fid-only` in the recorded environment.
- [Figure 2 qualitative PDF](../../final_paper/onestep_newer/figures/intro_examples_layout_draft.pdf) now says `ImageNet 256 × 256 (FID 1.19)`. Images/layout were preserved; rendered pixels outside the changed digits were identical. Its adjacent JSON preserves exact selections, hashes, and layout.
- [Figure 5 FDr⁵ PDF](../../final_paper/onestep_newer/figures/figure1_fdr5_panel_v2_wide.pdf) now uses the latest means for overlapping reevaluated models and highlights **7.10**, preserving its previous styling and axis ranges. pMF-B and JiT-L retain historical measurements. Rebuild with `python final_paper/prepare_fdr5_wide.py` in the recorded environment.
- Training trajectories, historical dashboard CSVs, and unrelated prose were not globally migrated. In particular, the abstract/introduction still contain older 1.25 prose pending a separate manuscript edit. Do not describe the whole manuscript as updated.
- Exported reusable [LaTeX result table](../../isolated_experiments/three_seed_eval_20260925/reports/paper_table.tex); detailed [evaluation README](../../isolated_experiments/three_seed_eval_20260925/README.md); [job/probe manifest](../../isolated_experiments/three_seed_eval_20260925/reports/launch_manifest.json).

When extending this record, regenerate numbers from the full-precision CSV and preserve the protocol/version distinction. Do not overwrite historical curves with isolated new endpoint values and imply that the entire curve was reevaluated.

## Table 4 ablation reevaluation — September 26, 2026

Completed separately from the 33-replicate headline batch above: **5 models, 15/15 validated sampling replicates, 750,000 generated images**, all on `kempner_h100_priority` (4 H100 GPUs/model). Each model uses its fixed 25,000-update terminal EMA 0.999 checkpoint, CFG 1, batch 32, and seeds 1/2/3 under the same FID and separate FDr⁵ contracts above. Every 50k run followed a validated 1k pilot; pilots are excluded.

Source of truth: [full-precision CSV](../../final_paper/data/table4_three_seed_20260926.csv), [per-seed results](../../isolated_experiments/table4_fid_20260926/reports/per_seed.csv), [validation](../../isolated_experiments/table4_fid_20260926/reports/validation.json), and [independent FID recomputation](../../isolated_experiments/table4_fid_20260926/reports/fid_recomputation.json). Checkpoints and SHA256 hashes: [model manifest](../../isolated_experiments/table4_fid_20260926/models.json).

| Table 4 column | FID ↓, mean ± sample SD | FDr⁵ ↓, mean ± sample SD |
|---|---:|---:|
| Reference | 1.3841 ± 0.0130 | 11.2286 ± 0.0157 |
| w/o perceptual | 2.0693 ± 0.0265 | 16.1266 ± 0.0535 |
| Energy | 1.9490 ± 0.0524 | 11.5825 ± 0.0213 |
| Laplacian | 1.8479 ± 0.0290 | 11.2593 ± 0.0245 |
| SIM | 1.6469 ± 0.0290 | 6.1135 ± 0.0203 |

Reference reproduction: all three seeds produced identical rounded-image byte streams to the September 25 Priming-B-160 evaluation, on every rank. Maximum absolute FID difference was `6.474346037066425e-08`; maximum FDr⁵ difference was `4.734541700912587e-06`. Thus images reproduce exactly and metrics reproduce numerically, rather than bitwise. Independently recomputing all 15 FIDs from saved generated moments with the pinned upstream evaluator matched the recorded FIDs exactly.

[Table 4 and its related paragraphs](../../final_paper/onestep_newer/5_Additional_experiments.tex) now use the new means, rounded to two decimals. **Masking and RAE were not reevaluated** and retain historical values/protocol; this distinction is stated in the manuscript. Historical endpoint CSVs and training curves remain unchanged. This new ablation batch does not replace the September 25 headline CSV or its reports.

Reproduction instructions and job records: [README](../../isolated_experiments/table4_fid_20260926/README.md), [launch manifest](../../isolated_experiments/table4_fid_20260926/reports/launch_manifest.json), [Slurm accounting](../../isolated_experiments/table4_fid_20260926/reports/slurm_accounting.txt). All five jobs completed with exit code 0. Retained scratch: `/n/netscratch/sham_lab/Lab/jaykim/table4_fid_20260926/`.
