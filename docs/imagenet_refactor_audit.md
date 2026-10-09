# ImageNet refactor audit — 2026-10-09

Comparison: previous release `443db44` → refactor `181ff14`.
The old source was checked out at that exact commit, separately from the active
checkout. The refactored model, training, and evaluation implementation was
left unchanged during this audit.

## Verification

- **55/55 CPU tests passed**, with `PRIMING_REFERENCE_ROOT` set to the previous
  release; no reference-comparison tests were skipped.
- **Source-only portability passed**: nine CLI entrypoints, all eight recipes,
  the numerical/reference suite, asset-helper tests, shell syntax, and shared
  import locations work in a relocated checkout without local assets.
- **12/12 fresh 50,000-image evaluations passed comparison** with the archived
  September 30 results: B/L × 80/160 × seeds 1/2/3, 600,000 images total. All
  48 rank-level image-byte hashes and official Inception feature arrays match.
- **Paired B-160/L-160 pilots passed** against the exact old checkout on the
  same four H200 GPUs, with 1,000 images/model and seed 1. All eight image-byte
  streams match. The largest pilot FID/FDr⁵ deltas are 2.23e-8 / 4.83e-5.
- **Eight-H100 training checks:** B/L perceptual pretraining and B post-training
  from both released and freshly trained checkpoints passed two updates and
  saved-state audits. **L post-training exhausted 80 GB GPU memory** before
  its first update completed. See the training table below.
- **Eight-H200 follow-up:** L post-training from both released and fresh custom
  pretraining checkpoints, plus B no-perceptual pretraining, passed two updates
  and saved-state audits on all eight ranks. H200 matches the original L
  training hardware.
- **Environment and public-source checks passed.** No full training reproduction
  is claimed.

The suite checks full B/L checkpoint tensor schemas, compact-model outputs and
reconstruction gradients, corruption/augmentation and sampling streams,
perceptual preprocessing, Muon/AdamW behavior, EMA, the five post-training
recipes, OT/energy/Laplacian losses and gradients, feature identities,
checkpoint continuation, scheduled evaluation, and standalone metric arithmetic.
Heavyweight encoders are partly replaced by compact models or synthetic weights
in CPU tests; those tests alone cannot validate released GPU inference.

To rerun the CPU/reference and portability checks from the repository root:

```bash
git worktree add --detach ../Priming-reference 443db44
PRIMING_REFERENCE_ROOT="$(cd ../Priming-reference && pwd)" \
  PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -q
python scripts/check_portability.py --reference-root ../Priming-reference
```

## Fresh released-checkpoint results

Each result is the mean ± sample SD over three sampling seeds of one fixed
checkpoint. Four H200 GPUs, batch 32, BF16 generators/features, FP32 official
Inception, and the same pinned encoder/reference assets were used.

| Model | FID mean ± SD | FDr⁵ mean ± SD | FDr⁵ mean change |
|---|---:|---:|---:|
| B-80 | 1.4682677920 ± 0.0127924082 | 12.7089285902 ± 0.0187658128 | +0.0000150821 |
| B-160 | 1.3841025724 ± 0.0129825787 | 11.2285526433 ± 0.0156987357 | +0.0000019318 |
| L-80 | 1.2694541504 ± 0.0127701986 | 8.9429773675 ± 0.0134049844 | −0.0000001573 |
| L-160 | 1.1871238324 ± 0.0147101747 | 7.1048147334 ± 0.0195924928 | +0.0000049609 |

All four FID/FDr⁵ means retain the paper's two-decimal values. Maximum per-seed
absolute changes are **1.71e-13 FID** and **3.14e-5 FDr⁵**. The generated images,
index coverage, official feature arrays, and official generated moments are
identical. Checkpoint, inference-config, and official-reference hashes match.
FDr⁵ is numerically close, not bitwise identical; the per-encoder differences
are preserved in the audit record.

[Full-precision CSV](imagenet_refactor_results.csv) ·
[Machine-readable comparisons and identities](imagenet_refactor_audit.json)

The full runs used the refactored evaluator without modifying its implementation.
The comparison checks every rank's count, IDs, image-byte hash, first-batch pixels,
full official feature array, generated mean/covariance, and each metric component.
The paired small pilots isolate old/new implementations on the same hardware;
the full runs compare against the original retained production artifacts.

## Eight-GPU training checks

The initial checks used eight H100 80 GB GPUs, real packed ImageNet, pinned released
encoder weights, compilation enabled, BF16 generator execution, and the
unchanged global batch of 1,024. B perceptual pretraining used two nodes with
four GPUs each; the subsequent checks used four nodes with two GPUs each.
Only storage paths, output directories, and the two-update stopping limit
were overridden. Custom B initialization used the checkpoint just produced
by this refactored pretraining run. The follow-up uses eight H200s across four
nodes, with the same two-update limit and scientific settings. Custom L
initialization uses the L checkpoint produced by the H100 pretraining check.

| Route | GPU type (eight ranks) | Result | Peak PyTorch allocated memory / GPU |
|---|---|---|---:|
| B perceptual pretraining | H100 80 GB | Passed two updates | 35.71 GiB |
| L perceptual pretraining | H100 80 GB | Passed two updates | 59.78 GiB |
| Released B-160 → B post-training | H100 80 GB | Passed two updates | 47.88 GiB |
| Fresh custom B → B post-training | H100 80 GB | Passed two updates | 47.88 GiB |
| Released L-160 → L post-training | H100 80 GB | CUDA out of memory before completing update 1 | About 77.34 GiB allocated at failure |
| Released L-160 → L post-training | H200 | Passed two updates | 81.32 GiB |
| Fresh custom L → L post-training | H200 | Passed two updates | 81.32 GiB |
| B no-perceptual pretraining | H200 | Passed two updates | 21.10 GiB |

On H200, both released L-160 and fresh custom L initialization subsequently
completed two post-training updates on all eight ranks. Each peaked at
81.3196 GiB allocated / 85.1211 GiB reserved per GPU. These measurements are
from short startup runs; they do not establish a full-training memory bound.

All eight ranks completed each passing case. Saved online/EMA/optimizer tensors
were finite, counters recorded 2,048 samples, and post-training AdamW states
recorded two updates. All four B/L post-training routes inherited the source EMA
counter and changed online weights. The first warmup update has zero learning
rate; the second establishes a nonzero parameter update. B pretraining saved
all eight optimizer shards; L retained its historical rank-zero save behavior.
Detailed per-rank telemetry and saved-state checks are in the
[machine-readable audit](imagenet_refactor_audit.json).

The L failure occurred while constructing the self-reference feature cost in
`losses/discrepancy.py`: an additional 80 MiB allocation failed with only about
43 MiB free. This is a capacity limit with this hardware/recipe, not a checkpoint
format rejection. It does **not** establish a regression from the old runtime:
an equivalent old/new eight-H100 training comparison has not been run. No model,
batch, population, loss, or precision setting was changed to make it fit.
The original released L-80 and L-160 production runs both used **eight H200
GPUs**. Their retained `memory_audit.json` files record all eight ranks at
25,000 completed updates, with maximum allocated memory of 85.0368 and
85.0370 GiB, respectively. Per-rank records and source-file hashes are retained
in this audit's JSON. The previous release's training guide likewise specified
eight H200s and reported 85.01 GiB allocated / 87.58 GiB reserved for its L
startup check. Eight 80 GB GPUs are therefore below the observed memory
requirement of both the historical execution and this refactored execution.
The follow-up audit passed on H200s with unchanged scientific settings; reducing
memory for H100s is a separate optimization, not a prerequisite for matching
the original hardware.

These are bounded startup, optimizer, and serialization checks, not convergence
tests. They do not cover scheduled 50,000-image training evaluation or GPU
resume equivalence. Custom initialization requires the full compatible
checkpoint format described in [post-training](posttrain.md#initialization-and-resume).

## Scientific settings retained

| Component | Previous release and refactor |
|---|---|
| Generator | Same B/L tensor names and shapes; released checkpoints load strictly |
| Pretraining | Fixed t=0.8, adaptive reconstruction, optional perceptual losses, Muon, EMA 0.9999; B/L-specific accumulation and seed behavior |
| OT post-training | 25,000 updates, eight ranks, global batch 1,024, 16 classes/update, 64 live + 64 independent generated references and 128 real references/class |
| Optimizer / EMA | Fresh AdamW, LR 1e-5, 125-update zero-based warmup; inherit pretrained EMA and continue at decay 0.999 |
| Official FID | FP32 Inception on rounded uint8 images, pinned JiT reference, original covariance/eigenspectrum arithmetic |
| FDr⁵ | Float-image BF16 encoder forwards, FP64 moments, original five representations and normalization; ADM reference for its Inception component |
| Paper draws | Four ranks, batch 32, seeds 1/2/3, balanced class blocks, padded final generation batch, rank seed `seed*1000003+rank` |

The settings, CPU comparisons, and released-checkpoint evaluations support
numerical compatibility. They do not prove equality of compiled GPU training
trajectories from random initialization.

## Major changes to be aware of

1. **Entry points and configuration layout changed.** The stage-specific
   runtime copies and launch wrappers are replaced by `pretrain.py`,
   `posttrain.py`, `eval.py`, and shared modules. Recipes now inherit adjacent
   `_base.yaml` files. Old shell commands need updating.
2. **Historical training resume is narrower than weight compatibility.**
   Released weights remain loadable for inference and fresh post-training
   initialization. Exact B-pretraining recovery requires the new config and
   manifest contract; new post-training recovery requires `posttrain_config`
   and `posttrain_ema_origin`, which historical production states lack.
   Historical L-pretraining loading remains supported with its previous
   rank-zero optimizer limitation. Preserve the old checkout for ongoing old
   training runs.
3. **Post-training filenames use completed updates.** The new terminal name is
   `step_025000.pth`, instead of `step_0024999.pth`; internal terminal counters
   remain `step=24999`, `current_step=25000`. `latest.json` selects the latest
   post-training save; pretraining still uses `latest.pth`.
4. **Evaluation scope changed.** The active evaluator supports the four Priming
   models; comparison-baseline samplers, the old result validators, and the
   dashboard were removed. Pretraining and online-weight evaluation were added.
   Those modes are distinct from the paper's post-training EMA results.
5. **Old reports are historical evidence.** The September 30 full reproduction
   predates this refactor. Its 33-run success alone does not validate `181ff14`. This audit adds 12 new
   Priming evaluations; it does not rerun the removed baseline samplers.
6. **`old/` remains an active dependency.** Environment pins, dataset/encoder
   manifests, and inference configurations moved there; do not remove it.

## Remaining reproduction scope

Released post-training checkpoint evaluation is complete for all four Priming
models and three seeds. Full 160-epoch pretraining, 25,000-update post-training,
and GPU trajectory/recovery equivalence remain distinct work; neither CPU
checks nor bounded training startup tests establish those full trajectories.
