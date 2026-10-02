# Reproduction status — 2026-09-30

## Paper headline batch

**Complete, verified from retained artifacts.** The September 25–26 batch
evaluated 11 fixed checkpoints with three independent sampling seeds each:
33/33 runs, 50,000 balanced ImageNet-256 images per run, 1,650,000 generated
images total. The release [read-only validator](eval/verify_results.py)
passed on all 33 copied reports and on the retained scratch IDs, rank
records, byte-stream hashes, and generated moments. Full-precision results
are in [paper_results.csv](eval/paper_results.csv). The
[dashboard](dashboard/index.html) now displays the separate September 30
fresh reproduction.

| Model | Replicates | FID mean ± SD | FDr⁵ mean ± SD |
|---|---:|---:|---:|
| WF-XL/2 | 3/3 | 1.3275 ± 0.0137 | 11.0700 ± 0.0462 |
| Drifting-B/16 | 3/3 | 1.7387 ± 0.0173 | 15.3034 ± 0.0399 |
| Drifting-L/16 | 3/3 | 1.6149 ± 0.0205 | 12.5892 ± 0.0705 |
| pMF-L/16 | 3/3 | 2.5544 ± 0.0152 | 10.8473 ± 0.0182 |
| pMF-H/16 | 3/3 | 2.1501 ± 0.0182 | 8.1446 ± 0.0058 |
| JiT-H/16 | 3/3 | 1.8800 ± 0.0177 | 8.6927 ± 0.0528 |
| iMF-XL/2 | 3/3 | 1.7397 ± 0.0150 | 9.7454 ± 0.0382 |
| Priming-B-80 | 3/3 | 1.4683 ± 0.0128 | 12.7089 ± 0.0188 |
| Priming-B-160 | 3/3 | 1.3841 ± 0.0130 | 11.2285 ± 0.0157 |
| Priming-L-80 | 3/3 | 1.2695 ± 0.0128 | 8.9430 ± 0.0134 |
| Priming-L-160 | 3/3 | 1.1871 ± 0.0147 | 7.1048 ± 0.0196 |

These reproduce the *local paper evaluation protocol*. Agreement with each
baseline's original publication is close but not exact in every case:
WF-XL/2 is 1.3275 versus the paper's 1.29; pMF-L is 2.5544 versus 2.52;
JiT-H is 1.8800 versus 1.86; iMF-XL/2 is 1.7397 versus 1.72. The
[paper record](eval/FID_SETUP_AND_RESULTS.md#comparison-to-original-reports)
lists the remaining comparisons and sampling caveats. No CFG or seed sweep
was used to close these gaps.

Official FID is computed on rounded uint8 images with FP32
`inception-v3-compat` and the JiT ImageNet-256 reference. FDr⁵ is computed
separately on float images with five frozen encoders; its Inception component
uses the legacy ADM reference. ConvNeXt is excluded. Each result is the mean
and sample SD of three *sampling* seeds of one checkpoint, not three
training runs. No pooled 150,000-image FID is reported.

## Four production checkpoints

**Linked and source-verified.** The B-80, B-160, L-80, and L-160 names in
[checkpoints](checkpoints) are relative symlinks to the production full
checkpoints on scratch. Each source has `step=24999` and
`current_step=25000`. All four source SHA256 hashes were checked again after
the links were made. The previous temporary EMA exports were tensor-equal to
the source EMA 0.999 copies, but those large home-directory files were
removed. The evaluator now reads EMA 0.999 directly through the symlinks.

## Release evaluation script

The [archived evaluator](eval/archive/evaluate_v2.py) and reports are the
provenance for the published numbers. The [release evaluator](eval/evaluate.py)
reads the linked production EMA weights and writes to a caller-selected new output
directory. Seven fresh 1,000-image pilots on `kempner_h100_priority` passed
the [new-batch validator](eval/verify_batch.py): Priming-B-80, Priming-L-160,
W-Flow XL, Drifting B, pMF L, iMF XL, and JiT H. This covers every distinct
sampler family in the headline batch. The
[pilot record](eval/pilots/validation_20260930.json) contains job IDs and
exact comparisons. For pMF-L, all four rank-level uint8 image-stream hashes
match the archived 1,000-image pilot, and FID differs by only `4.23e-8`;
FDr⁵ differs by `0.000574`. The first rank-0 uint8 image matches the
published 50,000-image seed-1 run exactly for the two previous Priming pilots,
W-Flow XL, and Drifting B. iMF XL agrees numerically with its archived pilot
(FID difference `3.29e-5`); the first-batch pixel differences are at most one
uint8 level. The archived L-160 **1,000-image** pilot used an older v1 seed
rule, and the archived JiT **1,000-image** pilot used four GPUs/batch 32;
neither full pilot stream is comparable to the current v2 pilot settings.

These are smoke tests, not new headline FIDs. They used the release evaluator
before its Priming branch was changed to read the production symlinks; the
baseline sampler branches were unchanged. Two additional September 30 pilots
using the linked full checkpoints (B-80 and L-160) passed and matched the
previous pilot uint8 streams exactly; see the
[linked-checkpoint pilot record](eval/pilots/link_validation_20260930.json).
The fresh 50,000-image batch is complete below.

Training pipeline cleanup was outside the September 30 evaluation staging pass.
The subsequent OT package is in [training](training/README.md), and the
October 2 pretraining migration is described below.

## Fresh 2026-09-30 full reproduction

**Complete: 33/33 fresh runs validated.** Eleven fixed checkpoints each
produced 50,000 balanced ImageNet-256 images for sampling seeds 1, 2, and 3:
1,650,000 new images. All jobs used `kempner_h100_priority` (four GPUs per
model, except eight per JiT seed), and their IDs are in
[the launch manifest](eval/full_reproduction_20260930_launch.json). The raw
artifacts are separate from the paper batch at
`/n/netscratch/sham_lab/Lab/jaykim/OST_public_full_eval_20260930`.

The [fresh comparison report](eval/full_reproduction_20260930.md) gives all
11 FID and FDr⁵ mean ± sample SD values, deltas from the September 25–26
batch, and byte-stream match counts. Every fresh mean matches the current
manuscript table at its two-decimal precision. The
[full-precision CSV](eval/full_reproduction_20260930.csv),
[per-seed CSV](eval/full_reproduction_20260930_per_seed.csv), and
[validation JSON](eval/full_reproduction_20260930.json) preserve the exact
numbers. All four Priming models reproduce all 48 archived rank-level uint8
streams exactly. Priming-L-160's fresh FID is
`1.187123832371905 ± 0.014710174689390175`, exactly the paper mean/SD;
its FDr⁵ is `7.104809772473572 ± 0.01959242610093468` (mean difference
`+0.0000065729`). W-Flow XL, iMF-XL, and JiT-H have small numerical shifts
and some differing byte streams; their two-decimal table values are unchanged.
For JiT-H, the fresh FID is `1.8797009382464012 ± 0.017704047790844293`
and FDr⁵ is `8.692799446784738 ± 0.05263633755542856`.

Official FID was independently recalculated from the saved generated moments
for the three Drifting-B seeds and JiT seed 1, matching each reported value
exactly. A separate CPU audit of all 33 saved moment files is queued on
`shared`; the complete sampling reproduction and artifact validation do not
depend on that optional second calculation.

On September 30, a separate arithmetic audit recomputed all 11 mean/SD rows
from the 33 saved September 25–26 result files and matched the full-precision
CSV, manuscript LaTeX table, and table above. See
[table_recheck_20260930.json](eval/reports/table_recheck_20260930.json). That
audit checks the earlier saved results; it does not count as new sampling.

## Pretraining code migration — 2026-10-02

[Pretraining](pretraining/README.md) packages B perceptual, B no-perceptual,
and L perceptual recipes with retained runtime sources, asset-path preparation,
torchrun/Slurm entrypoints, and production provenance. The 160-epoch recipes
retain both the 80- and 160-epoch sources used for post-training. No L
no-perceptual config is included.

This is a code/config migration, not a new pretraining reproduction. The
[validation report](pretraining/reports/validation.json) records source/config
identity and CPU checks. B's copied runtime matches its production manifest;
L's original arguments and logs establish its recipe and checkpoint lineage,
while its retained code snapshot comes from the later September study.
The [source record](pretraining/provenance/README.md) documents that distinction
and preserves L's historical restart limitations.

## Portable checkout

The public code now resolves active paths from the `priming/` repository and
includes its evaluation runtime and baseline sampler sources. Current setup
and H100/H200 commands are in [the setup guide](docs/setup.md) and each stage's
README. Historical cluster paths in this report and provenance identify the
original evidence; they are not required by new launches. Fresh launch state
lives in ignored `training/state/`. The portability checks use a newly installed
`priming` environment and a source-only relocated checkout; they do not claim
a new GPU training or sampling reproduction.
