# Validation results

Fresh full training and evaluation completed for RT-1 and Bridge-V2 on October
21, 2026. Both used eight H100s, 64 pretraining epochs and 4,000 post-training
updates initialized from their own new pretraining EMA checkpoints.

## Fresh training results

The baseline below is the historical checkpoint re-evaluated with the migrated
evaluator. This avoids ambiguities from rounding the paper's reported values.

| Metric | RT-1 historical | RT-1 fresh | Bridge historical | Bridge fresh |
| --- | ---: | ---: | ---: | ---: |
| SSIM ↑ | 0.878323 | 0.877785 | 0.857248 | 0.856240 |
| PSNR ↑ | 26.764670 | 26.738429 | 24.548739 | 24.462492 |
| LPIPS ↓ | 0.062409 | 0.062798 | 0.067280 | 0.067904 |
| FID ↓ | 3.160984 | 3.288167 | 2.821531 | 2.758949 |
| FVD ↓ | 30.955012 | 32.812995 | 32.571145 | 33.218214 |

RT-1 FID/FVD increased by 4.02% / 6.00%. Bridge FID decreased by 2.22%, while
FVD increased by 1.99%. The full pipeline is reproduced, but exact numerical
paper reproduction is not established. These single runs do not establish
statistical significance or isolate the causes of the differences.

Full evaluation matched the historical protocol, episode ordering, frame counts
and FVD clip selection across eight shards:

| Task | Validation trajectories | Generated frames | FVD clips |
| --- | ---: | ---: | ---: |
| RT-1 | 2,165 | 86,135 | 2,048 |
| Bridge | 6,871 | 228,693 | 2,048 |

Evaluation uses EMA, one-step prediction, eight-frame autoregressive chunks,
VAE-reconstructed targets, SSIM/PSNR/LPIPS, pytorch-fid FID and StyleGAN-V I3D FVD.

| Task | Pretraining | Post-training | Total including evaluation | H100 GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| RT-1 | 4.00 h | 3.09 h | 7 h 17 m | 58.22 |
| Bridge | 3.07 h | 3.36 h | 6 h 52 m | 54.88 |

Training times include checkpoint writes. Total allocation time also includes
startup, checkpoint audits and evaluation. Peak allocated memory was
10.92/8.66 GiB per GPU for RT-1 pretraining/post-training and 11.20/8.95 GiB for
Bridge. Resource use will depend on hardware and storage throughput.

Evidence: [RT-1 report](results/production-rt1-20261021.json),
[Bridge report](results/production-bridge-20261021.json),
[comparison and checkpoint audit](results/production-comparison-audit-20261021.json).

## Validation coverage

- All 26 CPU tests pass: reconstruction, sampling, replay gradients, EMA
  initialization, checkpoint recovery, evaluation conventions and production gates.
- Source parity checks cover predictions, reconstruction gradients, AdamW and EMA.
  Historical-evaluator subset checks matched scalar metrics and saved features
  for 16 RT-1 and 24 Bridge trajectories.
- The eight-GPU pilot verified fresh pretraining, separate-process resume,
  post-training from the new pretraining EMA, post-training resume and subset
  evaluation. A separate stop/recovery probe verified coordinated checkpointing.
- Full production audits verified final update counts (84,608 / 53,184 pretraining;
  4,000 post-training), finite model/EMA/optimizer tensors, optimizer counters,
  eight rank RNG states, complete metric histories, parent checkpoint hashes,
  frozen input provenance and evaluation checkpoint identities.

Supporting results: [source parity](results/source-parity.json),
[feature parity](results/feature-parity.json),
[evaluator parity](results/eval-subset-parity-20261020.json),
[eight-GPU pipeline](results/training-pipeline-report-20261020.json),
[recovery probe](results/training-recovery-probe-20261020.json).

## Reproduction limits

DistMatch is the recovered version 0.6.0; historical checkpoints record 0.4.0.
Exact historical dependency equivalence has not been established. H100 training
uses activation recomputation to fit the native matching population in memory.
Feature values matched the retained-activation path, with small BF16 gradient
differences documented in the feature audit. These differences are possible
contributors to run variation, not demonstrated explanations.

Resume restores optimizer, EMA and rank RNG states, but loader worker/prefetch
state is not serialized, so stochastic continuation is not bit-exact.

Historical-checkpoint evaluation closely matches archived results. Two paper
rounding discrepancies remain: archived RT-1 LPIPS is approximately 0.062409,
while the paper reports 0.063; Bridge FVD re-evaluation rounds to 32.57 versus
32.58 in the paper. See the [historical evaluation comparison](results/eval-full-comparison-20261020.json).
