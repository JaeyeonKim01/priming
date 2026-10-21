# World-modeling migration assessment

Source assessment dated October 20, 2026; status updated October 21. The
implementation lives entirely under `world_modeling/`. Migration, eight-GPU
pipeline validation, and fresh full training/evaluation are complete for both
datasets. See [the validation report](validation.md) for measured results and
remaining numerical differences. Historical assessment details below describe
the migration decisions, rather than current job status.
The source is `/n/netscratch/sitanc_lab/Lab/jaykim/t2i_and_world.tar.gz`, specifically
its `driftworld/rt1_driftworld` and `driftworld/bridge_driftworld` trees.
The paper references are [Appendix C.3](https://arxiv.org/html/2610.20331v1#A3.SS3)
and [Appendix B](https://arxiv.org/html/2610.20331v1#A2) for matching and scaling.
Recipe details below were also checked against source configs and saved checkpoint metadata.

The recommended first release covers RT-1 and Bridge-V2 through **pretraining →
post-training → evaluation**. The archive also contains other DriftWorld tasks,
drifting/self-forcing baselines, and T2I code; they are separate migration work.

## What the source actually implements

Each example has five consecutive RGB observations and aligned actions. Four
observations and four actions condition prediction of the fifth observation.
The last conditioning action describes the transition into that target.
Training samples a random contiguous clip from each selected episode, rather
than enumerating every possible window.

Frames are resized to 256×256. The frozen SD3 VAE samples its posterior and
produces 16×32×32 latents. The source applies the VAE scaling factor at encoding
and reverses it at decoding; it does not apply a shift factor. Preserve this
convention when loading historical weights. Despite its `dataset_precomputed.py`
name, the dataset reads MP4/NPZ pairs and encodes pixels during training.

The generator takes a noisy target latent, four history latents concatenated
along channels, and four actions. It predicts a clean target latent and has no
timestep input. Both U-Nets use widths `[160,320,640,640]`, two residual blocks
per level, and attention at the last two levels. RT-1 has 10-dimensional actions;
Bridge has 7-dimensional actions and additionally uses action-token cross-attention.
These need separate architecture variants, not just different action widths.
The historical CFG conditioner remains in state dictionaries but its projection
is frozen at zero for these recipes; sampling uses scale 1.

### Pretraining

The relevant entry is `main_train_flow.py` → `train_flow.py` →
`flow_matching_denoiser.py`, with `*_flow.yaml`. The files named
`main_train.py`, `train.py`, and `*_phase1/phase2.yaml` implement the DriftWorld
baseline rather than the Priming stages.

The source draws `t = sigmoid(N(0.8, 0.8²))` and forms
`z_t = (1-t) x + t ε`, with unit Gaussian noise. Its velocity-MSE code is
algebraically weighted clean-latent reconstruction:

```text
mean_i [ MSE(f(z_t, history, actions), x)_i / max(t_i, 0.05)^2 ]
```

This accounts for the paper's description as reweighted logit-normal
reconstruction. Preserve the source's sampling, loss scale, and denominator
clamp; changing to a normalized resampled density need not preserve updates.
The same clean predictor supports multi-step flow sampling for comparison and
direct one-step prediction after post-training.

| Setting | RT-1 and Bridge-V2 |
|---|---|
| Duration | 64 epochs |
| Batch | 8 examples/rank × 8 ranks; final loader batch may be smaller |
| Optimizer | AdamW, LR 1e-4, betas (0.9, 0.95), weight decay 0 |
| Warmup / gradient clip | 0 / 2 |
| Precision / EMA | BF16 generator / constant 0.9999 |
| Saved epoch milestones | 1, 2, 4, 8, 16, 32, 64 |

### Post-training

The relevant entry is `main_train_distmatch.py` → `train_distmatch.py`, with
`distmatch_generator.py`, `distmatch_replay.py`, and `*_flow_distmatch.yaml`.
Bridge already delegates to RT-1's shared training loop.

Initialize the online generator from the **pretraining EMA**, then create a
fresh AdamW optimizer and a fresh EMA copied from that generator. This differs
from ImageNet's current online-weight initialization and inherited EMA shadow.

Each rank draws four conditions. For each condition it generates 16 live samples
and 16 independent detached reference samples, with one real target. Across
eight ranks this is 32 conditions, 512 live samples, 512 reference samples, and
32 targets per update. The host asks the matcher for the cross term minus the
detached generated-reference term. Only the live population receives gradients.

Features are unpooled DINOv3-B/16 tokens from block indices `[2,5,8]`, plus
spatial latent tokens. DINO is applied after differentiable VAE decoding. Both
feature networks have frozen weights, but their input gradients must remain
enabled. The latent branch is explicitly `[N,H*W,C]`. The source requests
position-aware conditional matching with `pair_rms` branch scaling and weight
1 per branch. The recovered DistMatch source confirms that `[C,B,P,D]` becomes
`[C*P,B,D]`: it solves a separate problem for each condition and position,
averages within each branch, and sums the four branches without dividing by four.

The exact squared branch scale is the detached value
`max(0.5 * mean_cross_squared_distance + 0.5 * mean_self_squared_distance, 1e-12)`.
Cross/self means cover all sample pairs within aligned groups, then all groups.
The archived trainer does not pass a `distributed` option, so DistMatch 0.6.0
uses `distributed="local"`: each rank computes its scale over four conditions
and DDP averages generator gradients. The paper's minibatch notation does not
explicitly resolve this cross-rank scaling choice. Preserve the archived behavior
and specify it explicitly in migrated configs; `distributed="global"` changes it.

The recovered package returns an entropy-inclusive OT scalar, whereas the paper
writes a cost-only detached-plan surrogate. The KL term is detached, so its
presence changes the reported scalar but not the gradient for the same inputs,
scales, and plans. A synthetic check verified identical input gradients exactly.
With one real target, the cross plan is uniform and requires zero Sinkhorn
iterations; the 16×16 generated-reference plan uses the ten-iteration budget.

| Setting | RT-1 and Bridge-V2 |
|---|---|
| Updates | 4,000 |
| Sinkhorn | epsilon 0.05, 10 iterations, tolerance 1e-6, strict convergence off |
| Optimizer | AdamW, LR 1e-5, betas (0.9, 0.95), weight decay 0 |
| Warmup / gradient clip | 0 / 1 |
| EMA | constant 0.999 |
| Memory handling | generator replay microbatch 8; VAE feature decode chunk 1 |

Replay first generates both complete populations without a generator graph,
computes the full matching loss and its latent gradient, then replays the live
generator in microbatches with the saved random states. It does not independently
solve matching inside each microbatch or divide the cached gradient again.

The migrated H100 runtime also checkpoints frozen VAE decoder chunks and DINO
activations. The unchanged source activation-retention path exceeded 80 GB at
the native four-condition, sixteen-sample batch. Recomputing those activations
preserves the complete matching population and objective; the GPU feature audit
compares forward values and latent gradients against the unchanged source.
The audit found exact feature values in both precisions; maximum FP32 input-gradient
error was `2.29e-7` (relative L2 `7.41e-6`). BF16 input gradients had relative L2
error `0.003997` and cosine `0.999992`. Activation checkpointing therefore preserves
the objective but does not provide bit-exact mixed-precision gradients.

### Evaluation

Use `eval/eval_on_validation_streaming_shard.py` and its aggregate module as the
reference. Bridge delegates to these shared implementations. Preserve:

- EMA weights, one network evaluation per frame, and no CFG amplification.
- Four-frame history; eight generated frames per autoregressive chunk, then
  re-anchor history on the real trajectory. Actions always come from the dataset.
- VAE-reconstructed targets, excluding the initial four conditioning frames.
- SSIM/PSNR/LPIPS computed per trajectory and then averaged across trajectories.
- FID from all predicted frames using pytorch-fid pool3 features and sample
  covariance; FVD from the first 16 predictions of up to 2,048 eligible
  trajectories using the StyleGAN-V I3D detector and population covariance.
- The existing float-to-uint8 path for FID/FVD, validation ordering, rank-specific
  RNG streams, and metadata validation for cached real statistics.

The existing timing region includes rollout and combined prediction/target VAE
decoding. Preserve or explicitly distinguish that timing definition when
comparing speed. ImageNet's evaluator and FID reference assets do not implement
this protocol.

## Integration with Priming

Use a domain package with three explicit commands, retaining the same internal
separation as Priming. All domain files live together:

```text
world_modeling/
  pretrain.py
  posttrain.py
  eval.py
  models/             # native RT-1/Bridge U-Nets, SD3 VAE, DINO
  data/               # episode datasets and loaders
  losses/             # reconstruction, feature branches, DistMatch adapter
  metrics/            # rollout evaluation, FID/FVD, aggregation
  utils/              # configuration, training, checkpoints, replay
  configs/{pretrain,posttrain,eval}/
  tests/
  scripts/            # bounded Slurm smoke and source-parity audit
  docs/               # migration assessment and validation artifacts
```

Expose `python -m world_modeling.pretrain`, `.posttrain`, and `.eval` with
`--config`, explicit data/asset/output paths, and distinct initialization/resume
options. Keep an optional multi-step sampler for the flow baseline. Replace
working-directory-dependent `sys.path` injection with package-relative imports
and explicit architecture construction: the current Bridge wrappers rely on
which top-level `unet`, `datasets`, `utils`, and `encoders` modules import first.

| Existing Priming component | Migration decision |
|---|---|
| `utils/config.py`, file/path helpers | Reuse where their contracts fit; remove Hydra run-directory assumptions |
| `utils/ema.py` | Reuse constant-decay arithmetic only after verifying parameter/buffer and historical checkpoint parity |
| Checkpoint/RNG utilities | Reuse patterns; add world-specific state conversion and stage semantics |
| `utils/distributed.py` | Its initializer requires eight ranks; support small smoke runs through a domain initializer or compatible generalization |
| `models/priming.py` | Add the source U-Nets; ImageNet's class-conditioned transformer is a different model |
| `losses/reconstruction.py` | Add the source weighted latent objective rather than using ImageNet's fixed-noise/perceptual recipe |
| `losses/matching.py`, `losses/discrepancy.py` | Do not substitute directly: flattened spatial outputs, calibrated fixed bandwidths, epsilon mixture, iteration budget, and coefficient differ |
| Root training/eval scripts | Keep their ImageNet contracts; share lower-level helpers selectively |

The existing matcher interface `(live, detached_reference, real) → loss, metrics`
is a useful common boundary. Generalize implementations only after world-model
feature layout, scaling, scalar loss, latent gradients, and DDP normalization
match the source. In particular, a spatially flattened OT solve need not equal
separate transport problems at aligned positions.

## Available artifacts and dependency verification

The archive stores `driftworld/storage` as a link to
`/n/netscratch/sitanc_lab/Lab/jaykim/driftworld`. That directory is accessible
and contains processed data, SD3/DINO assets, metric libraries, and checkpoints.
The follow-up check verified readable SD3 VAE config/weights, DINO weights and
hub source, pytorch-fid source, StyleGAN-V directory, and the cached I3D detector.
All four final checkpoint files resolve. This is a path/readability check, not
a complete weight-integrity or dataset-decode audit.

Both datasets have readable train/validation manifests. First, middle, and last
MP4/NPZ pairs in each split exist and are readable. Manifest counts are:

| Dataset/split | Manifest entries | At least five frames |
|---|---:|---:|
| RT-1 train | 85,032 | 84,566 |
| RT-1 validation | 2,180 | 2,165 |
| Bridge train | 53,192 | 53,178 |
| Bridge validation | 6,872 | 6,871 |

Checkpoint metadata confirms:

| Dataset | Pretraining checkpoint under `outputs/` | Post-training checkpoint under `outputs/` |
|---|---|---|
| RT-1 | `rt1_flow_xpred/ckpt_save/ckpt-step84608.pth` | `rt1_flow_distmatch_sinkhorn/ckpt_save/ckpt-step4000.pth` |
| Bridge-V2 | `bridge_flow_xpred/ckpt_save/ckpt-step53184.pth` | `bridge_flow_distmatch_sinkhorn/ckpt_save/ckpt-step4000.pth` |

Both post-training checkpoints record initialization from the corresponding
pretraining EMA and include source hashes. Metadata was read with CPU memory
mapping; weight integrity and output parity have not yet been checked.

Existing saved evaluation summaries provide regression targets, not fresh measurements:

| Dataset | Videos | Predicted frames | SSIM | PSNR | LPIPS | FID | FVD |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-1 | 2,165 | 86,135 | 0.878323 | 26.764669 | 0.062409 | 3.161854 | 30.961238 |
| Bridge-V2 | 6,871 | 228,693 | 0.857248 | 24.548744 | 0.067281 | 2.821857 | 32.577865 |

Each summary is under its post-training output directory at
`eval_on_validation_multigpu/step4000_cfg1.0_flow-euler-1step_chunkAuto8_streaming/aggregate_summary.json`.

**Source now available:** `/n/netscratch/sitanc_lab/Lab/jaykim/distmatch/src`
imports successfully and provides the required `DistMatch`, `FeatureBranches`,
and `Sinkhorn` API. Use this source explicitly during migration. Both archived
projects still have links to the inaccessible `/n/home01/woobin/distmatch/`;
the new directory does not repair those links automatically. The older
`../packages/distmatch` next to Priming remains an incompatible substitute.

**Version provenance remains to be resolved:** the recovered `src/` and
`pyproject.toml` report 0.6.0; bundled `build/lib/` and stale egg-info report
0.5.0. Both historical world-model post-training checkpoints and their training
logs report **0.4.0**. No Git history or 0.4.0 source artifact was found in the
new directory. A synthetic FP32/FP64 comparison found exactly equal local
Sinkhorn losses, gradients, and diagnostics between its 0.5.0 build and 0.6.0
source. This supports compatibility but does not prove equivalence to 0.4.0.
Record the actual source and validate the historical recipe before claiming
exact training reproduction. No top-level license file was present in the copy;
include appropriate attribution/licensing when packaging it for release.

Source fingerprint recorded by both migrated H100 training runs for the 13
Python files under `src/distmatch`:
`587edb55204d5d55926872e3d250ae220f8478bc956457b3fc8e2f01bfb14114`.
This SHA-256 hashes sorted relative paths, a NUL, each file's bytes, and a NUL.
The directory also has a self-referential `storage` link and an inaccessible
`distmatch_deprecated` link; neither is needed to import the core matcher.

The environment also needs deliberate packaging: archived DriftWorld declares
Python 3.12, PyTorch 2.11/CUDA 12.8, and Diffusers 0.37.1, while Priming pins
Python 3.11 and PyTorch 2.6/CUDA 12.4. Start with a separate world-model runtime
specification and validate compatibility before changing shared pins. Dataset
conversion dependencies can remain optional. Retain upstream license notices
when importing the U-Nets and other third-party code.

## Implementation and validation order

1. Pin the recovered matcher source and record its provenance; resolve or
   explicitly document the historical 0.4.0 version gap. Reference checkpoint,
   asset-path, and dataset-manifest inventories are now available above.
2. Port the two model variants, VAE conventions, data adapters, historical
   checkpoint reader, and rollout code. Validate prediction parity using fixed
   latent inputs and actions; exercise both variants in the same process.
3. Port pretraining. Compare source/new weighted loss, parameter gradients,
   optimizer step, and EMA update on the same small batch. Add explicit recovery
   state: source pretraining checkpoints do not save RNG/worker states, so their
   resume behavior should not be described as exact stochastic continuation.
4. Port matching and replay using the recovered DistMatch source. Check branch layout and
   scaling, detached references, full-population/replay gradient equivalence,
   EMA initialization, and distributed gradient normalization.
5. Port streaming evaluation with fixed validation IDs, seed/rank layout, VAE
   posterior draws, chunk resets, metric preprocessing, and cache metadata.
   Compare short reference rollouts before running the full validation sets.
6. Run a small GPU pretrain → saved checkpoint → posttrain → eval smoke, then
   validate historical checkpoint evaluation against the recorded targets.
   Package portable configs, assets guidance, and root README links after parity.

During this assessment, the archived flow-matching unit suite passed all eight
tests on CPU in the existing Priming audit environment. These cover path/loss
conventions and one-/multi-step and autoregressive sampler behavior. The archived
streaming-metrics suite could not import because that environment lacks SciPy;
none of its metric checks ran. This does
not establish full dependency compatibility, model-output parity, or GPU training
correctness.

The follow-up DistMatch audit passed 44 existing CPU tests: 34 feature,
normalization, and objective tests; eight host-training tests; and two distributed
tests (including two-process Gloo). Host tests must run from the DistMatch root
to avoid its top-level `data`/`examples` imports colliding with Priming modules.
Additional synthetic checks passed for archived latent replay versus a direct
full-population gradient (`C=4, B=16, N=1`, microbatch 8), entropy-inclusive versus
cost-only gradients, and bundled 0.5.0 versus 0.6.0 local Sinkhorn behavior.
These initial checks used toy feature extractors. The migration subsequently
added 22 passing CPU checks and a bounded H100 validation job. See the package
README and validation record for the current implementation status.
