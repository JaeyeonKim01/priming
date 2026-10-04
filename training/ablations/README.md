# ImageNet post-training ablations

These experiments start from the released **perceptual Priming-B-160 online
weights** and run **25,000 updates**. They share the reference OT run's optimizer,
sampling populations, seeds, EMA initialization, and evaluation schedule.
Only the feature space or distributional loss changes.

Use the [shared environment](../../docs/setup.md), download
`checkpoints/pretraining/priming_b_160.pth`, and install the
[training and evaluation assets](../../docs/assets.md). Configure asset locations
in the selected JSON recipe and cache locations in [site.json](../configs/site.json).
The standalone commands below require **eight H200 GPUs** for training.
Run all commands from the repository root with `conda activate priming`.

## SIM feature-space training

**Ablation study 1.** Replace ResNet-MAE and ConvNeXt with three frozen pooled
features: SigLIP SO400M, ViT-MAE-L, and Inception-V3. Each receives weight 1/3.
The [SIM provider](../runtime/src/ot_representations_sim/provider.py) preserves
the recorded preprocessing, BF16 extraction, online real-image references,
and microbatch feature-gradient replay. The checked-in
[calibration](../runtime/src/ot_representations_sim/calibration.json) contains
the original bandwidths and feature identities; cached real feature banks are
not needed. SIM still needs the dataset class-index arrays and evaluation
assets described in the asset guide.

The [SIM recipe](../configs/b160_sim.json) keeps feature-wise OT with coefficient
5 and the reference epsilon mixture (0.001/0.01 with weights 0.75/0.25).

```bash
python training/scripts/verify.py --arm b160_sim --assets
python training/scripts/prepare.py --arm b160_sim --job outputs/posttraining_allocations/b160_sim
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job outputs/posttraining_allocations/b160_sim
```

Inception, SigLIP, and ViT-MAE also contribute to FDr⁵. The SIM FDr⁵ result
therefore measures performance partly in the optimized feature spaces, rather
than an independent held-out-feature comparison.

## Other distributional losses

**Ablation study 2.** Keep the full 84 ResNet-MAE and 13 ConvNeXt outputs and
their original family weights and bandwidths, replacing OT with:

| Recipe | Pairwise discrepancy for each feature output |
|---|---|
| [Laplacian MMD](../configs/b160_laplacian.json) | `-exp(-||x-y||₂ / sigma)` |
| [Energy](../configs/b160_energy.json) | `||x-y||₂ / sigma` |

Both use coefficient 5 and the same independent detached generated-reference
population as OT: the generated–real discrepancy minus the generated–self
discrepancy, with self coefficient 1. Energy uses an unsmoothed norm
(`energy_delta=0`). Laplacian MMD bounds the direct-distance temporary memory
while retaining every pair. Its logged scalar is a gradient surrogate, not
an MMD-squared estimate. The implementation and its retained numerical tests
are in [loss_ablation.py](../runtime/packages/distmatch/src/distmatch/objectives/loss_ablation.py)
and [tests](../tests).

```bash
# Laplacian MMD
python training/scripts/verify.py --arm b160_laplacian --assets
python training/scripts/prepare.py --arm b160_laplacian --job outputs/posttraining_allocations/b160_laplacian
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job outputs/posttraining_allocations/b160_laplacian

# Energy distance: run separately on eight GPUs
python training/scripts/verify.py --arm b160_energy --assets
python training/scripts/prepare.py --arm b160_energy --job outputs/posttraining_allocations/b160_energy
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job outputs/posttraining_allocations/b160_energy
```

For the matched OT reference, use `--arm b160` with the
[standard post-training instructions](../README.md). The Slurm launcher remains
specific to the B/L reference runs; submit the standalone ablation commands
through your scheduler when needed.

## Checkpoints and evaluation

The default run directories are
`outputs/posttraining/priming/priming_b160_{sim,laplacian,energy}_25000_seed1`.
Each run saves `checkpoints/step_0024999.pth` at 25,000 completed updates.
The training run inherits the pretrained EMA shadow and counter, then uses
decay 0.999. Online and EMA evaluations run every 1,250 updates.
Resume a prepared job with `training/scripts/train.py --job JOB --resume CHECKPOINT`.
For an independent rerun, choose a new `logging.exp_name` and job directory.

Training-time FID uses the historical BF16 evaluator. To compare a terminal
checkpoint with the final paper results, use the separate
[official FID/FDr⁵ evaluator](../../eval/README.md): four GPUs, batch 32,
50,000 images, and sampling seeds 1/2/3. `--model priming_b_160` selects the shared
generator architecture; `--checkpoint` selects your ablation weights and records
their SHA256 in the evaluation output. It must contain the terminal EMA 0.999
and 25,000-update counters.

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$PWD/assets/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/assets/torch}"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Start with a 1,000-image pilot in a separate output directory.
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  eval/evaluate.py --model priming_b_160 \
  --checkpoint outputs/posttraining/priming/priming_b160_sim_25000_seed1/checkpoints/step_0024999.pth \
  --num-images 1000 --seeds 1 --batch 32 --output-dir outputs/evaluation/sim_pilot
python eval/verify_batch.py --model priming_b_160 --num-images 1000 --seeds 1 \
  --checkpoint outputs/posttraining/priming/priming_b160_sim_25000_seed1/checkpoints/step_0024999.pth \
  --output-dir outputs/evaluation/sim_pilot

# Full paper-protocol evaluation
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  eval/evaluate.py --model priming_b_160 \
  --checkpoint outputs/posttraining/priming/priming_b160_sim_25000_seed1/checkpoints/step_0024999.pth \
  --num-images 50000 --seeds 1 2 3 --batch 32 --output-dir outputs/evaluation/sim
python eval/verify_batch.py --model priming_b_160 --num-images 50000 --seeds 1 2 3 \
  --checkpoint outputs/posttraining/priming/priming_b160_sim_25000_seed1/checkpoints/step_0024999.pth \
  --output-dir outputs/evaluation/sim
```

For energy or Laplacian MMD, substitute its checkpoint and use a distinct output
directory. The evaluator refuses to overwrite a nonempty seed directory.
Ablation terminal weights are not included in the eight released checkpoints;
these recipes produce them locally.

## Reference results

The September 26 reevaluation used one trained terminal checkpoint per arm,
with three sampling seeds × 50,000 images. These are sampling means ± sample
standard deviations, not independent training replicates.

| Arm | Official FID ↓ | FDr⁵ ↓ |
|---|---:|---:|
| Full-feature OT reference | 1.3841 ± 0.0130 | 11.2286 ± 0.0157 |
| SIM OT | 1.6469 ± 0.0290 | 6.1135 ± 0.0203 |
| Laplacian MMD | 1.8479 ± 0.0290 | 11.2593 ± 0.0245 |
| Energy | 1.9490 ± 0.0524 | 11.5825 ± 0.0213 |

[Full-precision results](results.csv) · [Per-seed results](per_seed.csv) ·
[Result provenance](provenance.json)

Use these retained measurements as the reference; a new training run can vary.
[Original recipe identities](../provenance/ablations.json) and
[runtime source hashes](../provenance/ablation_runtime.json) record the completed
experiments used for this package. Scientific config fields are preserved;
only data/checkpoint/cache/output locations and run names are made portable.
Run `python training/scripts/check_cpu.py` to check the packaged loss gradients,
SIM replay, sampling streams, and all existing training regressions.

For a bounded startup check, add `--smoke-steps 2 --output-dir EXTERNAL_OUTPUT`
to preparation and use a new job directory. See [GPU startup checks](../../docs/validation.md).
After a bounded or complete run, audit its saved counters, optimizer/EMA,
per-rank RNG state, configured loss contract, and finite updates with:

```bash
python training/scripts/audit_checkpoint.py --config JOB/config.json \
  --output JOB/checkpoint_audit.json
```
