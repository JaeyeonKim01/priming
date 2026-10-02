# Evaluation: FID and FDr⁵

Run from `priming` in the [shared environment](../docs/setup.md). All evaluator,
baseline sampler, and metric source code is included under `runtime/` and
`vendor/`; [source hashes](provenance/copied_sources.json) and
[vendor revisions](provenance/vendors.json) preserve their identities.
Only datasets, model weights, encoder caches, and reference statistics are
external; see [asset setup](../docs/assets.md).

## Retained results

The September 30 reproduction contains 11 models × three sampling seeds ×
50,000 images. All 33 saved runs passed validation. See the
[full report](full_reproduction_20260930.md),
[full-precision CSV](full_reproduction_20260930.csv), and
[quantitative dashboard](../dashboard/index.html).
`archive/`, `reports/`, and dated reports retain original run paths as provenance;
those paths are not dependencies of the public entrypoint.

```bash
python eval/verify_results.py
python eval/check_table.py
python dashboard/build.py
```

These verify the included result records and summary arithmetic without GPU
sampling. Optional raw-artifact validation requires your own downloaded/copied
artifact tree: `python eval/verify_results.py --scratch-root outputs/paper_evaluation`.
The [metric protocol](FID_SETUP_AND_RESULTS.md) documents historical precision
and reference choices; preserve those for comparisons against the paper.

## Generate a new evaluation

Start with a 1,000-image pilot. Use four GPUs and batch 32 except for JiT-H,
which uses eight GPUs and batch 64. The production protocol uses 50,000 images
and seeds 1, 2, 3. Changing rank count/batch size can change the sampling stream.

```bash
conda activate priming
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="$PWD/assets/huggingface" TORCH_HOME="$PWD/assets/torch"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  eval/evaluate.py --model priming_b_80 --num-images 1000 --seeds 1 \
  --batch 32 --output-dir outputs/evaluation/pilot_b80
```

Other model keys: `priming_b_160`, `priming_l_80`, `priming_l_160`, `wflow_xl`,
`drifting_b`, `drifting_l`, `pmf_l`, `pmf_h`, `imf_xl`, and `jit_h`.
For JiT, use `--nproc_per_node=8 --model jit_h --batch 64` and one seed per job.
The evaluator refuses to overwrite an existing nonempty seed directory.
A pilot checks execution; it does not establish a headline metric.

```bash
python eval/verify_batch.py --output-dir outputs/evaluation/pilot_b80 \
  --model priming_b_80 --num-images 1000 --seeds 1
```

On Slurm, activate `priming`, set a fresh output path, and add your scheduler's
partition/account options when required:

```bash
export PRIMING_ROOT="$PWD"
export PRIMING_EVAL_OUTPUT="$PWD/outputs/evaluation/pilot_b80"
sbatch eval/run_4gpu.sbatch --model priming_b_80 --num-images 1000 --seeds 1
# JiT: default two nodes × four GPUs; one node × eight is also supported.
sbatch --nodes=1 --gpus-per-node=8 eval/run_jit_8gpu.sbatch --num-images 1000 --seeds 1
```

Defaults use H100/H200-compatible PyTorch/CUDA and no institution-specific
scheduler settings. GPU execution on a new machine still requires a pilot.

## Asset overrides

Paths default to the following repository-relative locations. The corresponding
environment variable may select another location:

| Variable | Default |
|---|---|
| `PRIMING_COMPARISON_CACHE` | `assets/baselines/comparison` |
| `PRIMING_BASELINE_CHECKPOINTS` | `assets/baselines/checkpoints` |
| `PRIMING_FDR5_STATS` | `assets/fid_stats` |
| `PRIMING_INCEPTION_WEIGHTS` | `assets/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth` |
| `PRIMING_FID_REFERENCE` | `assets/fid_stats/jit_in256_stats.npz` |

The official FID reference must have SHA256
`412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
Frozen FDr⁵ encoders use the included DistMatch implementation. Official FID
uses the pinned LTH14 torch-fidelity source at commit
`bfe72b995473e4fee9351705d7b005f17442a4ca`.
