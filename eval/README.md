# ImageNet FID / FDr⁵ reproduction

The [paper protocol](FID_SETUP_AND_RESULTS.md) is the authority for this
evaluation. The copied [paper_results.csv](paper_results.csv) has the 11
headline models, and [reports](reports) contains all 33 individual 50,000
image results plus the original validation and environment records. The
unchanged scripts that produced those results are in [archive](archive).
Their recorded SHA256 hashes remain verifiable. Do not overwrite that archive
or the completed September 25 scratch batch.

The independent September 30 full reproduction is complete: 11 models,
three 50,000-image sampling seeds each, and both FID and FDr⁵ on every run.
See the [fresh comparison report](full_reproduction_20260930.md),
[full-precision CSV](full_reproduction_20260930.csv),
[per-seed CSV](full_reproduction_20260930_per_seed.csv), and
[validation JSON](full_reproduction_20260930.json). Its raw artifacts are at
`/n/netscratch/sham_lab/Lab/jaykim/OST_public_full_eval_20260930`; all jobs
used `kempner_h100_priority`. To rerun the complete artifact checks and
summary against that batch:

```bash
/n/home10/jaykim/.conda/envs/fdloss/bin/python OST_public/eval/summarize_full.py \
  --output-dir /n/netscratch/sham_lab/Lab/jaykim/OST_public_full_eval_20260930
```

Run the read-only result and artifact checks from the repository root:

```bash
/n/home10/jaykim/.conda/envs/fdloss/bin/python OST_public/eval/verify_results.py
/n/home10/jaykim/.conda/envs/fdloss/bin/python OST_public/eval/verify_results.py \
  --scratch-root /n/netscratch/sham_lab/Lab/jaykim/three_seed_eval_20260925
python OST_public/dashboard/build.py
```

The optional scratch check verifies sample IDs, 50 images per class, distinct
rank byte streams across seeds, rank counts, saved generated moments, and
report parity. The result check verifies the pinned evaluator and reference
hashes, FDr⁵ component arithmetic, and mean/sample SD against the paper CSV.
It does **not** regenerate images or recompute FID from retained moments.
The earlier paired FID audit and Table 4 audit provide separate upstream
cross-checks as described in the paper protocol.

Seven fresh GPU smoke jobs for the release adaptation have completed; their
[validation and comparison record](pilots/validation_20260930.json) covers
every distinct sampler family in the headline batch. These are 1,000-image
pilots and are excluded from the published 50,000-image summary.

For a new sampling batch, use [evaluate.py](evaluate.py). It preserves the
published generation, quantization, precision, feature, reference, and RNG
rules, while accepting a fresh `--output-dir` and the linked production EMA weights.
It is a new release-facing adaptation; the archived script is the exact
source of the published 33-run numbers. A 1,000-image smoke evaluation for
each distinct sampler family should precede the full batch. Use four GPUs and batch 32 for all
models except JiT-H; JiT-H uses eight GPUs and batch 64.

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/n/netscratch/sham_lab/Lab/workdir/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$PWD/tests/mmd/runs/torch_cache"
export DRIFT_COMPILE=0
PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  OST_public/eval/evaluate.py --model priming_b_80 --num-images 1000 \
  --seeds 1 --batch 32 --output-dir /path/to/new/eval_batch
```

The same command accepts `priming_b_160`, `priming_l_80`, `priming_l_160`,
`wflow_xl`, `drifting_b`, `drifting_l`, `pmf_l`, `pmf_h`, and `imf_xl`.
JiT-H uses `jit_h` with the archived two-node launch pattern. Output files
must go to a new namespace; the evaluator refuses to overwrite any existing
nonempty seed directory. A 1,000-image pilot is a runtime check, not a
headline metric.

After a new run, validate its artifact set with
`/n/home10/jaykim/.conda/envs/fdloss/bin/python OST_public/eval/verify_batch.py --output-dir "$OST_EVAL_OUTPUT" --model priming_b_80 --num-images 50000 --seeds 1 2 3`.
This check is read-only and separate from the archived 33-run validator.

On Kempner, the supplied Slurm wrappers request
`kempner_h100_priority` and account `kempner_sham_lab`. Submit from the
repository root after choosing a fresh scratch namespace:

```bash
export OST_EVAL_OUTPUT=/n/netscratch/sham_lab/Lab/jaykim/my_new_eval_batch
sbatch --export=ALL,OST_EVAL_OUTPUT="$OST_EVAL_OUTPUT" \
  OST_public/eval/run_4gpu.sbatch --model priming_b_80 --num-images 1000 --seeds 1
# JiT requires two nodes and one seed per job:
sbatch --export=ALL,OST_EVAL_OUTPUT="$OST_EVAL_OUTPUT" \
  OST_public/eval/run_jit_8gpu.sbatch --num-images 1000 --seeds 1
```

The default external assets are the same ones used for the paper. Their
locations can be changed with `OST_COMPARISON_CACHE`,
`OST_BASELINE_CHECKPOINTS`, `OST_FDR5_STATS`, `OST_INCEPTION_WEIGHTS`, and
`OST_FID_REFERENCE`. The official FID real reference must have SHA256
`412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
The pinned LTH14 torch-fidelity source is
`isolated_experiments/fid_protocol_audit_20260925/vendor/torch-fidelity`
at commit `bfe72b995473e4fee9351705d7b005f17442a4ca`.
The frozen FDr⁵ runtime is
`tests/runs/ot_metric_climb_20260904/runtime`. Baseline sampler sources
are under `isolated_experiments/three_seed_eval_20260925/vendor`.
The original pMF/iMF/JiT checkpoint SHA256 values are in
[checkpoint_hashes.json](reports/checkpoint_hashes.json).

The evaluator holds official Inception features for all generated images and
requires the frozen FDr⁵ encoders, ImageNet statistics, and the model-specific
VAE where applicable. Allocate adequate GPU memory and scratch space.
