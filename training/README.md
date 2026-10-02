# OT post-training reproduction

Reproduce Priming-B and Priming-L from their **160-epoch pretrained online
weights**, with **25,000 fresh AdamW updates**, eight H100/H200 GPUs per model, and
**50,000-image online and EMA evaluations every 1,250 updates**, including
the terminal update. The separate [pretraining package](../pretraining/README.md)
contains the production reconstruction recipes and retained source snapshots;
its migration did not launch new pretraining runs.

## Recipe and source

The numerical code in `runtime/` is copied from the identical frozen B and L
production runtimes. [Provenance](provenance) records the original configs,
checkpoint SHA256 values, and external asset identities. The scientific
configs are unchanged; only asset paths, output directories, project, and run names differ.

| Setting | Both models |
|---|---|
| Pretrained source | 160 epochs / 200,160 completed pretraining updates |
| OT duration | 25,000 optimizer updates (20 reporting epochs × 1,250) |
| Population | 16 classes × 64 live; 128 real and 64 detached self samples/class |
| Features | 84 ResNet-MAE + 13 ConvNeXt; equal family weights |
| Sinkhorn | Epsilon .001/.01, weights .75/.25; coefficient 5; 1,000 iterations; tolerance 1e-6 |
| AdamW | LR 1e-5; 125-update warmup, then constant; betas .9/.95; weight decay 0; clip 1 |
| EMA | Inherit source .9999 shadow/counter, then use decay .999 |
| Seeds | Main 1; private OT stream 7919 |
| Execution | Eight H100/H200 GPUs, microbatch 128/rank, no accumulation |
| Precision | Compiled BF16 generator; eager production encoder precision; encoder microbatch 8 |
| Evaluation | Online and EMA .999; 50k images each; every 1,250 updates; one step, CFG1 |

B uses perceptual pretraining. The generator initializes from pretrained
**online** weights with a fresh optimizer; probe weights never initialize
production. Checkpoints contain optimizer, EMA, counters, feature contract,
and all eight ranks' main and OT RNG states.

## Layout

- `configs/b160.json`, `configs/l160.json`: full production recipes.
- `configs/site.json`: output, GPU topology, and asset locations.
- `runtime/`: training entrypoint, model and feature code, local DistMatch,
  and the TVM import dependency with its license.
- `scripts/launch.py`: two independent `probe → production` Slurm chains.
- `scripts/status.py`: progress and strict completeness checks for scheduled draws.
- `reports/` and `provenance/`: retained historical evidence.
- `state/`: ignored fresh validation and scheduler state.
- `results/`: symlinks to allocation snapshots on scratch; weights stay on scratch.

## Run on H100/H200 GPUs

From `priming`, activate the [shared environment](../docs/setup.md) and install
[the external assets](../docs/assets.md). Paths in `configs/` resolve from the
repository regardless of the caller's working directory. The default is one
node with eight GPUs; the numerical recipe still uses eight ranks.

```bash
conda activate priming
python training/scripts/verify.py --assets
python training/scripts/check_cpu.py
# Prepare a standalone eight-GPU run:
python training/scripts/prepare.py --arm b160 --job outputs/posttraining_allocations/b160
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job outputs/posttraining_allocations/b160
```

Use `l160` with a separate job directory for L. Choose a new `logging.exp_name`
for each independent run. To resume a prepared standalone run, pass
`--resume outputs/posttraining/priming/RUN_NAME/checkpoints/latest.pth` to
`train.py`. The standalone entrypoint uses the retained initialization and
recovery instrumentation; it does not automatically run the Slurm probe chain.

For Slurm, edit `configs/site.json` to set your optional partition/account and
GPU layout (nodes × GPUs per node must equal eight):

```bash
python training/scripts/launch.py          # preview only
python training/scripts/launch.py --submit
python training/scripts/status.py
```

The launcher creates two independent `probe → production` chains. Probes run
130 updates, audit saved state, check identical-gradient Adam/EMA replay, and
resume for five updates. Production requires the matching successful probe.
Probes request two hours; production requests 48 hours with up to two
checkpoint-based requeues. Up to 16 GPUs are requested concurrently.
CPU/memory/time reservations can be adjusted for your system. GPU backward
repeatability is recorded separately; bitwise agreement across hardware or
allocations is not promised. H100 memory suitability must be checked by the
probe; no new H100/H200 training run was performed for this portability change.

Historical launch evidence is retained in `reports/` and `provenance/`.
New launches use ignored `state/` files and never reuse historical job IDs.

## Evaluation and recovery

The online evaluator preserves the historical production protocol: six raw
FD components, FDr5 excluding ConvNeXt, FID from the BF16 DistMatch path,
and precision/recall from the FP32 ADM path. Those FID values should be
compared with the historical production trajectory. The release's
[official final FID protocol](../eval/README.md) is a separate evaluation
and should be used for comparisons with the final paper table.

The first evaluations are at update 1,250, followed by 2,500. Each milestone
has two separate 50k draws per model. The status collector only accepts a
complete draw with matching timestamp, weights, interval, CFG, and six
components; fragments from separate draws cannot fill missing results.
`reports/milestone_2500.json` appears once both models pass through 2,500.

After both milestones finish, verify their saved optimizer/EMA/RNG state,
finite updates, allocation source hashes, complete draws, and historical
metric comparison with:

```bash
python training/audit_milestone.py --through 2500
```

The resulting `reports/verified_both_2500.json` reports measured differences
from the historical trajectory without assuming bitwise reproducibility.

Slurm requeue restores complete training state from the allocation's own
latest checkpoint. A missing intermediate evaluation is completed before
the next training update. The final checkpoint is named
`step_0024999.pth`, representing **25,000 completed updates**. The four
released reference checkpoint links are separate from these new outputs.
