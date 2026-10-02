# OT post-training reproduction

Reproduce Priming-B and Priming-L from their **160-epoch pretrained online
weights**, with **25,000 fresh AdamW updates**, eight H200s per model, and
**50,000-image online and EMA evaluations every 1,250 updates**, including
the terminal update. The separate [pretraining package](../pretraining/README.md)
contains the production reconstruction recipes and retained source snapshots;
its migration did not launch new pretraining runs.

## Recipe and source

The numerical code in `runtime/` is copied from the identical frozen B and L
production runtimes. [Provenance](provenance) records the original configs,
checkpoint SHA256 values, and external asset identities. The scientific
configs are unchanged; output directories, project, and run names are new.

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
| Execution | Four nodes × two H200s, microbatch 128/rank, no accumulation |
| Precision | Compiled BF16 generator; eager production encoder precision; encoder microbatch 8 |
| Evaluation | Online and EMA .999; 50k images each; every 1,250 updates; one step, CFG1 |

B uses perceptual pretraining. The generator initializes from pretrained
**online** weights with a fresh optimizer; probe weights never initialize
production. Checkpoints contain optimizer, EMA, counters, feature contract,
and all eight ranks' main and OT RNG states.

## Layout

- `configs/b160.json`, `configs/l160.json`: full production recipes.
- `configs/site.json`: Python environment, scratch output, and asset locations.
- `runtime/`: training entrypoint, model and feature code, local DistMatch,
  and the TVM import dependency with its license.
- `scripts/launch.py`: two independent `probe → production` Slurm chains.
- `scripts/status.py`: progress and strict completeness checks for scheduled draws.
- `reports/`: validation, preflight, launch, progress, and evaluation evidence.
- `results/`: symlinks to allocation snapshots on scratch; weights stay on scratch.

## Run on Kempner

From `OST_public` (commands also work with an absolute training path):

```bash
PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
"$PYTHON" training/scripts/verify.py --assets
"$PYTHON" training/scripts/launch.py          # show submission commands
"$PYTHON" training/scripts/launch.py --submit
"$PYTHON" training/scripts/status.py
```

The launcher requests `kempner_h200`, account `kempner_sitanc_lab`, four nodes
and two GPUs per node. Probes request two hours; production requests 48 hours
with up to two checkpoint-based requeues. At most 16 H200s run concurrently.
Each production job depends on its own successful probe and checks that the
validated code/config manifests still match. Repeated launch commands reuse
recorded job IDs. Fresh output directories are required.

The probe runs 130 updates through warmup, verifies saved state, checks exact
identical-gradient Adam/EMA replay, and resumes for five more updates.
CUDA backward repeatability is recorded separately; bitwise training
trajectory agreement across GPU allocations is not promised.

For another machine, install the versions in `reports/environment.json`,
provide ImageNet images, encoder weights/calibrations, statistics, and the
two pretrained checkpoints listed in `provenance/`, then adapt site and
asset paths before freezing a new launch. The checked-in configs currently
record this cluster's asset paths; datasets and large weights are external.
This is a production reproduction package, not a one-command asset download.

The September 30 reproduction exhausted the `sham_lab` scratch group quota
before production could create its allocation directories. This experiment's
storage was moved to
`/n/netscratch/sitanc_lab/Lab/jaykim_OST_public_ot_20260930`; its original
configured scratch path is a symlink to that directory. Production was
resubmitted through `sg sitanc_lab -c 'sbatch ...'` so newly created files use
the available Sitan C group quota. The Slurm account is still
`kempner_sitanc_lab`. Scientific configs and preflight evidence are unchanged.
See `reports/storage_relocation.json` and `reports/launch.json` for the
storage mapping and current job IDs; the failed submissions are retained in
`reports/quota_failure_launch.json`.

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
"$PYTHON" training/audit_milestone.py --through 2500
```

The resulting `reports/verified_both_2500.json` reports measured differences
from the historical trajectory without assuming bitwise reproducibility.

Slurm requeue restores complete training state from the allocation's own
latest checkpoint. A missing intermediate evaluation is completed before
the next training update. The final checkpoint is named
`step_0024999.pth`, representing **25,000 completed updates**. The four
released reference checkpoint links are separate from these new outputs.
