# Pretraining source record

`copied_files.json` records the original repository-relative source and SHA256
for every copied runtime/provenance file. `runtime_manifest.json` freezes the
release runtime files independently. `scripts/verify.py` checks those hashes
and compares each complete runnable recipe against its original source,
allowing only resource paths, output/project/run-name, and W&B logging changes.

| Record | Meaning |
|---|---|
| `b_production_manifest.json` | Historical B production runtime/config/data identities |
| `b_*_production.yaml` | Exact B perceptual and no-perceptual configs |
| `b_*_initialization.json` | Actual production architecture, scientific contract, numerical backend and initialization hashes |
| `b_production_started.json` | Production inputs/initialization confirmed against validated B probes |
| `b_*_completed.json` | Retained original completion audits; not new runs |
| `b_original_launch.sbatch`, `b_original_environment.sh` | Original B topology, deterministic environment and launch command |
| `l_original_arguments.json` | August 12 original 160-epoch recipe and executed overrides |
| `l_resume_arguments.json` | August 14 resumed 160-epoch allocation arguments |
| `l_training_log_evidence.json` | Numbered log excerpts, original log hash, and Eastern-time interpretation |
| `l_later_320epoch_config.yaml` | Later overwritten config, retained to explain why it is not the release recipe |
| `l_snapshot_manifest.json` | September 10 retained L-study source identities; not an August allocation manifest |
| `l_original_launch.sh` | Retained L launch script |
| `l_historical_requirements.txt` | Original August L environment inventory |
| `environment.json` | Installed dependency versions inspected during migration |
| `release_lineage.json` | Actual released post-training arguments and their online pretraining sources |

All 103 copied B runtime files covered by its historical manifest match that
manifest. The remaining two files are the TVM license and README. All 93
copied L Python dependency files covered by the September snapshot manifest
match it. Its eight TVM dependency/license/documentation files were materialized
from the existing release training runtime, replacing the source snapshot's
external-directory symlink.

The August L allocation did not retain a complete immutable source manifest.
Its W&B metadata references a July git commit while the logged trainer differs
from that commit, so the commit alone cannot establish the executed source.
This package preserves the later retained L code snapshot and the confirmed
historical configuration; exact August code identity and GPU kernel selection
are not asserted. L also resumed at epoch 110 using its older checkpoint
format. No new training run has been performed to measure trajectory agreement.

## Historical training details

The four released checkpoints trace to these pretraining runs:

- B-80/B-160: `OST_B_t08_e160_perceptual_seed1_20260912`.
- L-80/L-160: `OST_L_aux_unif_t08_38545843`.

The B trainer, deterministic crop implementation, and dependencies match the
retained September 12 production manifest. B explicitly uses math SDPA,
deterministic cuDNN/algorithms, a native CPU bicubic crop backward, stateless
flip/noise/crop seeds, and complete all-rank Muon/RNG recovery.

The L recipe comes from the **original executed arguments**, not the run
directory's subsequently overwritten 320-epoch config. Its original training
ran August 12–14, 2026 (Eastern time), with a restart at epoch 110. The 80-epoch
source was saved August 13 at 16:37; the 160-epoch source August 14 at 14:12.
The separate 320-epoch continuation started August 16.

`runtime/l` is the unchanged September 10 retained L-study code snapshot.
It is not an independently hash-verified snapshot of the August pretraining
allocation. Original arguments and logs verify the L recipe and checkpoint
lineage; they do not record which SDPA kernel was selected. The retained code
uses the original nondeterministic backend settings, ordinary RNG/dataloader
streams, and CUDA crop backward. Its checkpoint format stores rank-zero
optimizer state rather than B's complete all-rank state. These limitations are
preserved and documented, not silently replaced by the B implementation.

The perceptual thresholds are also preserved: B 0.8 and L 1.0. Both include
all examples at fixed t=0.8. The underlying objectives, model factories, and
optimizer code have not been refactored or unified.

See [production provenance](README.md),
[release lineage](release_lineage.json), and
[migration validation](../reports/validation.json).
