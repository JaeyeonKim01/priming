# Pretraining source record

`copied_files.json` records the original repository-relative source and SHA256
for every copied runtime/provenance file. `runtime_manifest.json` freezes the
release runtime files independently. `scripts/verify.py` checks those hashes
and compares each complete runnable recipe against its original source,
allowing only output/project/run-name and W&B logging changes.

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
