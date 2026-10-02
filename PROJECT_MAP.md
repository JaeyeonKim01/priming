# Project map for the ImageNet release

The paper calls the released models **Priming Models**; the code and saved
runs use **OST**. Four ImageNet configurations pair B/L generators with
80/160-epoch reconstruction checkpoints. Each receives 25,000 feature-wise
OT post-training updates and is evaluated from EMA 0.999 weights.

| Stage | Current project location | Release relevance |
|---|---|---|
| Release reconstruction pretraining | `OST_public/pretraining/` | B perceptual/no-perceptual and L perceptual production recipes, retained B/L source snapshots, portable launch support and provenance; migration only |
| Feature-wise OT post-training | `scripts/main_mmd_and_eval.py`, `src/ot_representations_full/`, `packages/distmatch/` | Produces the four terminal checkpoints |
| Release OT reproduction | `OST_public/training/` | Frozen production runtime, B/L-160 configs, eight-H200 launch chains and scheduled evaluation audits |
| Production run arguments and full checkpoints | `fdloss_work_dirs/OST/<run>/` | Source identities and SHA256 hashes in the release manifest |
| Frozen FDr⁵ metric runtime | `tests/runs/ot_metric_climb_20260904/runtime/` | Five BF16 feature families, FP64 moments, ADM Inception reference |
| Official FID implementation | `isolated_experiments/fid_protocol_audit_20260925/vendor/torch-fidelity/` | Pinned LTH14 source, FP32 extraction, uint8 images |
| Baseline samplers | `isolated_experiments/three_seed_eval_20260925/vendor/`, `figures/W-Flow/` | pMF, iMF, JiT, W-Flow, Drifting |
| Three-seed evaluation | `isolated_experiments/three_seed_eval_20260925/` | Frozen scripts, 33 run reports, retained scratch features and IDs |
| Final manuscript and result record | `final_paper/onestep_newer/`, `final_paper/FID_SETUP_AND_RESULTS.md` | Headline definitions and paper table |
| Historical dashboards | `figures/benchmarks/`, `figures/samples/` | Some headline values use older protocols |

The release dashboard uses the September 30 independently reproduced
full-precision three-seed CSV and shows differences from the September 25–26
paper batch. It does not inherit old benchmark numbers from
`figures/benchmarks/`. Its four qualitative overview images are
copied from the production sample panels; they are separate from the 50,000
images per seed used for metrics.

The [evaluation guide](eval/README.md) gives runnable commands and asset
paths. The [reproduction status](REPRODUCTION_STATUS.md) separates the
completed paper batch, released checkpoint verification, and new pilot runs.
