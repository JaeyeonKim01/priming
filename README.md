# Priming Models / OST release staging

This directory collects the four ImageNet production EMA models, the frozen
FID/FDr⁵ evaluation record, and an offline [results dashboard](dashboard/index.html).
The paper calls these models **Priming Models**; the original code and training
run names use **OST**. The production OT post-training reproduction workflow
is in [training](training/README.md). The production pretraining recipes and
retained source code are packaged in [pretraining](pretraining/README.md).

| Item | Location |
|---|---|
| Project and paper map | [PROJECT_MAP.md](PROJECT_MAP.md) |
| B/L at 80/160 epochs, 25,000 OT updates | [checkpoint manifest](checkpoints/manifest.json) |
| Latest ImageNet results and metric protocol | [evaluation README](eval/README.md) |
| Reproduction audit | [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) |
| B/L 160-epoch → 25k OT reproduction | [training guide](training/README.md) |
| B perceptual / B no-perceptual / L perceptual pretraining | [pretraining guide](pretraining/README.md) |
| Fresh full-precision results | [full_reproduction_20260930.csv](eval/full_reproduction_20260930.csv) |
| Paper-batch full-precision results | [paper_results.csv](eval/paper_results.csv) |

Each `checkpoints/*.pth` is a relative symlink to its original full production
checkpoint through the repository's `fdloss_work_dirs` scratch mount. No
model weights are stored under this home directory. The evaluator loads EMA
0.999 directly from those source files. The JSON beside each link contains
the inference configuration. The manifest records source SHA256 hashes and
symlink targets. To use this staging directory elsewhere, provide the
checkpoint files at the linked paths or replace the links with corresponding
local paths after verifying the recorded hashes. Third-party baseline
checkpoints, feature encoder weights, and real-image statistics are identified
in [the evaluation guide](eval/README.md).

```bash
python OST_public/verify_checkpoints.py
```

The evaluation package uses the repository's frozen evaluation runtime and
pinned vendor sources. The training package contains its own frozen production
runtime; large datasets, pretrained weights, and encoder assets remain external.
The pretraining package preserves separate B/L source snapshots and production
configs. Its migration is checked without launching new pretraining runs;
historical source and restart limitations are documented in its guide.

Git includes the runtime source code, production configs, provenance, and
retained validation/evaluation records. Weights (including local checkpoint
symlinks), caches, logs, allocation links, and live scheduler/launch state are
ignored. On this cluster, `python link_checkpoints.py` recreates the four local
checkpoint links from the manifest; elsewhere, supply the matching checkpoint
files at the configured locations. Empty `training/logs/` and `training/results/`
directories are retained for the launch scripts.
