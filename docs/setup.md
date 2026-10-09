# Environment and storage setup

Use Linux x86-64, Python 3.11, and the pinned PyTorch 2.6.0/CUDA 12.4 environment.
The [environment lock](../old/environment/requirements-linux-cu124.txt) retains the
production pins for the retained workflows and their dependencies. Packages
used by removed baselines, dashboards, and experiment tracking are excluded.
The CPU tests use Python's standard `unittest` module.

```bash
bash scripts/create_env.sh
conda activate priming
export PYTHONNOUSERSITE=1
python scripts/check_environment.py
python scripts/check_environment.py --cuda  # inside a GPU allocation
```

`environment.yml` creates the environment; `create_env.sh` runs dependency and
import checks against the environment destination you select. A fresh custom
prefix can be used with `bash scripts/create_env.sh --prefix /scratch/priming/env`.
Eight ranks are required for the supplied training recipes. Paper FID/FDr5
uses four ranks and batch 32. Preserve hardware, precision, rank/batch settings,
and B/L backend behavior when comparing numerical trajectories.

## Assets outside the checkout

Choose a shared storage directory and install the environment/assets there:

```bash
export PRIMING_STORAGE=/scratch/priming
mkdir -p "$PRIMING_STORAGE"/{conda-pkgs,pip-cache,huggingface,torch}
export CONDA_PKGS_DIRS="$PRIMING_STORAGE/conda-pkgs"
export PIP_CACHE_DIR="$PRIMING_STORAGE/pip-cache"
bash scripts/create_env.sh --prefix "$PRIMING_STORAGE/env"
conda activate "$PRIMING_STORAGE/env"
hf auth login
python scripts/download_assets.py --group dataset --root "$PRIMING_STORAGE"
python scripts/download_assets.py --group pretraining --root "$PRIMING_STORAGE"
```

The download helper writes `data/` and `assets/` under `--root`; environment
cache overrides do not redirect its destinations. Point the commands at the
prepared paths:

```bash
export HF_HOME="$PRIMING_STORAGE/assets/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PRIMING_STORAGE/assets/torch"
```

Pretraining accepts `--data-path`, `--hf-home`, `--torch-home`, and
`--lpips-weights`. Post-training accepts `--data-path`, `--encoder-cache`,
`--init-checkpoint`, `--hf-home`, and `--torch-home`; its recipe specifies the
scheduled evaluation reference locations. Standalone evaluation accepts
`--checkpoint`, `--stats-dir`, `--fid-reference`, and `--inception-weights`.
All selected paths may be absolute. Dataset paths select the **parent** of
`imagenet-latents-images`. Output paths may be on shared storage.

Training compiler caches and temporary libraries use node-local `/tmp`,
separated by run, job, and rank. Set `PRIMING_LOCAL_TMP` to another node-local
writable disk if needed. Keep checkpoints and assets on shared storage.
Offline execution loads already verified local encoder files.

See [asset preparation](assets.md), [pretraining](pretrain.md),
[post-training](posttrain.md), and [evaluation](eval.md).
