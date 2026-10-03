# Portable environment and GPU setup

All examples start in the repository directory, named `Priming`. Its Python
code, pinned baseline source, configs, and retained result records are included.
Data, checkpoints, and encoder assets are installed separately under this root;
see [assets.md](assets.md). No parent OST checkout is needed.

## Create the environment

On Linux x86-64 with Conda available:

```bash
bash scripts/create_env.sh
conda activate priming
export PYTHONNOUSERSITE=1
python scripts/check_environment.py
python scripts/check_environment.py --cuda  # run on an allocated GPU node
```

`environment.yml` creates **priming**, using Python 3.11.15 and a fully pinned
[pip dependency list](../environment/requirements-linux-cu124.txt), including
PyTorch 2.6.0 + CUDA 12.4 and torchvision 0.21.0. This is an environment recipe,
not a clone of a pre-existing environment. It uses no user-site packages.
The script fails if the environment already exists; use a fresh prefix or
remove your own old environment deliberately before recreating it.
Conda supports `CONDA_ENVS_PATH` and `CONDA_PKGS_DIRS` for alternative storage.
Allow ample space for CUDA wheels, caches, and the environment.

The wheels include their CUDA runtime. Use an NVIDIA driver compatible with
CUDA 12.4 and H100/H200 GPUs. GPU checks run inside an allocation, not on a
CPU-only login host. Eight ranks are required by the production training
recipes; use one eight-GPU server or an equivalent multi-node allocation.
Preserve batch sizes, accumulation, precision, and B/L-specific backends.
A different GPU/topology may change numerical trajectories and memory usage;
validate the supplied probe before a full reproduction.

The pinned environment was built from scratch and checked on CPU and H200. [Environment validation](../environment/validation.json) records
installed versions and whether CUDA was available. The separate [pipeline validation](validation.md) describes bounded training
checks; these do not establish full training or 50k metric reproduction.

## Paths and cached downloads

```bash
export HF_HOME="$PWD/assets/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PWD/assets/torch"
```

Entrypoints set these defaults automatically. Download/cache model assets
before running offline training. `pretraining/configs/site.json` and
`training/configs/site.json` use repository-relative paths. Absolute paths are
accepted for external storage. Prepare bundles again after moving a checkout:
a prepared bundle freezes the resolved paths and asset identities.

Slurm examples contain no institution-specific account or partition. Activate
`priming` before submission and supply your scheduler's partition/account where
needed. Direct `torchrun` commands are provided in each stage's guide.

Run `python scripts/check_portability.py` to repeat the source-only relocation
check (requires a Git checkout). It copies the public source into a temporary
`Priming/` directory and checks imports, CPU regressions, and saved results.

Official references: [NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) and [Conda environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Keep large files outside home

Choose a writable shared scratch directory before creating an environment or
fetching assets. Replace `/scratch/priming` with your storage location:

```bash
export PRIMING_STORAGE=/scratch/priming
mkdir -p "$PRIMING_STORAGE"/{conda-pkgs,pip-cache,tmp,huggingface,torch}
export CONDA_PKGS_DIRS="$PRIMING_STORAGE/conda-pkgs"
export PIP_CACHE_DIR="$PRIMING_STORAGE/pip-cache"
export TMPDIR="$PRIMING_STORAGE/tmp"
export HF_HOME="$PRIMING_STORAGE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PRIMING_STORAGE/torch"
export HF_DATASETS_CACHE="$PRIMING_STORAGE/datasets"
export XDG_CACHE_HOME="$PRIMING_STORAGE/cache"
bash scripts/create_env.sh --prefix "$PRIMING_STORAGE/env"
conda activate "$PRIMING_STORAGE/env"
python scripts/check_environment.py
hf download jaykim1102/priming checkpoints/priming_b_160.pth \
  --local-dir "$PRIMING_STORAGE"
python scripts/sample.py --model priming_b_160 \
  --checkpoint "$PRIMING_STORAGE/checkpoints/priming_b_160.pth" \
  --num-images 2 --output-dir "$PRIMING_STORAGE/samples/b160"
```

For training, set the site/config asset paths to the corresponding external
locations. Pretraining accepts an absolute `--output`; post-training preparation
accepts `--output-dir`. Store bundles/jobs on shared scratch as well. Conda
activation in batch jobs needs `source "$(conda info --base)/etc/profile.d/conda.sh"`
before `conda activate`. Do not assume login-shell activation is inherited.

Training launchers put compiler caches and temporary shared libraries in
node-local `/tmp/priming-UID/`, separated by job and rank. Set
`PRIMING_LOCAL_TMP` to another **node-local** writable disk if needed. Avoid a
network filesystem for compiler temporary files: loaded libraries can leave
open-file placeholders that make Triton's cleanup fail. This setting is
independent of the shared scratch location used for checkpoints and bundles.
