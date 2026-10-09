# Environment and storage setup

Use Linux x86-64, Python 3.11, and the pinned PyTorch 2.6.0/CUDA 12.4 environment.
The [environment lock](../old/environment/requirements-linux-cu124.txt) retains the
production pins for the retained workflows and their dependencies. Packages
used by removed baselines, dashboards, and experiment tracking are excluded.
The CPU tests use Python's standard `unittest` module.

Install Conda before running the setup script. GPU training nodes need an
NVIDIA driver compatible with CUDA 12.4 and a working C/C++ toolchain
(`gcc` and `g++`) for compiled training; these system prerequisites are not
installed by the Python environment file.

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

The [eight-GPU audit](imagenet_refactor_audit.md#eight-gpu-training-checks)
passed short B/L perceptual pretraining and B post-training runs on H100 80 GB.
The unchanged L post-training recipe ran out of GPU memory on that hardware;
the original L-80/L-160 production runs used eight H200s and peaked above
85 GiB allocated per GPU. Use H200s to match that training hardware.

## Assets outside the checkout

Choose a shared storage directory and install the environment/assets there:

```bash
export PRIMING_STORAGE=/scratch/priming
mkdir -p "$PRIMING_STORAGE"/{conda-pkgs,pip-cache,assets/huggingface,assets/torch}
export CONDA_PKGS_DIRS="$PRIMING_STORAGE/conda-pkgs"
export PIP_CACHE_DIR="$PRIMING_STORAGE/pip-cache"
bash scripts/create_env.sh --prefix "$PRIMING_STORAGE/env"
conda activate "$PRIMING_STORAGE/env"
export HF_HOME="$PRIMING_STORAGE/assets/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PRIMING_STORAGE/assets/torch"
hf auth login
python scripts/download_assets.py --group dataset --root "$PRIMING_STORAGE"
python scripts/download_assets.py --group pretraining --root "$PRIMING_STORAGE"
```

The download helper writes `data/` and `assets/` under `--root`; environment
cache overrides do not redirect its destinations. Set `HF_HOME` before logging
in, as above: the Hub client also stores its login token there. If you change
`HF_HOME` after logging in, run `hf auth login` again under the new setting.
Keep these exports when running the remaining download and training commands.

Pretraining accepts `--data-path`, `--hf-home`, `--torch-home`, and
`--lpips-weights`. Post-training accepts `--data-path`, `--encoder-cache`,
`--init-checkpoint`, `--hf-home`, and `--torch-home`; its recipe specifies the
scheduled evaluation reference locations. Standalone evaluation accepts
`--checkpoint`, `--stats-dir`, `--fid-reference`, and `--inception-weights`.
All selected paths may be absolute. Dataset paths select the **parent** of
`imagenet-latents-images`. Output paths may be on shared storage.

For example, pretraining with externally downloaded perceptual assets uses:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml \
  --data-path "$PRIMING_STORAGE/data/imagenet" \
  --hf-home "$PRIMING_STORAGE/assets/huggingface" \
  --torch-home "$PRIMING_STORAGE/assets/torch" \
  --lpips-weights "$PRIMING_STORAGE/assets/lpips/vgg.pth" \
  --output "$PRIMING_STORAGE/outputs/pretrain/my_b"
```

For post-training, download the `posttraining` and `conversion` groups with
the same `--root`, then run `scripts/convert_ot_encoders.py --root
"$PRIMING_STORAGE"` as described in the [asset guide](assets.md#post-training-encoders-and-ot-calibration).
The scheduled evaluation paths are recipe fields, so create
`configs/posttrain/b160_external.yaml` beside `_base.yaml`, replacing
`/scratch/priming` with your actual storage root:

```yaml
_base_: _base.yaml
data:
  data_path: /scratch/priming/data/imagenet
  encoder_cache: /scratch/priming/assets/ot/representations
evaluation:
  stats_dir: /scratch/priming/assets/fid_stats
  prc_reference: /scratch/priming/data/prc/VIRTUAL_imagenet256_labeled_inception.pt
```

```bash
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160_external.yaml \
  --hf-home "$PRIMING_STORAGE/assets/huggingface" \
  --torch-home "$PRIMING_STORAGE/assets/torch" \
  --init-checkpoint "$PRIMING_STORAGE/outputs/pretrain/my_b/checkpoints/latest.pth" \
  --output "$PRIMING_STORAGE/outputs/posttrain/my_b"
```

The source may instead be a downloaded released pretrained checkpoint. For L,
use its pretraining recipe and checkpoint and add `model: {model: OST_L}` to
the external post-training recipe. Setting the Hub/Torch cache roots alone
does not relocate LPIPS, OT calibration, or the scheduled evaluation references.

Training compiler caches and temporary libraries use node-local `/tmp`,
separated by run, job, and rank. Set `PRIMING_LOCAL_TMP` to another node-local
writable disk if needed. Keep checkpoints and assets on shared storage.
Offline execution loads already verified local encoder files.

## Training across multiple nodes

The `--standalone --nproc_per_node=8` examples require eight GPUs on one node.
For Slurm nodes with four GPUs each, save the following as `train.sbatch` in
the repository root. Submit from that root with the shared environment active;
the checkout, assets, environment, and outputs must be visible on every node.
Select your site's GPU partition and account when submitting (H200s for L
post-training).

```bash
#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONNOUSERSITE=1
export PRIMING_PYTHON="$(command -v python)"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export PRIMING_GPUS_PER_NODE=4
srun --kill-on-bad-exit=1 bash -c \
  'exec "$PRIMING_PYTHON" -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$PRIMING_GPUS_PER_NODE" \
    --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" "$@"' bash "$@"
```

```bash
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT train.sbatch \
  pretrain.py --config configs/pretrain/b160_perceptual.yaml \
  --output outputs/pretrain/check_b --smoke-steps 2
# After pretraining completes:
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT train.sbatch \
  posttrain.py --config configs/posttrain/b160.yaml \
  --init-checkpoint outputs/pretrain/check_b/checkpoints/latest.pth \
  --output outputs/posttrain/check_b --smoke-steps 2
```

For four nodes with two GPUs each, change `--nodes` to 4, `--gpus-per-node`
and `PRIMING_GPUS_PER_NODE` to 2, and CPUs per task to 16. The total remains
eight ranks. Add the external-storage path arguments above when needed.
The one-hour limit is for startup checks; request a suitable time limit and
use the stage guides' checkpoint recovery instructions for full training.

See [asset preparation](assets.md), [pretraining](pretrain.md),
[post-training](posttrain.md), and [evaluation](eval.md).
