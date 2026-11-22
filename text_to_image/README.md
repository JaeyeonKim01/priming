# Text-to-image

This directory ports the local `~/minit2i-torch` working tree into Priming. It
contains MiniT2I reconstruction training, one-step SIM/DistMatch post-training,
native checkpoint sampling, and GenEval/DPG-Bench/FID entry points. The port keeps
the source `mini_t2i` package and its training math. See `source_manifest.json`
for a file-by-file source and destination hash record. Source MIT and GenEval
license notices are included.

All commands below run from the Priming repository root. Datasets, checkpoints,
pretrained encoders, benchmark assets, caches, and outputs stay outside this
directory. Set `T2I_STORAGE_ROOT` to a shared directory containing `data/`,
`checkpoints/`, `assets/`, and `output/`. Without it, paths default to
`outputs/t2i/` under the Priming checkout.

## Starting from a fresh clone

The model has a data-free CPU check. After the root [environment setup](../docs/setup.md),
run:

```bash
python text_to_image/smoke.py
```

This checks a tiny MM-JiT forward pass and one-step backward pass. It does not
generate a trained image. It needs PyTorch but no dataset, checkpoint, text
encoder download, or DistMatch package.

For full T2I work, prepare the following external inputs:

| Task | Required external inputs |
| --- | --- |
| Native checkpoint sampling | A compatible MiniT2I training or one-step checkpoint and FLAN-T5-Large weights |
| Reconstruction training | CC12M tensor chunks or the 120K mixture, plus FLAN-T5-Large weights |
| One-step post-training | A reconstruction checkpoint, the 120K mixture, and frozen SIM encoder weights |
| GenEval / DPG-Bench | Generated images, benchmark prompts, detector/VQA weights, and their optional Python dependencies |

The released Priming T2I checkpoints below support inference with this code.
Training still requires external data and frozen feature encoder weights;
DistMatch 0.6.0 is included under [`external/distmatch`](../external/distmatch/). The
[MiniT2I model card](https://huggingface.co/MiniT2I/MiniT2I) documents separate
multi-step baseline inference with Diffusers.

## Released checkpoints

Two native MM-JiT-B/16 checkpoints for 512×512 generation are hosted at
[jaykim1102/priming](https://huggingface.co/jaykim1102/priming/tree/main/t2i).

| Checkpoint | Completed training |
| --- | --- |
| [Pre-OT](https://huggingface.co/jaykim1102/priming/blob/main/t2i/priming_b_lognorm/pre_ot/checkpoint.pt) | 250k CC12M reconstruction updates + 40k curated reconstruction updates |
| [Post-OT](https://huggingface.co/jaykim1102/priming/blob/main/t2i/priming_b_lognorm/post_ot_20k/checkpoint.pt) | The same initializer + 20k distributional OT updates |

Both files contain online (`model`) and EMA weights plus architecture and
sampling config. They support inference and initialization of new training
runs; optimizer and RNG states are omitted. The pre-OT online weights are the
exact initializer for the post-OT run. Its stored step counter is 40k within
the second reconstruction phase, or 290k reconstruction updates in total.
File sizes, SHA256 hashes, and the pinned Hub revision are in
[`checkpoints.json`](checkpoints.json).

After installing the T2I environment below, run from the Priming repository root:

```bash
hf download jaykim1102/priming \
  --revision 3601fbd3babf5f9f83656490a8188b2b5e9e8818 \
  --include 't2i/*' --local-dir ./outputs/t2i/weights

python text_to_image/sample.py \
  --checkpoint ./outputs/t2i/weights/t2i/priming_b_lognorm/post_ot_20k/checkpoint.pt \
  --weights model --steps 1 --cfg-scale 1 --attention-impl sdpa --amp-dtype bf16 \
  --prompt 'A red panda wearing a blue baseball cap.' \
  --seed 123 --output-dir ./outputs/t2i/post_ot_sample
```

Sampling requires CUDA and the separately downloaded `google/flan-t5-large`
text encoder. Explicitly select `--weights model` for the reported online
results; the sampling script otherwise defaults to EMA. To inspect the
initializer with the same settings, replace `post_ot_20k` with `pre_ot` and
use a different output directory. The initializer's one-step samples do not
represent the finished generator's quality. For new OT training, use the
pre-OT file as `--init-from` with `init_weights: model`.

The extended 20k post-OT run scores **0.8280 GenEval** and **82.71 DPG-Bench**
with online weights, one network evaluation, CFG=1, and noise scale 2.
EMA scores are 0.8257 and 82.59, respectively. These are the extended run's
results, distinct from the paper's original checkpoint scores. See the
[HF release guide](https://huggingface.co/jaykim1102/priming/blob/main/t2i/README.md)
for evaluation details and export compatibility information.

## Environment

Use a dedicated Python 3.11 environment with CUDA PyTorch. The copied
`requirements.txt` is the source project's full dependency list;
`requirements-overlay.txt` records its historical cluster overlay.
`requirements-train.txt` records the core versions used for this port's CPU
checks. Dataset preparation has separate pins in `requirements-data.txt` and
the [dataset guide](DATASETS.md). Install a CUDA-compatible PyTorch 2.6.0 and
torchvision 0.21.0 first,
then the training requirements in that environment. Keep this
environment separate from Priming's ImageNet environment: the source T2I overlay
pins Transformers 4.40.2, while Priming pins 5.12.1. GenEval and DPG-Bench need
their detector/VQA assets and additional dependencies from the full list.

```bash
python -m pip install -r text_to_image/requirements-train.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Install `pytorch-fid` if using a recipe with scheduled FID evaluation. Diffusers
Hub inference should use its own compatible dependency set; the historical T2I
overlay cannot import the Diffusers 0.40.0 available in this workspace because
their Hugging Face Hub APIs differ.

Post-training loads the bundled DistMatch package automatically. To select
another compatible copy, set `DISTMATCH_SOURCE` to its repository root, its
`src/` directory, or the `src/distmatch/` package directory.

```bash
export T2I_STORAGE_ROOT=/path/to/t2i-storage
python text_to_image/main.py --help
```

## Training

The B/16 reconstruction recipes use CC12M tensor chunks for pretraining and
the 120K fine-tuning mixture. The [dataset guide](DATASETS.md) gives the public
sources, download and conversion commands, storage layout, and full verification
steps for a fresh clone. The default B/32 configs are also included. Prepare
data and set `T2I_STORAGE_ROOT` before launching an eight-GPU job.

On this machine, the prepared datasets remain in the original MiniT2I scratch
directory. Point the port at them and check the completion records and files:

```bash
export T2I_STORAGE_ROOT="$HOME/minit2i-torch/netscratch"
python text_to_image/check_data.py
```

The current CC12M rebuild has 6,933,021 examples in 6,771 verified chunks;
the materialized fine-tuning mix has 120,000 examples in 118 verified chunks.
Both B/16 loaders produced a sample batch in this checkout. A fresh CC12M URL
download can produce a different surviving image count from the historical
source run. The WebDataset materialization tools use
`configs/finetune_data.yml` for their source layout and sampling weights;
ordinary training reads the finished tensor chunks.

```bash
torchrun --standalone --nproc_per_node=8 text_to_image/main.py train \
  --config-file text_to_image/configs/pretrain_b16.yml
torchrun --standalone --nproc_per_node=8 text_to_image/main.py train \
  --config-file text_to_image/configs/finetune_b16.yml \
  --resume-from /path/to/pretrain/checkpoint.pt --no-auto-resume
```

For one-step post-training, choose a source recipe and supply the matching
reconstruction checkpoint. `--dry-run` checks checkpoint and data metadata
without loading the model or encoders. The included `finetune_distmatch.yml`
is the source's B/16 marginal MMD recipe; the conditional Sinkhorn B/16 and L/16
recipes are also included. `n_c` is per rank, so changing GPU count changes the
global caption population.

```bash
python text_to_image/main.py train-distmatch \
  --config text_to_image/configs/finetune_distmatch.yml \
  --init-from /path/to/reconstruction/checkpoint.pt --dry-run
torchrun --standalone --nproc_per_node=8 text_to_image/main.py train-distmatch \
  --config text_to_image/configs/finetune_distmatch.yml \
  --init-from /path/to/reconstruction/checkpoint.pt \
  --output-dir /path/to/new/posttrain/run
```

The source's full-size no-replay B/16 attempt ran out of H200 memory. Its
four-GPU smoke test checked replay, finite gradients, checkpointing, and resume;
it did not establish full-run quality. Size a new run with a bounded GPU check
before relying on a long training job.

## Reproducing the released 20k OT run

[`configs/post_ot_20k.yaml`](configs/post_ot_20k.yaml) records the settings
from the released run's saved 4k config and 8k/20k resume records. It starts
from the released **pre-OT online weights**, with a fresh OT optimizer and EMA.
Architecture, 512×512 resolution, FLAN-T5-Large, and noise scale 2 are read
from that checkpoint. Prepare the same 120K mixture using [DATASETS.md](DATASETS.md).

The original run used **16 H200 GPUs**, with `n_c: 4` captions per rank and
`b: 16` generated images per caption in each of the live and detached
populations: 64 captions, 64 real images, and 1,024 images in **each** generated
population globally. Matching is conditional and rank-local. Changing rank
count changes these global populations; keep 16 ranks to match this recipe.
Generator replay uses chunks of 32 images per rank.

Install the [T2I environment](#environment). The saved run records PyTorch
2.6.0+cu124, Transformers 4.40.2, and **timm 1.0.27**; the general port
requirements pin a later timm, so use `python -m pip install timm==1.0.27`
to match this run. DistMatch is bundled in `external/distmatch` and loads
automatically when `DISTMATCH_SOURCE` is unset. SIM uses the named pretrained
SigLIP SO400M and MAE-L weights plus TF-FID Inception. The historical Inception
file SHA256 is `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`;
`sim_weights: {}` uses the same public pretrained sources through cache/download.

```bash
export T2I_STORAGE_ROOT=/absolute/path/to/t2i-storage
hf download jaykim1102/priming \
  --revision 3601fbd3babf5f9f83656490a8188b2b5e9e8818 \
  --include 't2i/priming_b_lognorm/pre_ot/*' \
  --local-dir "$T2I_STORAGE_ROOT/checkpoints"

WORLD_SIZE=16 python text_to_image/main.py train-distmatch \
  --config text_to_image/configs/post_ot_20k.yaml --dry-run

# Example for one 8-GPU node in a two-node job; run on BOTH nodes.
# Set NODE_RANK to 0/1 and MASTER_ADDR to the address of node 0.
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port=29500 \
  text_to_image/main.py train-distmatch \
  --config text_to_image/configs/post_ot_20k.yaml
```

The historical job ran 0→4k, resumed 4k→8k, then resumed 8k→20k with
unchanged matching and optimizer settings. LR stays at `1e-5` with no warmup
or decay; `num_steps` is the total target, not an additional update count.
The YAML supports a single 20k run. To follow the original stage boundaries,
add `--num-steps 4000` to the first launch, then launch with
`--resume-from /path/to/run/checkpoints/checkpoint_004000.pt --num-steps 8000`,
and finally `--resume-from /path/to/run/checkpoints/checkpoint_008000.pt`
(the YAML already targets 20k). Retain the same world size, data and encoder
identities when resuming. The Hub exports omit optimizer/RNG state: use `pre_ot`
as initialization, and your own full trainer checkpoints for resumption.

This reproduces the recorded training recipe; a fresh run is not guaranteed
bit-identical across datasets, hardware or library versions. Evaluate **online**
weights (`--weights model`), one step, CFG=1, noise scale 2 to match the released
20k results (GenEval 0.8280; DPG-Bench 82.71).

## Sampling and evaluation

`sample.py` uses the source benchmark sampler. It reads the architecture from a
native training checkpoint. A DistMatch checkpoint uses one network call,
`steps=1`, and CFG scale 1. A reconstruction checkpoint uses the sampling
settings recorded in its config unless overridden.

```bash
python text_to_image/sample.py \
  --checkpoint /path/to/posttrain/checkpoint.pt \
  --prompt 'A red panda wearing a blue baseball cap.' \
  --output-dir outputs/t2i/example

python text_to_image/main.py eval \
  --config-file text_to_image/configs/eval_one_step_b16.yml --dry-run
```

Edit the one-step evaluation config's `checkpoint` and `train_config` for the
actual model variant. GenEval and DPG-Bench also require prompt metadata and
their evaluator assets under `T2I_STORAGE_ROOT/assets/`. The source's
`eval_hf_b16.yml` and `eval_hf_l16.yml` remain available for released
multi-step MiniT2I comparison checkpoints.

## Verification

The copied CPU tests cover loss direction, one-step gradients, replay,
checkpoint resume, and benchmark sampling. The [verification record](verification.md)
lists the checks run for this port. From a T2I environment:

```bash
PYTHONPATH="$PWD/text_to_image" \
  python -m pytest -q text_to_image/tests/test_diffusion.py \
  text_to_image/tests/test_mean_flow.py \
  text_to_image/tests/test_distmatch_training.py \
  text_to_image/tests/test_generate_benchmark_images.py
```

This port has not rerun full GPU training or benchmark scoring. Those results
must be recorded separately before claiming reproduction of the T2I paper
figures.
