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

## Environment

Use a dedicated Python 3.11 environment with CUDA PyTorch. The copied
`requirements.txt` is the source project's full dependency list;
`requirements-overlay.txt` records its verified cluster overlay. Keep this
environment separate from Priming's ImageNet environment: the source T2I overlay
pins Transformers 4.40.2, while Priming pins 5.12.1. GenEval and DPG-Bench need
their detector/VQA assets and additional dependencies from the full list.

Post-training also requires the external DistMatch package. Set
`DISTMATCH_SOURCE` to its repository root, its `src/` directory, or the
`src/distmatch/` package directory. The package is not copied into Priming.

```bash
export T2I_STORAGE_ROOT=/path/to/t2i-storage
export DISTMATCH_SOURCE=/path/to/distmatch
python text_to_image/main.py --help
```

## Training

The B/16 reconstruction recipes use CC12M tensor chunks for pretraining and
the 120K fine-tuning mixture. `tools/` contains the source preparation and
verification scripts. The default B/32 configs are also included. Prepare data
and set `T2I_STORAGE_ROOT` before launching an eight-GPU job.

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
source run. For a new storage root, prepare `data/cc12m_tensor_chunks/` and
`data/finetune_tensor_chunks_120k/` with the copied tools. The WebDataset
materialization tools use `configs/finetune_data.yml` for their source layout
and sampling weights; ordinary training reads the finished tensor chunks.

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
PYTHONPATH="$PWD/text_to_image" DISTMATCH_SOURCE=/path/to/distmatch \
  python -m pytest -q text_to_image/tests/test_diffusion.py \
  text_to_image/tests/test_mean_flow.py \
  text_to_image/tests/test_distmatch_training.py \
  text_to_image/tests/test_generate_benchmark_images.py
```

This port has not rerun full GPU training or benchmark scoring. Those results
must be recorded separately before claiming reproduction of the T2I paper
figures.
