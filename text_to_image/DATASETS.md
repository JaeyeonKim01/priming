# Preparing the text-to-image training datasets

Run these commands from the Priming repository root on Linux with Python 3.11.
They build the `local_folder` tensor chunks expected by
`configs/pretrain_b16.yml` and `configs/finetune_b16.yml`. No checkpoint or
DistMatch package is needed for data preparation. The datasets are external to
the Git checkout; review each source's terms before downloading it.

The finished CC12M chunks in the local rebuild use 5.48 TB and the 120K chunks
use 95 GB. Allow about **8 TiB** for raw downloads, the Hugging Face cache,
intermediate WebDataset shards, and finished chunks together. These commands
use `T2I_STORAGE_ROOT/data/`; keep the Hugging Face cache while the 120K staging
directory contains symlinks into it. Data conversion runs on CPUs; model
training uses a separate GPU environment.

## 1. Create a data-preparation environment and storage root

The pins in `requirements-data.txt` come from the source checkout's dataset
audit. They differ from the training environment's Transformers and Hugging
Face Hub versions, so use a separate virtual environment. For Linux CPU
preparation, install the [PyTorch 2.6 pair](https://pytorch.org/get-started/previous-versions/)
first:

```bash
python3.11 -m venv .venv-t2i-data
source .venv-t2i-data/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r text_to_image/requirements-data.txt

export T2I_STORAGE_ROOT=/absolute/path/to/t2i-storage
export T2I_DATA_ROOT="$T2I_STORAGE_ROOT/data"
export HF_HOME="$T2I_STORAGE_ROOT/hf_home"
mkdir -p "$T2I_DATA_ROOT/cc12m_prep/meta" \
  "$T2I_DATA_ROOT/cc12m_prep/parquet" \
  "$T2I_DATA_ROOT/cc12m_prep/wds" \
  "$T2I_DATA_ROOT/cc12m_tensor_chunks" \
  "$T2I_DATA_ROOT/finetune_wds" \
  "$T2I_DATA_ROOT/finetune_tensor_chunks_120k"
```

Use a new, empty storage root for a full rebuild. The CC12M tools can resume
interrupted work, but changing inputs or options partway through a run can
change the output. `T2I_DATA_ROOT` is shell shorthand in this guide; the model
code reads `T2I_STORAGE_ROOT` directly.

## 2. Build CC12M pretraining chunks

The source recipe uses URL and recaption metadata from
[`CaptionEmporium/conceptual-captions-cc12m-llavanext`](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext).
The source audit recorded revision
`1c24bdba3f81257082612654ba72fe18a83c7a37`. Download it and convert
successful URL/caption rows to Parquet:

```bash
hf download CaptionEmporium/conceptual-captions-cc12m-llavanext \
  train.jsonl.gz --repo-type dataset \
  --revision 1c24bdba3f81257082612654ba72fe18a83c7a37 \
  --local-dir "$T2I_DATA_ROOT/cc12m_prep/meta"

python text_to_image/tools/cc12m_jsonl_to_parquet.py \
  --jsonl "$T2I_DATA_ROOT/cc12m_prep/meta/train.jsonl.gz" \
  --out "$T2I_DATA_ROOT/cc12m_prep/parquet/metadata.parquet.tmp"
mv "$T2I_DATA_ROOT/cc12m_prep/parquet/metadata.parquet.tmp" \
  "$T2I_DATA_ROOT/cc12m_prep/parquet/metadata.parquet"
```

The Parquet directory must contain only the completed `metadata.parquet` before
downloading. `img2dataset` writes 10,000-input-row WebDataset shards and
per-shard outcome records. Image URLs can fail or change over time, so the
number of successfully downloaded images is not a fixed target. If Parquet
conversion stops early, remove its `.tmp` file before retrying.

```bash
img2dataset \
  --url_list "$T2I_DATA_ROOT/cc12m_prep/parquet" \
  --input_format parquet \
  --url_col url \
  --caption_col caption_llava \
  --output_folder "$T2I_DATA_ROOT/cc12m_prep/wds" \
  --output_format webdataset \
  --image_size 512 \
  --resize_mode center_crop \
  --encode_format jpg \
  --encode_quality 95 \
  --resize_only_if_bigger False \
  --upscale_interpolation lanczos \
  --downscale_interpolation area \
  --skip_reencode False \
  --disable_all_reencoding False \
  --processes_count 16 \
  --thread_count 64 \
  --number_sample_per_shard 10000 \
  --timeout 10 \
  --enable_wandb False \
  --retries 2 \
  --incremental_mode incremental

python text_to_image/tools/verify_cc12m_download.py \
  --parquet "$T2I_DATA_ROOT/cc12m_prep/parquet/metadata.parquet" \
  --wds "$T2I_DATA_ROOT/cc12m_prep/wds"

python text_to_image/tools/cc12m_wds_to_chunks.py \
  --wds "$T2I_DATA_ROOT/cc12m_prep/wds" \
  --out "$T2I_DATA_ROOT/cc12m_tensor_chunks" \
  --chunk-size 1024 --image-size 512 --prompt-length 256 \
  --tokenizer google/flan-t5-large --resume

python text_to_image/tools/verify_cc12m_chunks.py \
  --chunks "$T2I_DATA_ROOT/cc12m_tensor_chunks"
```

The last verifier reads every token array and checks image shapes and sample
pixels. It writes `_cc12m_chunks_verified.json`. The local October 2026
rebuild produced 6,933,021 usable examples in 6,771 chunks; an earlier source
run produced 7,081,720 examples. Record your own totals from the completion
marker and verifier instead of forcing either historical count.

## 3. Build the materialized 120K fine-tuning mix

The source groups are
[`BLIP3o/BLIP3o-60k`](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k),
[`OpenDatasets/dalle-3-dataset`](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset),
and [`FreedomIntelligence/ShareGPT-4o-Image`](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image).
The preparation helper downloads the archives and DALL-E3 Parquet files, turns
the latter into image/text tar shards, and stages the source layout. It uses
the repositories' current default revisions. The source audit recorded these
revisions, which help identify changes in upstream data:

| Source | Revision recorded in the source audit |
| --- | --- |
| BLIP3o-60K | `f7316b0aa446338ee1707484924aa59457b4bbf3` |
| DALL-E3 | `96822d5c0b1338ec4bfd04b0d5909fa070e301b6` |
| ShareGPT-4o-Image | `f9bcb9494e562f3b6d46585bda0e7f0f0d6ec43f` |

The saved caption IDs use `google/flan-t5-large`; the ShareGPT4o prompt-length
filter uses `google/flan-t5-base`. The source audit recorded tokenizer revisions
`0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` and
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, respectively. The current
preparation tools resolve those model IDs through the Hugging Face cache.

```bash
python text_to_image/tools/prepare_120k_mix_layout.py \
  --out "$T2I_DATA_ROOT/finetune_wds" \
  --cache-dir "$T2I_STORAGE_ROOT/hf_cache"

python text_to_image/tools/finetune_wds_to_chunks.py \
  --config-file text_to_image/configs/finetune_data.yml \
  --out "$T2I_DATA_ROOT/finetune_tensor_chunks_120k" \
  --total-samples 120000 --chunk-size 1024 \
  --loader-batch-size 64 --num-workers 8 \
  --shuffle-buffer 32 --resume

python text_to_image/tools/verify_finetune_chunks.py \
  --chunks "$T2I_DATA_ROOT/finetune_tensor_chunks_120k" \
  --config-file text_to_image/configs/finetune_data.yml \
  --expected-samples 120000
```

`finetune_data.yml` resolves the staged source directory from
`T2I_STORAGE_ROOT` and sets the three source weights to `[0.06, 0.016, 0.04]`.
The output is a **weighted stream of 120,000 examples**, with repeated source
examples possible. It is not the concatenation of three unique datasets. Keep
the source tars and cache until conversion and full verification finish. The
verifier writes `_finetune_chunks_verified.json` and checks the mixture's
source counts, tensors, and captions. Resuming an interrupted 120K conversion
does not restore the loader's random state exactly, so rebuild from an empty
output directory when the sampled sequence itself matters.

## 4. Check the finished layout and use it for training

```bash
python text_to_image/check_data.py --storage-root "$T2I_STORAGE_ROOT"
```

Both datasets should be reported under these directories:

```text
$T2I_STORAGE_ROOT/data/cc12m_tensor_chunks/chunk_*.pt
$T2I_STORAGE_ROOT/data/finetune_tensor_chunks_120k/chunk_*.pt
```

Run model training in the separate T2I training environment described in the
[T2I README](README.md#environment), with the same `T2I_STORAGE_ROOT`. The B/16
pretraining and fine-tuning configs already point to these directories. The
fine-tuning verifier requires the staged raw source tars, but ordinary
`local_folder` training reads the finished tensor chunks.

This procedure reconstructs the source preparation pipeline; it does not
promise byte-identical data. CC12M URL responses and download completion order
vary, and the 120K staging helper currently follows upstream default
revisions. Reproducing an exact existing dataset requires copying its tensor
chunks and verification records.
