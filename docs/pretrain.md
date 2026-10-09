# Reconstruction pretraining

Prepare the [environment](setup.md) and [packed ImageNet](assets.md). Perceptual
recipes also need LPIPS, VGG16, and ConvNeXt weights.

[`data/imagenet.py`](../data/imagenet.py) contains the ImageNet datasets,
augmentation, and resumable pretraining sampler.

```bash
python scripts/download_assets.py --group dataset
python scripts/download_assets.py --group pretraining
python pretrain.py --config configs/pretrain/b160_perceptual.yaml --check-config
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml --output outputs/pretrain/b160
```

Use `b160_no_perceptual.yaml` for B without the perceptual term, or
`l160_perceptual.yaml` for L. Recipes inherit shared settings from
[`_base.yaml`](../configs/pretrain/_base.yaml); only variant differences live
in the named files. Each recipe performs 1,251 updates per epoch for
160 epochs, with global batch 1,024 (microbatch 64, accumulation two), fixed
t=0.8 reconstruction, Muon, and constant EMA 0.9999. Eight torchrun ranks are
required. Adaptive pixel loss, perceptual crop behavior, B deterministic seeds,
and L augmentation/accumulation follow the original recipe.

Paths resolve from the repository root. `--data-path` selects the parent of
`imagenet-latents-images`. `--hf-home`, `--torch-home`, and `--lpips-weights`
select prepared local assets. `--output` may be an absolute path on shared
storage. Compiler caches use node-local temporary storage; configure it with
`PRIMING_LOCAL_TMP`.

## Checkpoint recovery

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml --output outputs/pretrain/b160 \
  --resume outputs/pretrain/b160/checkpoints/latest.pth
```

Checkpoints include online weights, EMA, per-rank optimizer state, counters,
RNG state, and the recipe/source/asset identities. The 80- and 160-epoch
milestones are retained. B checkpoints preserve the stateless microbatch and
perceptual seed streams. Recovery validates the original world size and
identities. Historical L checkpoints are supported; use renewed B checkpoints
for exact B recovery.

`--smoke-steps N` stops a fresh run after N updates and saves its state.
USR1 or `PREEMPT_FLAG_FILE` requests a checkpointed stop at an update boundary.
The resulting perceptual checkpoint can initialize
[post-training](posttrain.md) with `--init-checkpoint PATH`.
