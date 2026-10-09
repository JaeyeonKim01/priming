# Standalone FID and FDr5 evaluation

Download the selected [generator checkpoint](checkpoints.md) and the pinned
[encoder/reference assets](assets.md#evaluation-download-existing-reference-statistics).
This workflow needs no ImageNet images or OT training assets.

```bash
python scripts/download_assets.py --group evaluation
hf download jaykim1102/priming checkpoints/priming_b_160.pth --local-dir .
torchrun --standalone --nproc_per_node=4 eval.py \
  --model priming_b_160 --num-images 1000 --seeds 1 --batch 32 \
  --output-dir outputs/eval/pilot_b160
```

The paper protocol uses four ranks, batch 32, 50,000 images, and seeds 1, 2, 3.
A 1,000-image pilot checks execution. Rank count and batch size affect the
sampling stream, so preserve them for metric comparisons. Single-GPU execution
is also available with `python eval.py` and the same arguments.

Model keys: `priming_b_80`, `priming_b_160`, `priming_l_80`, `priming_l_160`.
Default evaluation loads released post-training EMA 0.999, requires 25,000
completed updates, and verifies the release hash. For your own terminal
post-training checkpoint, select its matching model and add `--checkpoint PATH`.
Use `--stage pretraining` for pretraining weights (EMA 0.9999), or
`--weights online` for online weights. Both choices are recorded in the output.

## Preserved metric protocol

- Official FID: FP32 LTH14-compatible Inception on rounded uint8 images,
  TF1.x slow bilinear resize, TF32 disabled, the pinned JiT reference, and the
  original NumPy covariance/eigenspectrum formula with reference-dtype casts.
- FDr5: the same generated images in float format, frozen BF16 feature forwards,
  exact FP64 moments, and the original normalized Fréchet distances for Inception,
  DINOv2, MAE, SigLIP, and CLIP. ConvNeXt is excluded. The Inception component
  retains its separate ADM reference.

Each seed initializes Torch and NumPy with `seed*1000003+rank` after all model
construction. Classes occur in balanced blocks; ranks process interleaved full
batches. The final batch is padded before generation and trimmed afterward,
preserving the original noise consumption and image ordering.

## Outputs and paths

Each seed writes `OUTPUT/MODEL/n50000_seed1/` (or the requested count/seed):
`results.json`, official generated statistics, per-rank feature/index arrays,
first-image bytes, and per-rank image-byte hashes. Existing nonempty seed
folders are rejected before any rank writes. Results record checkpoint and
inference-config hashes, stage/weight choice, reference identity, rank count,
batch, seed, FID, and the five FDr components.

Paths resolve from the repository root. Use `--hf-home`, `--torch-home`,
`--stats-dir`, `--fid-reference`, and `--inception-weights` for prepared external
assets. The historical `PRIMING_FDR5_STATS`, `PRIMING_FID_REFERENCE`, and
`PRIMING_INCEPTION_WEIGHTS` environment overrides are also accepted. All
encoders and reference files are verified against the release manifest before
offline loading.
