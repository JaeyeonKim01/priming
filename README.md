<div align="center">

<h1>Are We Overthinking One-Step Generation?</h1>
<h2>Priming Models</h2>

<p><strong><a href="https://parkwbin.github.io/">Woobin Park</a><sup>1,*</sup>, <a href="https://jaeyeonkim01.github.io/">Jaeyeon Kim</a><sup>2,*</sup>, <a href="https://yilundu.github.io/">Yilun Du</a><sup>2</sup>, <a href="https://www.sitanchen.com/">Sitan Chen</a><sup>2</sup>, <a href="https://cv.snu.ac.kr/index.php/~bhhan/">Bohyung Han</a><sup>1</sup></strong></p>

<p><sup>1</sup> Seoul National University &nbsp;·&nbsp; <sup>2</sup> Harvard University<br>
<sup>*</sup> Equal contribution (co-first authors)</p>

<!-- Replace the placeholder href values with the paper and announcement URLs when available. -->
<p id="paper-links">
  <a href="#paper-links" title="arXiv preprint coming soon"><img src="https://img.shields.io/badge/ArXiv-Coming_soon-red" alt="ArXiv: coming soon"></a>
  <a href="#paper-links" title="X announcement coming soon"><img src="https://img.shields.io/badge/X-Coming_soon-black?logo=x&amp;logoColor=white" alt="X: coming soon"></a>
</p>

</div>

Official implementation of [Are We Overthinking One-Step Generation?](#paper-links) (arXiv coming soon).

[![Figure 1: Priming Models improve the training compute vs. FID trade-off (left); reconstruction pretraining accelerates distributional post-training (right).](docs/figures/figure1.png)](docs/figures/figure1.pdf)

## Setup

```bash
bash scripts/create_env.sh
conda activate priming
export PYTHONNOUSERSITE=1
python scripts/check_environment.py
```

Prepare [data and encoder assets](docs/assets.md), then download the
[generator checkpoints](docs/checkpoints.md). Training requires eight GPUs;
paper evaluation uses four GPUs and batch 32. Data, weights, caches, and outputs
are stored outside Git. [External storage and environment setup](docs/setup.md).

## Code layout

```text
pretrain.py          reconstruction pretraining
posttrain.py         distributional post-training
eval.py             standalone FID/FDr5 and shared scheduled evaluation
models/              shared Priming architecture and frozen encoders
losses/              reconstruction, perceptual, and distributional losses
metrics/             FID, Fréchet moments, Inception Score, precision/recall
data/                task-specific datasets, preprocessing, and sampling
utils/               optimizers, EMA, checkpoints, paths, distributed helpers
configs/             the pretraining and post-training recipes
scripts/             environment, asset, checkpoint, and sampling tools
```

The matcher follows the sample-in, loss-out design: it receives class-aligned
live samples, independent generated references, and real images. The training
script owns sampling, optimizer updates, EMA, and checkpoint recovery.

Data code is organized by task. [`data/imagenet.py`](data/imagenet.py) contains
ImageNet loaders, preprocessing, real-reference access, and class sampling.

Each recipe overrides the `_base.yaml` beside it. Configuration loading merges
the two before validation; saved checkpoints contain the complete settings.

Dependency lists, release manifests, and released inference configurations are
temporarily collected under [`old/`](old/README.md). Setup and checkpoint tools
still read these files. Downloaded weights use their published `checkpoints/`
paths or explicit external paths.

## Pretrain

```bash
python pretrain.py --config configs/pretrain/b160_perceptual.yaml --check-config
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml --output outputs/pretrain/b160
```

Recipes cover B with/without perceptual loss and L with perceptual loss.
They use reconstruction at corruption t=0.8, Muon, and EMA 0.9999, retaining
80- and 160-epoch checkpoints. [Pretraining guide](docs/pretrain.md).

## Post-train

```bash
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/b160
```

Initialize from pretrained online weights, start fresh AdamW, inherit the EMA
shadow, and use decay 0.999 for 25,000 updates. The five recipes cover B/L
full-feature OT, B SIM OT, B energy distance, and B Laplacian MMD. Both online
and EMA weights receive 50,000-image FDr6/IS/precision/recall evaluations every
1,250 updates. [Post-training and resume guide](docs/posttrain.md).

## Evaluate

```bash
python scripts/download_assets.py --group evaluation
torchrun --standalone --nproc_per_node=4 eval.py \
  --model priming_b_160 --num-images 1000 --seeds 1 --batch 32 \
  --output-dir outputs/eval/pilot_b160
```

Use 50,000 images and seeds 1, 2, 3 for the paper protocol. Official FID uses
FP32 Inception on rounded uint8 images and the JiT reference; FDr5 uses frozen
BF16 float-image features and FP64 moments. [Evaluation guide](docs/eval.md).

For your own terminal post-training checkpoint, add `--checkpoint PATH` and
select the matching B/L model key. For pretraining weights, use
`--stage pretraining`; `--weights online` selects online weights.

## Checkpoints and samples

B/L models at 80/160 pretraining epochs are available before and after
post-training at [jaykim1102/priming](https://huggingface.co/jaykim1102/priming).
The manifests under `old/checkpoints/` pin all eight weight files and inference
configurations.

```bash
hf auth login
hf download jaykim1102/priming checkpoints/priming_b_160.pth --local-dir .
python scripts/sample.py --model priming_b_160 --num-images 16 \
  --class-ids 207 --output-dir outputs/samples/b160
```

Sampling needs only the generator checkpoint. [Download and sampling options](docs/checkpoints.md).

## Verification

CPU equivalence checks read the original checkout without writing to it:

```bash
PRIMING_REFERENCE_ROOT=~/priming_old PYTHONDONTWRITEBYTECODE=1 \
  python -B -m unittest discover -s tests -q
python -B -m unittest discover -s scripts/tests -q
python scripts/check_portability.py --reference-root ~/priming_old
```

These cover numerical losses/gradients, sampling, encoder features, metric
formulas, and checkpoint continuation. Full GPU training and 50,000-image
metrics require separate validation. [Validation scope](docs/validation.md).
