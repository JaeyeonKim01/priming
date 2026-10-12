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

## Getting started

Clone this repository as `Priming`; all commands below start there.

```bash
git clone https://github.com/JaeyeonKim01/priming.git Priming
cd Priming
bash scripts/create_env.sh
conda activate priming
export PYTHONNOUSERSITE=1
python scripts/check_environment.py
```

- [Environment setup](docs/setup.md): pinned environment, GPU execution, and external storage.
- [Data and model assets](docs/assets.md): processed ImageNet, [encoder configurations and paths](docs/assets.md#encoder-configuration-and-paths), OT calibration, and precomputed evaluation statistics.
- [Checkpoint downloads](docs/checkpoints.md): download and verify the released models. Sampling needs only a generator checkpoint.
- [Workflow and checks](docs/validation.md): end-to-end training, startup checks, and numerical tests.

ImageNet pretraining and post-training require **eight GPUs** and approximately
**259 GB** for the prepared dataset, plus model assets and training outputs.
B training has been checked on H100 80 GB; use H200s for L post-training.
Accept the ImageNet dataset terms and prepare the assets before launching.
Follow the [two-update training check](docs/validation.md#check-the-training-setup-before-a-full-run)
to test pretraining → your saved checkpoint → post-training on your setup.

## Checkpoints

Four model/epoch combinations, each available before and after post-training.
Epochs refer to reconstruction pretraining; post-training adds 25,000 OT updates.

| Model | Pretraining epochs | Pretrained | Post-trained |
|---|---:|---|---|
| Priming-B | 80 | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/pretraining/priming_b_80.pth) | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/priming_b_80.pth) |
| Priming-B | 160 | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/pretraining/priming_b_160.pth) | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/priming_b_160.pth) |
| Priming-L | 80 | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/pretraining/priming_l_80.pth) | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/priming_l_80.pth) |
| Priming-L | 160 | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/pretraining/priming_l_160.pth) | [View checkpoint](https://huggingface.co/jaykim1102/priming/blob/main/checkpoints/priming_l_160.pth) |

All eight files are hosted at [jaykim1102/priming](https://huggingface.co/jaykim1102/priming).
See the [download guide](docs/checkpoints.md) and the [checked-in manifests](old/checkpoints).
Pretraining checkpoints contain online and EMA weights; post-training initializes
from online weights, and final evaluation uses post-training EMA 0.999.

### Sampling from our models

[![Figure 2: One-step samples from Priming Models across class-conditional image generation, text-to-image generation, and world modeling.](docs/figures/figure2.png)](docs/figures/figure2.pdf)

Download one checkpoint and generate samples on a single GPU:

```bash
hf auth login
hf download jaykim1102/priming checkpoints/priming_b_160.pth --local-dir .
python scripts/sample.py --model priming_b_160 --num-images 16 \
  --class-ids 207 --output-dir outputs/samples/b160
```

Saves individual PNGs and a `samples.json` record. `--class-ids` accepts ImageNet
IDs 0–999; omit it to cycle through classes. B/L and 80/160 checkpoints are
supported. Use `--stage pretraining` for pretrained weights after downloading
the corresponding file under `checkpoints/pretraining/`.
See [sampling options](docs/checkpoints.md#sampling-options).

### Evaluation: ImageNet

Evaluate the four released Priming models with **FID** and **FDr⁵**. Download the
selected generator checkpoint first, as above, then prepare the evaluation
encoders and precomputed reference statistics. No ImageNet images are needed.
Start with a 1,000-image pilot on four GPUs:

```bash
python scripts/download_assets.py --group evaluation
torchrun --standalone --nproc_per_node=4 eval.py \
  --model priming_b_160 --num-images 1000 --seeds 1 --batch 32 \
  --output-dir outputs/eval/pilot_b160
```

For the paper protocol, use `--num-images 50000 --seeds 1 2 3` and a new output
directory. Preserve four ranks and batch 32: changing either changes the sample
stream. Official FID uses FP32 Inception on rounded uint8 images and the JiT
reference; FDr⁵ uses five frozen encoders on float images, excluding ConvNeXt.

[Evaluation guide and options](docs/eval.md)

<details>
<summary><strong>ImageNet evaluation results</strong></summary>

Each row is the mean ± sample SD over three sampling seeds of one checkpoint,
with 50,000 images per seed; these are not independent training runs.

| Model | FID ↓ | FDr⁵ ↓ |
|---|---:|---:|
| Priming-B-80 | 1.4683 ± 0.0128 | 12.7089 ± 0.0188 |
| Priming-B-160 | 1.3841 ± 0.0130 | 11.2286 ± 0.0157 |
| Priming-L-80 | 1.2695 ± 0.0128 | 8.9430 ± 0.0134 |
| Priming-L-160 | 1.1871 ± 0.0147 | 7.1048 ± 0.0196 |

These results use released post-training checkpoints.

</details>

## Experiment scope

This release covers ImageNet-256: B/L sampling at 80/160 pretraining epochs,
three reconstruction recipes, B/L-160 OT post-training, and B-160 SIM,
Laplacian MMD, and energy ablations. Text-to-image and world modeling code will
be ready in two weeks (October 25, 2026); see [News](#news). Comparison-baseline
samplers and the old dashboard remain available in the previous Git revision.

Post-training initializes from released pretrained online weights with a fresh
optimizer and inherits the pretrained EMA shadow. Use `--init-checkpoint PATH`
for your own compatible perceptual pretraining weights. Evaluate a terminal
post-training checkpoint with `eval.py --checkpoint PATH`, or select
`--stage pretraining` / `--weights online` for the additional evaluation modes.
The paper results above use post-training EMA 0.999.

Use eight H200 GPUs for L post-training. The original L runs exceeded
85 GiB allocated per GPU, above H100 80 GB capacity.

## ImageNet experiments

<details>
<summary><strong>Click to expand ImageNet pretraining and post-training</strong></summary>

### Training 1: Reconstruction pretraining

Train at fixed corruption t=0.8 with adaptive pixel MSE, optionally adding
LPIPS and ConvNeXt perceptual losses. Recipes cover **B with/without perceptual
loss** and **L with perceptual loss**, using Muon and EMA 0.9999. The 160-epoch
runs retain both 80- and 160-epoch checkpoints.

```bash
python pretrain.py --config configs/pretrain/b160_perceptual.yaml --check-config
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/b160_perceptual.yaml --output outputs/pretrain/b160
```

[Pretraining guide](docs/pretrain.md) ·
[Production configs](configs/pretrain) · [Source code](pretrain.py)

### Training 2: Distributional post-training

Apply feature-wise optimal transport in ResNet-MAE and ConvNeXt feature spaces.
The **B/L-160** workflows run 25,000 AdamW updates with EMA 0.999. Scheduled
online/EMA evaluations use FDr⁶, Inception Score, and precision/recall; use the
standalone evaluator above for official final FID/FDr⁵.

```bash
torchrun --standalone --nproc_per_node=8 posttrain.py \
  --config configs/posttrain/b160.yaml --output outputs/posttrain/b160
```

[Post-training and resume guide](docs/posttrain.md) ·
[Production configs](configs/posttrain) · [Source code](posttrain.py)

- **Ablation study 1: SIM feature-space training.** Use SigLIP, Inception, and ViT-MAE features with [b160_sim.yaml](configs/posttrain/b160_sim.yaml).
- **Ablation study 2: Other distributional losses.** Replace OT with [Laplacian MMD](configs/posttrain/b160_laplacian.yaml) or [energy distance](configs/posttrain/b160_energy.yaml).

Prepare the dataset and encoders using the [asset guide](docs/assets.md) before
training. Start with the [two-update training check](docs/validation.md#check-the-training-setup-before-a-full-run)
to verify pretraining → post-training on your setup. See the stage guides for
custom initialization and checkpoint recovery.

</details>

## Code layout

`pretrain.py`, `posttrain.py`, and `eval.py` share the implementations in
`models/`, `losses/`, `metrics/`, `data/`, and `utils/`. Each recipe inherits the
`_base.yaml` beside it. The matcher receives generated and real samples and
returns a loss; the training loop owns sampling, optimizer updates, and recovery.

Dependency pins, asset manifests, and released inference configurations live
under [`old/`](old/README.md) and are still required by the active tools.
Downloaded generator weights retain their published `checkpoints/` paths.

See [validation instructions](docs/validation.md) to run the numerical comparison
suite against a checkout of the previous implementation.

## News

- **October 11, 2026:** Text-to-image (T2I) and world modeling code will be ready
  in two weeks, on **October 25, 2026**.

## Citation

If you use Priming in your research, please cite:

```bibtex
@misc{park2026priming,
  title   = {Are We Overthinking One-Step Generation?},
  author  = {Park, Woobin and Kim, Jaeyeon and Du, Yilun and Chen, Sitan and Han, Bohyung},
  year    = {2026}
}
```

The arXiv identifier and link will be added when the preprint is available.
