# Priming Models

Code, checkpoints, and reproduction records for Priming Models on ImageNet-256.
The source code and historical run names use **OST**.

**Pipeline:** reconstruction pretraining → distributional post-training → evaluation.

Clone this repository as `priming`; all commands and relative paths below start
there. The code is self-contained and does not import a parent OST checkout.

```bash
git clone https://github.com/JaeyeonKim01/one_step_priming.git priming
cd priming
bash scripts/create_env.sh
conda activate priming
python scripts/check_environment.py
```

[Environment and H100/H200 setup](docs/setup.md) ·
[Data and model assets](docs/assets.md) · [Checkpoint upload](docs/checkpoints.md)

## 1. Checkpoints

Four model/epoch combinations, each available before and after post-training.
Epochs refer to reconstruction pretraining; post-training adds 25,000 OT updates.

| Model | Pretraining epochs | Before post-training | After post-training |
|---|---:|---|---|
| Priming-B | 80 | [Pretrained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/pretraining/priming_b_80.pth?download=true) | [Post-trained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/priming_b_80.pth?download=true) |
| Priming-B | 160 | [Pretrained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/pretraining/priming_b_160.pth?download=true) | [Post-trained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/priming_b_160.pth?download=true) |
| Priming-L | 80 | [Pretrained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/pretraining/priming_l_80.pth?download=true) | [Post-trained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/priming_l_80.pth?download=true) |
| Priming-L | 160 | [Pretrained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/pretraining/priming_l_160.pth?download=true) | [Post-trained](https://huggingface.co/jaykim1102/priming/resolve/main/checkpoints/priming_l_160.pth?download=true) |

Weights are downloaded separately into these paths; they are not stored in Git.
All eight files are hosted at [jaykim1102/priming](https://huggingface.co/jaykim1102/priming).
The Hub repository is currently private, so downloads require an authorized
Hugging Face login. See the [download guide](docs/checkpoints.md) and the
checked-in manifests.
Pretraining checkpoints contain online and EMA weights; post-training initializes
from online weights, and final evaluation uses post-training EMA 0.999.

[Checkpoint manifest and inference configs](checkpoints/manifest.json) ·
[Pretraining source record](pretraining/provenance/release_lineage.json)

## 2. Evaluation

Evaluate the released models and comparison baselines with **FID** and **FDr⁵**.
The reproduced batch covers 11 checkpoints × 3 sampling seeds × 50,000 images;
all 33 runs passed validation and match the paper means at two decimal places.

[Evaluation guide and commands](eval/README.md) ·
[Reproduced results](eval/full_reproduction_20260930.md) ·
[Results dashboard](dashboard/index.html)

## 3. Training 1: Reconstruction pretraining

Train at fixed corruption t=0.8 with adaptive pixel MSE, optionally adding
LPIPS and ConvNeXt perceptual losses. Production recipes cover **B with/without
perceptual loss** and **L with perceptual loss**. The 160-epoch runs retain
both the 80- and 160-epoch checkpoints.

[Pretraining guide](pretraining/README.md) ·
[Production configs](pretraining/configs) ·
[Source code](pretraining/runtime)

## 4. Training 2: Distributional post-training

Apply feature-wise optimal transport in ResNet-MAE and ConvNeXt feature spaces,
starting from pretrained online weights with a fresh AdamW optimizer.
The packaged **B/L-160** workflows run 25,000 updates and include EMA,
scheduled evaluations, checkpoint recovery, and validation tools.

[Post-training guide](training/README.md) ·
[Production configs](training/configs) ·
[Source code](training/runtime)

---

Datasets, weights, and encoder assets remain external; each stage's guide lists
its setup and dependencies. [Reproduction status and provenance](REPRODUCTION_STATUS.md)
records completed checks and historical limitations. Pretraining code has been
migrated and checked without launching a new training reproduction.
