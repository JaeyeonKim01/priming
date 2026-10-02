# Priming Models

Code, checkpoints, and reproduction records for Priming Models on ImageNet-256.
The source code and historical run names use **OST**.

**Pipeline:** reconstruction pretraining → distributional post-training → evaluation.

## 1. Checkpoints

Four model/epoch combinations, each available before and after post-training.
Epochs refer to reconstruction pretraining; post-training adds 25,000 OT updates.

| Model | Pretraining epochs | Before post-training | After post-training |
|---|---:|---|---|
| Priming-B | 80 | [Local checkpoint](../fdloss_work_dirs/OST/OST_B_t08_e160_perceptual_seed1_20260912/checkpoints/step_0100080.pth) | [Local checkpoint](checkpoints/priming_b_80.pth) |
| Priming-B | 160 | [Local checkpoint](../fdloss_work_dirs/OST/OST_B_t08_e160_perceptual_seed1_20260912/checkpoints/step_0200160.pth) | [Local checkpoint](checkpoints/priming_b_160.pth) |
| Priming-L | 80 | [Local checkpoint](../fdloss_work_dirs/OST/OST_L_aux_unif_t08_38545843/checkpoints/step_0100080.pth) | [Local checkpoint](checkpoints/priming_l_80.pth) |
| Priming-L | 160 | [Local checkpoint](../fdloss_work_dirs/OST/OST_L_aux_unif_t08_38545843/checkpoints/step_0200160.pth) | [Local checkpoint](checkpoints/priming_l_160.pth) |

These links currently resolve in the original local workspace; weights are not
stored in Git. **Hugging Face download links are planned.**
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
