# Pipeline validation

## Downloaded assets and conversion

The [asset release validation](../environment/asset_release_validation.json)
records an additional check of the public preparation workflow:

- All 517 packed ImageNet Arrow shards were hashed; image path fields were
  null and images were embedded. The train/validation counts and all training
  labels were checked against the included class indices.
- Rebuilding the three class index files produced the released SHA256 values.
  The frozen calibration's row selection and encoder source identity matched.
- Conversion from the pinned upstream MAE and ConvNeXt weights reproduced
  both released encoder files byte-for-byte.
- A new evaluation cache was populated with the download helper. A 1,000-image
  Priming-B-160 pilot on four H200 GPUs passed the saved-result audit using
  those encoders and precomputed statistics. It used no ImageNet dataset and
  did not recompute real-image statistics. This pilot is an execution check,
  not a replacement for the 50,000-image paper evaluation.

Dataset publication status and the pinned download revision are recorded in
[the dataset release manifest](../releases/dataset.json).

A startup check exercises asset loading, distributed forward/backward passes,
optimizer updates, EMA, and checkpoint writing. It does not establish a full
160-epoch or 25,000-update reproduction, final metrics, or recovery fidelity.
The production launcher's longer probe additionally checks optimizer/EMA replay
and resume behavior.

## Bounded GPU checks

Use a new output directory for every check. Keep environments, assets, bundles,
and output checkpoints on external storage, with compiler caches on node-local
`/tmp` (or `PRIMING_LOCAL_TMP`); see
[the scratch setup example](setup.md#keep-large-files-outside-home).
Prepare the dataset and encoders in [assets.md](assets.md) before running.
Eight ranks and the production microbatch/accumulation are preserved.

Reconstruction pretraining starts from random initialization. Its released
Hugging Face checkpoints are outputs of this stage and inputs to post-training;
they are not prerequisites for training the reconstruction model from scratch.

```bash
python pretraining/scripts/prepare.py --recipe b160_perceptual \
  --bundle "$PRIMING_STORAGE/checks/b160_bundle" \
  --output "$PRIMING_STORAGE/checks/b160_run"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  pretraining/scripts/train.py --bundle "$PRIMING_STORAGE/checks/b160_bundle" \
  --smoke-steps 2
```

Repeat with `b160_no_perceptual` and `l160_perceptual`, using separate bundle
and output directories. For multiple nodes, use the Slurm pretraining launcher
with `SMOKE_STEPS=2`, the desired node/GPU layout, and your account/partition.

Post-training checks initialize online weights and EMA from the released
160-epoch checkpoint, verify its SHA256, and use a fresh optimizer:

```bash
python training/scripts/prepare.py --arm b160 \
  --job "$PRIMING_STORAGE/checks/post_b160" \
  --output-dir "$PRIMING_STORAGE/checks/post_outputs" --smoke-steps 2
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  training/scripts/train.py --job "$PRIMING_STORAGE/checks/post_b160"
```

Repeat with `l160`. The bounded post-training config retains the 125-update
warmup and full OT populations; it disables scheduled/final metric evaluation,
sets the run horizon to two updates, and saves the terminal checkpoint. Smoke
outputs are for validation and should not be used for production continuation.
A prepared job records `mode: smoke` in its config manifest.

## CPU and source checks

```bash
python scripts/check_environment.py
python scripts/check_portability.py
python scripts/check_public_tree.py
python -m pytest -q tests eval/tests
```

The portability check runs isolated imports, packaging and numerical CPU
regressions, configuration/source verification, Slurm command previews, and
arithmetic checks of the included final evaluation results.

## Measured startup validation (2026-10-03)

The [validation record](../environment/pipeline_validation.json) covers eight
H200 GPUs on two nodes, with production global batch 1,024 and compiled BF16
models. Every listed recipe completed two optimizer updates and saved its
checkpoint using real ImageNet training data.

| Stage | Recipes checked | Initialization |
|---|---|---|
| Reconstruction | B perceptual, B no-perceptual, L perceptual | Random |
| Reference OT post-training | B-160, L-160 | Released pretrained online + EMA weights |
| Post-training ablations | B-160 SIM, Laplacian MMD, energy | Released pretrained online + EMA weights |

Saved-state audits checked finite weights/optimizer/EMA, counters, and the
configured losses. Post-training additionally verified all eight ranks'
initialization and main/OT RNG states. B pretraining saved complete all-rank
optimizer state; L retains its documented rank-zero optimizer format.
L post-training peaked at **85.01 GiB allocated / 87.58 GiB reserved**, above
80 GB device capacity at the tested batch size. Other hardware was not tested.

[Sampling validation](../environment/sampling_validation.json) covers all eight
B/L × 80/160 × pretrained/post-trained release checkpoints: two finite
256×256 RGB PNGs each, using EMA and BF16 on H200. Local checkpoint SHA256s
match the Hub's published LFS identities; existing checkpoint bytes were reused.
[Environment validation](../environment/validation.json) and
[portability validation](../environment/portability_validation.json) record the
package/import and source-only checks.

These checks did not rerun full training, the 130-update recovery probe, or
50,000-image metric evaluation. The included final paper/ablation metric values
were preserved and checked arithmetically. Raw job logs and temporary training
artifacts are not distributed.
