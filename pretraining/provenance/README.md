# Pretraining source and reproducibility

The release includes the B and L runtime snapshots, full scientific recipes,
and their SHA256 manifests. Source records use portable paths and neutral
logging names. Private launch scripts, accounts, emails, and old logs are not
part of the release.

`copied_files.json` and `runtime_manifest.json` identify the distributed bytes.
`b_production_manifest.json` and `l_snapshot_manifest.json` retain the original
runtime hashes. `path_sanitization.json` records the original and public hashes
for files whose private defaults were replaced; verification checks both.
These edits do not change the numerical training functions.

## Historical training details

B uses math SDPA, deterministic cuDNN/algorithms, native CPU bicubic crop
backward, stateless flip/noise/crop seeds, and complete all-rank Muon/RNG
recovery. The released B-80 and B-160 weights share the perceptual 160-epoch
recipe. The no-perceptual recipe is a separate ablation.

L uses the retained source snapshot with the confirmed 160-epoch recipe,
ordinary RNG/dataloader streams, nondeterministic backend settings, and CUDA
crop backward. The original run resumed at epoch 110. Its checkpoint format
stores rank-zero optimizer state, so restarting L is not claimed to reproduce
an uninterrupted all-rank trajectory. A complete immutable snapshot of the
original L allocation was not retained; exact original code and SDPA kernel
identity are not asserted.

Perceptual thresholds remain B 0.8 and L 1.0; both include all examples at
fixed t=0.8. Model factories, objectives, and optimizers remain separate.
See [checkpoint identities](../../checkpoints/pretraining_manifest.json),
[recipes](../configs), and [startup validation](../../docs/validation.md).
