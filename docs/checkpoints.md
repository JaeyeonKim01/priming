# Hugging Face checkpoint upload and download

The release contains eight PyTorch `.pth` files: B/L × 80/160 pretraining epochs
× before/after post-training. The two checked-in manifests pin filenames,
sizes, and SHA256. Pretraining files contain online and EMA state; post-training
initializes from online weights. Release evaluation uses post-training EMA .999.
These are full training checkpoints, not Diffusers or safetensors models.

## Download the hosted checkpoints

The eight files are hosted at [jaykim1102/priming](https://huggingface.co/jaykim1102/priming).
The repository was private when checked; log in with an account that has read
access. From the code repository root with the `priming` environment active:

```bash
hf auth login
hf download jaykim1102/priming --include 'checkpoints/*' --local-dir .
python verify_checkpoints.py
```

For the exact checked snapshot, add
`--revision 9dc5fe94583b4479edb954327110850dc748ce44` to the download command.
The [verification report](../environment/huggingface_validation.json) records
remote identities, downloaded checksums, and online/EMA state-loading results.
This checks checkpoint loading; it is not a new GPU sampling or metric run.

`--metadata-only` on `verify_checkpoints.py` checks sizes and inference configs
but skips checkpoint hashes. The default performs full verification. Ordinary
files downloaded from the Hub are accepted; original-workspace symlinks are
not required. Evaluation also needs the [encoder/reference assets](assets.md).

## Upload or update your release

Activate `priming` and run from the repository root. Put the eight weights at
the paths listed by this preview (it makes no network changes):

```bash
python scripts/upload_checkpoints.py --repo-id YOUR_ACCOUNT/priming
```

If another directory already contains the same `checkpoints/` layout, link it
with `python link_checkpoints.py --source-dir /path/to/export`. Otherwise copy
the files into `checkpoints/` and `checkpoints/pretraining/` yourself.

```bash
python verify_checkpoints.py
hf auth login
python scripts/upload_checkpoints.py --repo-id YOUR_ACCOUNT/priming --upload
```

Use a Hugging Face account/token with write access to that namespace. The upload
command verifies every checkpoint, creates a **private** model repo by default,
and uploads weights, both manifests, inference configs, and the model card.
Add `--public` to create a public repo. This flag does not change the visibility
of an existing repository; manage that in the Hub settings. Review the model
card and decide the release license before publishing it publicly.

The API uploads files individually; rerunning the command can finish an
interrupted upload. Keep the same manifests and verify the download afterward.

Official references: [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and [upload API](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
