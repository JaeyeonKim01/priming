# Hugging Face checkpoint upload and download

The release contains eight PyTorch `.pth` files: B/L × 80/160 pretraining epochs
× before/after post-training. The two checked-in manifests pin filenames,
sizes, and SHA256. Pretraining files contain online and EMA state; post-training
initializes from online weights. Release evaluation uses post-training EMA .999.
These are full training checkpoints, not Diffusers or safetensors models.

## Upload your release

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
of an existing repository; manage that in the Hub settings. No upload has been
performed as part of this code migration. Review the model card and decide the
release license before publishing it publicly.

The API uploads files individually; rerunning the command can finish an
interrupted upload. Keep the same manifests and verify the download afterward.

## Download once the release exists

Replace `YOUR_ACCOUNT/priming` with the published model repository ID. From the
code repository root:

```bash
hf download YOUR_ACCOUNT/priming --include 'checkpoints/*' --local-dir .
python verify_checkpoints.py
```

For a private release, log in first with read access. `--metadata-only` checks
sizes and inference configs but skips checkpoint hashes; use the default full
verification before training or publishing. Files copied or downloaded from
the Hub are accepted; the verifier does not require original-workspace symlinks.

Official references: [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
and [upload API](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
