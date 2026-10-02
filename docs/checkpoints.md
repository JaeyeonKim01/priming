# Checkpoint downloads and sampling

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

## Sampling options

[The sampling script](../scripts/sample.py) runs on one GPU and requires only
the selected generator checkpoint. It uses one step, CFG 1, and EMA weights
by default: .999 for post-training, .9999 for pretraining.

```bash
# Post-trained model; repeat these class IDs in order.
python scripts/sample.py --model priming_b_160 --num-images 16 \
  --class-ids 207 360 --seed 1 --output-dir outputs/samples/b160

# Pretrained model.
hf download jaykim1102/priming checkpoints/pretraining/priming_l_80.pth --local-dir .
python scripts/sample.py --model priming_l_80 --stage pretraining \
  --num-images 8 --output-dir outputs/samples/l80_pretrained
```

Choose `priming_b_80`, `priming_b_160`, `priming_l_80`, or `priming_l_160`.
`--weights online` selects the online generator instead of EMA.
`--checkpoint PATH` accepts another local copy of the selected release file;
its size and SHA256 must match the manifest. `--batch-size` defaults to 8.
Outputs go to a new repository-relative directory; an existing directory is
never overwritten. PNG filenames and `samples.json` record the class labels,
seed, precision, batch size, and checkpoint identity.

CUDA sampling uses BF16. `--device cpu --num-images 1 --batch-size 1` provides
a small FP32 smoke test. Preserve device, batch size and seed when comparing
sample runs. For paper FID/FDr⁵ measurements, use [the evaluation guide](../eval/README.md).

Official reference: [Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
