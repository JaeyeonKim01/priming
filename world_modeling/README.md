# World modeling

RT-1 and Bridge-V2 pretraining, post-training, and evaluation, contained entirely
under `world_modeling/`. Run commands from the Priming repository root.

Both tasks have completed fresh 64-epoch pretraining, 4,000-update post-training,
and full validation evaluation. Results are close to the historical checkpoints
but do not exactly match their metrics. See [validation results](docs/validation.md).

## Setup

Use Python 3.11 with Priming's PyTorch 2.6 / CUDA 12.4 environment and install
`world_modeling/requirements.txt`. Keep datasets and pretrained assets external:

```bash
export WORLD_MODEL_STORAGE=/path/to/world-model-assets
export DISTMATCH_SOURCE=/path/to/distmatch/src
```

The asset root contains `data/`, `models/`, `libraries/`, and `cache/`.
DistMatch is loaded from the explicit source path. These runs use version 0.6.0;
historical checkpoints record 0.4.0. See the [recipe and asset details](docs/migration.md)
and [source provenance](docs/provenance.md).

## Training and evaluation

Full training uses eight GPUs. Post-training initializes from the new pretraining
EMA with a fresh optimizer and EMA. Replace `rt1.yaml` with `bridge.yaml` for Bridge.
For multiple nodes, supply the corresponding torchrun rendezvous arguments.

```bash
# Pretraining: 64 epochs.
torchrun --standalone --nproc_per_node=8 -m world_modeling.pretrain \
  --config world_modeling/configs/pretrain/rt1.yaml --output /path/to/run/pretrain

# Post-training: 4,000 updates from the final pretraining EMA.
torchrun --standalone --nproc_per_node=8 -m world_modeling.posttrain \
  --config world_modeling/configs/posttrain/rt1.yaml \
  --init-checkpoint /path/to/run/pretrain/ckpt_save/ckpt-step84608.pth \
  --output /path/to/run/posttrain

# Full validation evaluation.
torchrun --standalone --nproc_per_node=8 -m world_modeling.eval \
  --config world_modeling/configs/eval/rt1.yaml \
  --checkpoint /path/to/run/posttrain/ckpt_save/ckpt-step4000.pth \
  --output /path/to/run/eval --set evaluation.save_videos=0
```

Bridge pretraining ends at update 53,184; RT-1 ends at 84,608. Use `--resume`
to recover a stage, `--max-steps` for bounded checks, and repeat `--set KEY=VALUE`
for overrides. Resume restores optimizer, EMA and rank RNG states, but not loader
worker/prefetch state, so continuation is not bit-exact.

## Slurm launchers

The production launcher runs pretraining → post-training → full evaluation for
one task on two nodes with four H100s each. It freezes code/configs, records input
provenance, validates stage handoffs, and resumes existing checkpoints.

```bash
export WORLD_MODEL_PYTHON=/path/to/environment/bin/python
export WORLD_MODEL_RUN=/path/to/new/production
"$WORLD_MODEL_PYTHON" -m world_modeling.scripts.production prepare \
  --run-root "$WORLD_MODEL_RUN" --storage "$WORLD_MODEL_STORAGE" \
  --distmatch "$DISTMATCH_SOURCE"
for WORLD_MODEL_DATASET in rt1 bridge; do
  export WORLD_MODEL_DATASET
  sbatch --job-name="world-prod-$WORLD_MODEL_DATASET" \
    --output="$WORLD_MODEL_RUN/$WORLD_MODEL_DATASET-%j.log" \
    "$WORLD_MODEL_RUN/code/world_modeling/scripts/production.slurm"
done
```

Adapt the account, partition and notification settings to your cluster.
`scripts/training_pipeline.slurm` provides bounded training/resume checks;
`scripts/reproduce_eval.slurm` compares historical-checkpoint evaluation against
the original implementation. Each launcher documents its required environment
variables. Checkpoints, datasets and generated artifacts stay outside the repo.

## Tests

```bash
python -m unittest discover -s world_modeling/tests -v
```

The tests cover reconstruction, sampling, replay gradients, EMA initialization,
checkpoint recovery, metric conventions and production checkpoint validation.
