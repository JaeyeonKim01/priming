"""Reject reuse of evaluation outputs from a different checkpoint or recipe."""
import json
from pathlib import Path
from omegaconf import OmegaConf


def evaluation_identity(cfg, world_size, weights):
    checkpoint = Path(cfg.checkpoint).resolve()
    stat = checkpoint.stat()
    resolved = OmegaConf.to_container(cfg, resolve=True)
    fields = ('model', 'flow', 'rt1', 'bridge', 'assets', 'eval', 'evaluation')
    return dict(format_version=1, checkpoint=str(checkpoint), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, world_size=world_size, weights=weights,
                seed=int(cfg.train.seed), recipe={k: resolved[k] for k in fields if k in resolved})


def validate_output_identity(output, identity, *, aggregate_only=False):
    output = Path(output)
    path = output / 'evaluation_identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Evaluation output belongs to a different checkpoint or recipe; use a new output directory')
    elif aggregate_only or (output / 'eval_on_validation_multigpu').exists():
        raise ValueError('Existing evaluation has no verified identity; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, indent=2)+'\n')
