"""Complete scheduled draws atomically and restore all training RNG streams."""
import gc
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from metrics.specs import BASELINES
from .files import sha256, write_json
from .posttrain_checkpoint import snapshot_rng, restore_rng


def validate_rows(rows, config):
    if not isinstance(rows, list) or len(rows) != 2 or [row.get('ema_label') for row in rows] != ['online', '0.999']:
        raise ValueError('Evaluation needs online and EMA 0.999 results')
    for row in rows:
        if row.get('num_samples') != config['evaluation']['num_images'] or row.get('cfg') != config['sampling']['cfg']:
            raise ValueError('Evaluation sample count or CFG differs')
        if set(row.get('fd', {})) != set(BASELINES) or set(row.get('fdr', {})) != set(BASELINES):
            raise ValueError('Evaluation needs all six feature metrics')
        values = [*row['fd'].values(), *row['fdr'].values(), row['fdr6'], row['is'], row['precision'], row['recall']]
        if any(not math.isfinite(value) for value in values):
            raise FloatingPointError('Nonfinite evaluation metric')
        if any(row['fdr'][name] != row['fd'][name]/baseline for name, baseline in BASELINES.items()):
            raise ValueError('Evaluation normalization differs from its six baselines')
        if row['fdr6'] != sum(row['fdr'][name] for name in BASELINES)/6 or not (0 <= row['precision'] <= 1 and 0 <= row['recall'] <= 1):
            raise ValueError('Invalid FDr-6 or precision/recall result')


def scheduled_evaluation(output, checkpoint, completed, config, evaluate, *, rank=0, world=1):
    """An interrupted draw is retried from its saved checkpoint before training resumes."""
    path = Path(output)/'evaluation'/f'step_{completed:06d}.json'
    message = [None]
    if rank == 0:
        try:
            identity = dict(version=1, step=completed, checkpoint_sha256=sha256(checkpoint),
                            evaluation=config['evaluation'], sampling=config['sampling'])
            record = json.loads(path.read_text()) if path.is_file() else None
            if record is not None and record.get('identity') == identity:
                validate_rows(record['rows'], config)
                message[0] = dict(identity=identity, cached=record['rows'])
            else:
                message[0] = dict(identity=identity)
        except Exception as exc:
            message[0] = dict(error=repr(exc))
    if world > 1:
        dist.broadcast_object_list(message, src=0)
    if 'error' in message[0]:
        raise RuntimeError(message[0]['error'])
    if 'cached' in message[0]:
        return message[0]['cached']
    state = snapshot_rng()
    try:
        rows = evaluate()
        validate_rows(rows, config)
    finally:
        restore_rng(state)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    message = [None]
    if rank == 0:
        try:
            write_json(path, dict(identity=identity, rows=rows))
            message[0] = dict(ok=True)
        except Exception as exc:
            message[0] = dict(error=repr(exc))
    if world > 1:
        dist.broadcast_object_list(message, src=0)
    if 'error' in message[0]:
        raise RuntimeError(message[0]['error'])
    return rows
