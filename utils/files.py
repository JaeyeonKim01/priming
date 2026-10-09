"""Streaming file checksums and atomic records; no model dependencies."""
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def prepare_output_directory(path, rank, distributed=None):
    """Broadcast the overwrite decision before any rank writes artifacts."""
    path = Path(path)
    error = [None]
    if rank == 0:
        try:
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f'Refusing to overwrite {path}')
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError) as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if distributed is not None:
        distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    return path
