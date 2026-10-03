"""Coordinate filesystem setup before distributed evaluation writes artifacts."""
from pathlib import Path


def prepare_output_directory(path, rank, distributed):
    path = Path(path)
    error = [None]
    if rank == 0:
        try:
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f'Refusing to overwrite {path}')
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError) as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    # Every rank sees the same decision before writing. Other ranks must not
    # inspect directory contents after a faster rank has started producing files.
    distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    return path
