"""Repository-relative operational paths; historical provenance is never consulted."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve(value, root=ROOT):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else Path(root) / path


def asset(name, default):
    return resolve(os.environ.get(name, default))


def configure_caches():
    defaults = {'HF_HOME': 'assets/huggingface', 'TORCH_HOME': 'assets/torch'}
    for name, value in defaults.items():
        os.environ[name] = str(resolve(os.environ.get(name, value)))
    os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
