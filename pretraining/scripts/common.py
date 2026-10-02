"""Paths and file identities shared by pretraining packaging tools."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from priming_paths import resolve
RECIPES = ('b160_perceptual', 'b160_no_perceptual', 'l160_perceptual')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify_runtime(family):
    manifest = json.loads((ROOT / 'provenance/runtime_manifest.json').read_text())
    prefix = f'runtime/{family}/'
    files = {name[len(prefix):]: value for name, value in manifest.items() if name.startswith(prefix)}
    if not files:
        raise ValueError(f'Missing runtime manifest for {family}')
    for name, value in files.items():
        if sha(ROOT / prefix / name) != value:
            raise ValueError(f'Migrated runtime changed: {prefix}{name}')
    return files
