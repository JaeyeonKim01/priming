"""Download a selected release asset group into an explicit storage root."""
import argparse
import sys
import json
import os
from pathlib import Path, PurePosixPath
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.files import sha256


def matches(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256']


def safe_path(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError(f'Invalid manifest path: {name}')
    return root.joinpath(*p.parts)


def download(root, row, verify_only=False):
    target = safe_path(root, row['path'])
    if not matches(target, row):
        if target.exists():
            raise ValueError(f'Existing file differs from release: {target}')
        if verify_only:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if row['kind'] == 'url':
            tmp = target.with_name(target.name + '.partial')
            with urllib.request.urlopen(row['url'], timeout=120) as response, tmp.open('wb') as out:
                while block := response.read(8 << 20):
                    out.write(block)
            if not matches(tmp, row):
                raise ValueError(f'Download checksum differs: {target.name}')
            tmp.replace(target)
        else:
            from huggingface_hub import hf_hub_download
            if not row.get('revision'):
                raise ValueError('This asset release has not been published/pinned yet')
            kwargs = dict(repo_id=row['repo_id'], filename=row['filename'], revision=row['revision'])
            if row['kind'] == 'hub_cache':
                kwargs['cache_dir'] = root / 'assets/huggingface/hub'
            else:
                suffix = PurePosixPath(row['filename']).parts
                parts = PurePosixPath(row['path']).parts
                if parts[-len(suffix):] != suffix:
                    raise ValueError('Hub filename is not a suffix of its destination')
                kwargs.update(repo_type=row.get('repo_type', 'model'),
                              local_dir=root.joinpath(*parts[:-len(suffix)]))
            hf_hub_download(**kwargs)
            if not matches(target, row):
                raise ValueError(f'Download checksum differs: {target.name}')
    if row['kind'] == 'hub_cache':
        # timm/Transformers resolve the model name through refs/main offline.
        ref = root / 'assets/huggingface/hub' / ('models--' + row['repo_id'].replace('/', '--')) / 'refs/main'
        if ref.exists() and ref.read_text().strip() != row['revision']:
            raise ValueError(f'Use a fresh cache: {ref} selects a different revision')
        if verify_only:
            if not ref.is_file():
                raise FileNotFoundError(ref)
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(row['revision'])
    print(f"Verified {row['path']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=['dataset', 'pretraining', 'posttraining', 'evaluation', 'conversion'], required=True)
    p.add_argument('--root', type=Path, default=ROOT, help='Destination containing data/ and assets/')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    root = a.root.expanduser().resolve()
    # Authentication is read from the existing login; all large caches live here.
    os.environ.setdefault('HF_XET_CACHE', str(root / 'assets/.xet-cache'))
    if a.group == 'dataset':
        release = json.loads((ROOT / 'old/releases/dataset.json').read_text())
        rows = [dict(row, kind='hub_file', repo_id=release['repo_id'], repo_type='dataset',
                     revision=release['revision'], filename=row['path']) for row in release['files']]
    else:
        rows = [r for r in json.loads((ROOT / 'old/releases/encoders.json').read_text())['files'] if a.group in r['groups']]
    print(f"{len(rows)} files; {sum(r['bytes'] for r in rows) / 1e9:.3f} GB", flush=True)
    for row in rows:
        download(root, row, a.verify_only)


if __name__ == '__main__':
    main()
