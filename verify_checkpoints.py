"""Verify portable checkpoint identities; accepts ordinary files or local symlinks."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            result.update(block)
    return result.hexdigest()


def records(stage='all'):
    for name in ('manifest.json', 'pretraining_manifest.json'):
        rows = json.loads((HERE / 'checkpoints' / name).read_text())
        for key, item in sorted(rows.items()):
            if stage == 'all' or stage == item['stage']:
                yield key, item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'pretraining', 'posttraining'], default='all')
    parser.add_argument('--metadata-only', action='store_true', help='Check sizes/configs; skip large weight hashes')
    args = parser.parse_args()
    count = 0
    for key, item in records(args.stage):
        path = HERE / item['checkpoint']
        assert path.is_file() and path.stat().st_size == item['checkpoint_bytes'], str(path)
        if not args.metadata_only:
            assert digest(path) == item['checkpoint_sha256'], str(path)
        if 'config' in item:
            assert digest(HERE / item['config']) == item['config_sha256'], item['config']
        print(f'{item["stage"]}/{key}: verified', flush=True)
        count += 1
    print(f'{count} checkpoints checked; SHA256={not args.metadata_only}')


if __name__ == '__main__':
    main()
