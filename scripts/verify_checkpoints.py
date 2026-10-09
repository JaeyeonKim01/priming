"""Verify portable checkpoint identities; accepts ordinary files or local symlinks."""
import argparse
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from utils.files import sha256


def records(stage='all'):
    for name in ('manifest.json', 'pretraining_manifest.json'):
        rows = json.loads((HERE / 'old/checkpoints' / name).read_text())
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
            assert sha256(path) == item['checkpoint_sha256'], str(path)
        if 'config' in item:
            assert sha256(HERE / 'old' / item['config']) == item['config_sha256'], item['config']
        print(f'{item["stage"]}/{key}: verified', flush=True)
        count += 1
    print(f'{count} checkpoints checked; SHA256={not args.metadata_only}')


if __name__ == '__main__':
    main()
