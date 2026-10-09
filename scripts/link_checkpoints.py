"""Link checkpoint files from an explicitly supplied source tree into this checkout."""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True,
                        help='Directory containing the checkpoints/ layout from the Hugging Face release')
    args = parser.parse_args()
    source_root = args.source_dir.expanduser().resolve()
    for filename in ('manifest.json', 'pretraining_manifest.json'):
        for key, item in json.loads((HERE / 'old/checkpoints' / filename).read_text()).items():
            target = source_root / item['checkpoint']
            link = HERE / item['checkpoint']
            if not target.is_file() or target.stat().st_size != item['checkpoint_bytes']:
                raise ValueError(f'Missing or wrong-size source: {target}')
            if link.exists() or link.is_symlink():
                if link.resolve() != target.resolve():
                    raise FileExistsError(f'Refusing to replace {link}')
            else:
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(target, link.parent))
            print(f'{item["stage"]}/{key}: {link}')


if __name__ == '__main__':
    main()
