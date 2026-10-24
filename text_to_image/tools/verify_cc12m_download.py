"""Check every expected img2dataset shard before tensor conversion."""
from __future__ import annotations

import argparse
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--wds', required=True)
    parser.add_argument('--shard-size', type=int, default=10_000)
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError('shard-size must be positive')
    metadata = pq.ParquetFile(args.parquet)
    if not {'url', 'caption_llava'} <= set(metadata.schema.names):
        raise ValueError('metadata must contain url and caption_llava')
    rows = metadata.metadata.num_rows
    if rows == 0:
        raise ValueError('metadata is empty')
    root = Path(args.wds)
    count = math.ceil(rows / args.shard_size)
    expected = [f'{i:05d}.tar' for i in range(count)]
    if sorted(p.name for p in root.glob('*.tar')) != expected:
        raise ValueError(f'expected {count} contiguous tar shards under {root}')
    totals = {key: 0 for key in ['count', 'successes', 'failed_to_download', 'failed_to_resize']}
    for i in range(count):
        tar = root / f'{i:05d}.tar'
        table = root / f'{i:05d}.parquet'
        stats = json.loads((root / f'{i:05d}_stats.json').read_text())
        expected_rows = min(args.shard_size, rows - i * args.shard_size)
        if pq.ParquetFile(table).metadata.num_rows != expected_rows:
            raise ValueError(f'{table}: wrong number of rows')
        if not tarfile.is_tarfile(tar):
            raise ValueError(f'{tar}: unreadable tar header')
        values = {key: int(stats[key]) for key in totals}
        if values['count'] != expected_rows:
            raise ValueError(f'{tar}: stats count differs from metadata')
        if values['successes'] + values['failed_to_download'] + values['failed_to_resize'] != expected_rows:
            raise ValueError(f'{tar}: stats outcomes do not sum to count')
        for key, value in values.items():
            totals[key] += value
    if totals['successes'] <= 0:
        raise ValueError('no successfully downloaded images')
    report = {'verified_at': datetime.now(timezone.utc).isoformat(),
              'metadata_rows': rows, 'total_shards': count, 'shard_size': args.shard_size,
              'totals': totals, 'tar_bytes': sum(p.stat().st_size for p in root.glob('*.tar'))}
    out = root.parent / '_cc12m_download_verified.json'
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out)
    print('CC12M_DOWNLOAD_VERIFIED ' + json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
