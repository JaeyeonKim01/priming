"""Validate every CC12M tensor chunk and read samples before training."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks', required=True)
    args = parser.parse_args()
    root = Path(args.chunks)
    marker = json.loads((root / '_cc12m_wds_to_chunks_complete.json').read_text())
    total_expected = int(marker['total_samples'])
    chunk_size = int(marker['chunk_size'])
    expected_count = math.ceil(total_expected / chunk_size)
    files = sorted(root.glob('chunk_*.pt'))
    if total_expected <= 0 or int(marker['total_chunks']) != expected_count:
        raise ValueError('invalid sample/chunk counts in completion marker')
    if len(files) != expected_count or [p.name for p in files] != [f'chunk_{i:06d}.pt' for i in range(expected_count)]:
        raise ValueError('missing, extra, or non-contiguous tensor chunks')
    if list(root.glob('*.tmp')):
        raise ValueError('temporary files remain; conversion has not completed cleanly')
    total = 0
    size = 0
    empty = 0
    sampled = 0
    for i, path in enumerate(files):
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        n = min(chunk_size, total_expected - total)
        image_size = int(marker['image_size'])
        prompt_length = int(marker['prompt_length'])
        for key, shape, dtype in [
            ('pixel_values', (n, 3, image_size, image_size), torch.uint8),
            ('input_ids', (n, prompt_length), torch.int64),
            ('attention_mask', (n, prompt_length), torch.int64),
        ]:
            value = payload[key]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f'{path.name}: invalid {key} shape or dtype')
        captions = payload['caption']
        if len(captions) != n or not all(isinstance(c, str) for c in captions):
            raise ValueError(f'{path.name}: invalid captions')
        ids, mask = payload['input_ids'], payload['attention_mask']
        if ids.min().item() < 0 or ids.max().item() >= 32128:
            raise ValueError(f'{path.name}: token ID out of FLAN-T5 vocabulary range')
        if mask.min().item() < 0 or mask.max().item() > 1:
            raise ValueError(f'{path.name}: invalid attention mask')
        for index in sorted({0, n // 2, n - 1}):
            payload['pixel_values'][index].sum().item()
            sampled += 1
        empty += sum(not c.strip() for c in captions)
        total += n
        size += path.stat().st_size
        del payload, ids, mask, value
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'verified {i + 1}/{len(files)} chunks: {total}/{total_expected} samples', flush=True)
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'total_samples': total,
              'total_chunks': len(files), 'total_bytes': size, 'empty_captions': empty,
              'sampled_images_read': sampled, 'all_token_arrays_read': True,
              'full_pixel_checksum_scan': False}
    out = root / '_cc12m_chunks_verified.json'
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out)
    print('CC12M_CHUNKS_VERIFIED ' + json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
