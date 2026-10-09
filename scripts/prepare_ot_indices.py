"""Rebuild or verify class indices against the packed release dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--calibration', type=Path, help='Also verify the frozen full-MAE OT calibration')
    a = p.parse_args()
    # Read only labels; do not decode or copy image columns.
    data = load_from_disk(str(a.dataset_dir))
    labels = np.asarray(data.select_columns(['label'])['label'], dtype=np.int64)
    if labels.shape != (1281167,) or labels.min() != 0 or labels.max() != 999:
        raise ValueError('Expected the ImageNet training split with 1000 classes')
    order = np.argsort(labels, kind='stable')
    counts = np.bincount(labels, minlength=1000)
    if counts.min() < 128:
        raise ValueError('Insufficient examples for the released OT populations')
    for name, array in [('dataset_labels.npy', labels), ('class_order.npy', order), ('class_counts.npy', counts)]:
        path = a.output_dir / name
        if a.verify_only:
            if not np.array_equal(np.load(path, allow_pickle=False), array):
                raise ValueError(f'Index does not match dataset: {name}')
        else:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array, allow_pickle=False)
    if a.calibration:
        c = json.loads(a.calibration.read_text())
        selected = np.asarray(c['sorted_row_indices'], dtype=np.int64)
        if len(selected) != c['samples'] or len(np.unique(selected)) != len(selected):
            raise ValueError('Invalid calibration sample indices')
        if selected.min() < 0 or selected.max() >= len(order):
            raise ValueError('Calibration index outside dataset')
        if not np.array_equal(order[selected], c['dataset_indices']):
            raise ValueError('Calibration and packed dataset ordering differ')
        names = set(c['bandwidths'])
        if len(names) != 97 or names != set(c['weights']) or names != set(c['dimensions']):
            raise ValueError('Full-MAE/ConvNeXt feature identities differ')
        if not all(np.isfinite(x) and x > 0 for x in c['bandwidths'].values()):
            raise ValueError('Invalid bandwidths')
        if not all(np.isfinite(x) and x > 0 for x in c['weights'].values()) or not np.isclose(sum(c['weights'].values()), 1):
            raise ValueError('Invalid feature weights')
        source = ROOT / 'models/encoders/full'
        files = [source / 'encoders.py', *sorted((source / 'vendor').glob('*.py'))]
        identity = hashlib.sha256(b''.join(f.read_bytes() for f in files)).hexdigest()
        if identity != c['encoder_source_sha256']:
            raise ValueError('Calibration belongs to a different encoder implementation')
    print('PASS: packed dataset class indices' + (' and frozen calibration' if a.calibration else ''))


if __name__ == '__main__':
    main()
