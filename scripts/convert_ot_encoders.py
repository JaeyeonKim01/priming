"""Convert pinned upstream MAE/ConvNeXt weights to the released encoder format."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/runtime'))
from download_assets import matches


def read_flax_tree(path):
    """Read the numeric Flax MessagePack format without installing JAX.

    Format specification: google/flax v0.10.4, flax/serialization.py.
    Only numeric ndarrays/scalars and chunked numeric arrays are accepted.
    """
    import msgpack
    import numpy as np

    def extension(code, payload):
        if code not in (1, 3):
            raise ValueError(f'Unsupported MessagePack extension {code}')
        shape, name, data = msgpack.unpackb(payload, raw=True)
        dtype = np.dtype(name.decode())
        if dtype.kind not in 'biufc' or dtype.hasobject:
            raise ValueError('Only numeric tensor data are supported')
        array = np.frombuffer(data, dtype=dtype).reshape(shape)
        return array if code == 1 else array[()]

    def restore(value):
        if not isinstance(value, dict):
            return value
        if value.get('__msgpack_chunked_array__') is True:
            chunks = value['chunks']
            shape = value['shape']
            return np.concatenate([chunks[str(i)] for i in range(len(chunks))]).reshape(
                tuple(shape[str(i)] for i in range(len(shape))))
        return {k: restore(v) for k, v in value.items()}

    with path.open('rb') as f:
        return restore(msgpack.unpack(f, raw=False, ext_hook=extension, strict_map_key=False))


def convert(root, family, output):
    import torch
    from src.ot_representations_full.encoders import make_mae, make_convnext
    from src.ot_representations_full.vendor.param_convert import convert_mae_tree_to_state_dict
    from src.ot_representations_full.vendor.convnext import _map_hf_key_to_local
    rows = [r for r in json.loads((ROOT / 'releases/encoders.json').read_text())['files']
            if 'conversion' in r['groups'] and f'/encoder_sources/{family}/' in r['path']]
    for row in rows:
        if not matches(root / row['path'], row):
            raise ValueError(f"Missing or changed source: {row['path']}")
    source = root / 'assets/encoder_sources' / family
    if family == 'mae':
        source = source / 'models/mae/jax/mae_pixel_640'
        config = json.loads((source / 'metadata.json').read_text())['model_config']
        model = make_mae(config)
        tree = read_flax_tree(source / 'ema_params.msgpack')
        state = convert_mae_tree_to_state_dict(tree, model.state_dict())
        del tree
    else:
        model = make_convnext()
        upstream = torch.load(source / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
        mapped = {_map_hf_key_to_local(k): v for k, v in upstream.items()}
        state = {}
        target_state = model.state_dict()
        for key, value in mapped.items():
            if key not in target_state:
                continue  # The upstream classifier is unused by the encoder.
            target = target_state[key]
            if value.shape != target.shape:
                raise ValueError(f'ConvNeXt shape mismatch: {key}')
            state[key] = value.to(dtype=target.dtype).contiguous()
    model.load_state_dict(state, strict=True, assign=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(state, output)
    expected = next(row for row in json.loads((ROOT / 'training/assets.json').read_text())['files']
                    if row['path'] == f'assets/ot/representations/{family}_encoder.pt')
    if not matches(output, expected):
        raise ValueError(f'Converted file differs from the pinned release: {output}. '
                         'Use the pinned environment; do not bypass asset verification.')
    print(f'{family}: strict loading and release SHA256 passed', flush=True)
    del model, state
    gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--family', choices=['mae', 'convnext', 'both'], default='both')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    destination = a.output_dir or a.root / 'assets/ot/representations'
    for family in (['mae', 'convnext'] if a.family == 'both' else [a.family]):
        convert(a.root, family, destination / f'{family}_encoder.pt')


if __name__ == '__main__':
    main()
