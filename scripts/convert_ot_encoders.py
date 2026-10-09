"""Convert pinned upstream MAE/ConvNeXt weights to the released encoder format."""
import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_assets import matches


def normalize_serialization_id(path, identity):
    """Pin PyTorch's non-tensor ZIP metadata before checking the full file hash.

    The generated ID can differ across environments even with byte-identical
    pickle/tensor records. Change only that stored record and its ZIP CRCs;
    preserve all other bytes, including PyTorch's alignment and ZIP64 headers.
    """
    if len(identity) != 40 or not identity.isascii() or not identity.isdigit():
        raise ValueError('Expected the released 40-digit serialization ID')
    replacement = identity.encode('ascii')
    with zipfile.ZipFile(path) as archive:
        records = archive.infolist()
        candidates = [r for r in records if r.filename.endswith('/.data/serialization_id')]
        if len(candidates) != 1:
            raise ValueError('Expected one PyTorch serialization ID record')
        record = candidates[0]
        if record.compress_type != zipfile.ZIP_STORED or record.file_size != 40:
            raise ValueError('Unexpected serialization ID encoding')
        original = archive.read(record)  # Also validates the original CRC.
        central_offset = archive.start_dir
    if original == replacement:
        return
    crc = zlib.crc32(replacement)
    with Path(path).open('r+b') as stream:
        stream.seek(record.header_offset)
        header = stream.read(30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP local header')
        name_size, extra_size = struct.unpack_from('<HH', header, 26)
        data_offset = record.header_offset + 30 + name_size + extra_size
        # Find the central-directory CRC using the standard ZIP record lengths.
        central_crc = None
        stream.seek(central_offset)
        for _ in records:
            offset = stream.tell()
            header = stream.read(46)
            if header[:4] != b'PK\x01\x02':
                raise ValueError('Invalid ZIP central directory')
            name_size, extra_size, comment_size = struct.unpack_from('<HHH', header, 28)
            name = stream.read(name_size)
            if name == record.filename.encode('utf-8'):
                central_crc = offset + 16
                break
            stream.seek(extra_size + comment_size, os.SEEK_CUR)
        if central_crc is None:
            raise ValueError('Missing serialization ID directory entry')
        if record.flag_bits & 8:
            descriptor = data_offset + record.compress_size
            stream.seek(descriptor)
            if stream.read(4) != b'PK\x07\x08':
                raise ValueError('Expected PyTorch ZIP data descriptor')
            local_crc = descriptor + 4  # Same position for ZIP32 and ZIP64.
        else:
            local_crc = record.header_offset + 14
        stream.seek(data_offset)
        stream.write(replacement)
        for offset in (local_crc, central_crc):
            stream.seek(offset)
            stream.write(struct.pack('<I', crc))


def save_verified_encoder(state, output, expected):
    """Publish only a complete encoder matching the original release SHA256."""
    import torch
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.convert-', dir=output.parent) as folder:
        temporary = Path(folder) / output.name  # torch.save embeds the basename.
        torch.save(state, temporary)
        normalize_serialization_id(temporary, expected['serialization_id'])
        if not matches(temporary, expected):
            raise ValueError(f'Converted file differs from the pinned release: {output}. '
                             'Use the pinned environment; do not bypass asset verification.')
        os.link(temporary, output)  # Atomic, and refuses a concurrently created output.


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
    from models.encoders.full.encoders import make_mae, make_convnext
    from models.encoders.full.vendor.param_convert import convert_mae_tree_to_state_dict
    from models.encoders.full.vendor.convnext import _map_hf_key_to_local
    rows = [r for r in json.loads((ROOT / 'old/releases/encoders.json').read_text())['files']
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
    expected = next(row for row in json.loads((ROOT / 'old/releases/posttrain_encoders.json').read_text())['files']
                    if row['path'] == f'{family}_encoder.pt')
    save_verified_encoder(state, output, expected)
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
