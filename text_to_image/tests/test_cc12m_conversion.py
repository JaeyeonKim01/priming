import io
import sys
import pytest
import torch
import webdataset as wds
from PIL import Image
from transformers import AutoTokenizer

import tools.cc12m_wds_to_chunks as converter


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained('google/flan-t5-large', local_files_only=True)


def make_shards(root, corrupt=False):
    root.mkdir()
    captions = []
    for shard_index in range(2):
        with wds.TarWriter(str(root / f'{shard_index:05d}.tar')) as writer:
            for index in range(7):
                number = shard_index * 7 + index
                caption = f'Caption {number}: café, 猫.' if number else ''
                captions.append(caption)
                mode = ['RGB', 'L', 'RGBA'][number % 3]
                image = Image.new(mode, (53 + number, 71 - number), 50 + number)
                buf = io.BytesIO()
                fmt = 'JPEG' if mode == 'RGB' else 'PNG'
                image.save(buf, format=fmt)
                writer.write({'__key__': f'{number:09d}', fmt.lower(): buf.getvalue(), 'txt': caption})
                if corrupt and number == 3:
                    writer.write({'__key__': 'bad', 'jpg': b'invalid JPEG', 'txt': 'bad image'})
    return sorted(str(p) for p in root.glob('*.tar')), captions


@pytest.mark.parametrize('workers,async_write', [(0, False), (3, False), (0, True), (3, True)])
def test_pixels_tokens_order_and_corrupt_skip_match_reference(tmp_path, tokenizer, workers, async_write):
    shards, captions = make_shards(tmp_path / 'raw', corrupt=True)
    transform = converter.build_transform(32)
    reference = list(converter.make_webdataset(shards).decode('pil', handler=wds.warn_and_continue).map(
        lambda sample: converter.to_sample(sample, transform), handler=wds.warn_and_continue))
    out = tmp_path / 'out'
    out.mkdir()
    samples = converter.iter_decoded_samples(converter.make_webdataset(shards), transform, workers)
    total, written, _, _ = converter.convert_samples(samples, out, tokenizer, 16, 6,
                                                    async_write=async_write)
    assert total == len(reference) == len(captions) == 14
    assert written == 3
    for i, path in enumerate(sorted(out.glob('chunk_*.pt'))):
        payload = torch.load(path, weights_only=False)
        group = reference[i * 6:(i + 1) * 6]
        expected_captions = captions[i * 6:(i + 1) * 6]
        tokens = tokenizer(expected_captions, max_length=16, padding='max_length',
                           truncation=True, return_tensors='pt')
        assert payload['caption'] == expected_captions
        assert torch.equal(payload['pixel_values'], torch.stack([sample[0] for sample in group]))
        assert torch.equal(payload['input_ids'], tokens.input_ids)
        assert torch.equal(payload['attention_mask'], tokens.attention_mask)


def test_async_write_failure_propagates(tmp_path, tokenizer, monkeypatch):
    def failed_write(*args, **kwargs):
        raise OSError('simulated storage failure')
    monkeypatch.setattr(converter, 'write_chunk', failed_write)
    samples = [(torch.zeros(3, 8, 8, dtype=torch.uint8), 'caption') for _ in range(6)]
    with pytest.raises(OSError, match='simulated storage failure'):
        converter.convert_samples(samples, tmp_path, tokenizer, 16, 3, async_write=True)


@pytest.mark.parametrize('corrupt', [False, True])
def test_parallel_cli_resume_preserves_completed_chunks(tmp_path, tokenizer, monkeypatch, corrupt):
    _, captions = make_shards(tmp_path / 'raw', corrupt=corrupt)
    out = tmp_path / 'out'
    argv = ['cc12m_wds_to_chunks.py', '--wds', str(tmp_path / 'raw'), '--out', str(out),
            '--chunk-size', '6', '--image-size', '32', '--prompt-length', '16',
            '--decode-workers', '3', '--async-write', '--resume']
    monkeypatch.setattr(sys, 'argv', argv)
    converter.main()
    expected = [torch.load(p, weights_only=False) for p in sorted(out.glob('chunk_*.pt'))]
    first = out / 'chunk_000000.pt'
    first_mtime = first.stat().st_mtime_ns
    (out / converter.COMPLETE_MARKER).unlink()
    (out / 'chunk_000002.pt').unlink()
    converter.main()
    assert first.stat().st_mtime_ns == first_mtime
    actual = [torch.load(p, weights_only=False) for p in sorted(out.glob('chunk_*.pt'))]
    assert [caption for group in actual for caption in group['caption']] == captions
    for before, after in zip(expected, actual):
        for key in ['pixel_values', 'input_ids', 'attention_mask']:
            assert torch.equal(before[key], after[key])


def test_cli_limit_counts_successfully_decoded_samples(tmp_path, tokenizer, monkeypatch):
    _, captions = make_shards(tmp_path / 'raw', corrupt=True)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'cc12m_wds_to_chunks.py', '--wds', str(tmp_path / 'raw'), '--out', str(out),
        '--chunk-size', '6', '--image-size', '32', '--prompt-length', '16',
        '--decode-workers', '3', '--async-write', '--limit', '6'])
    converter.main()
    payload = torch.load(out / 'chunk_000000.pt', weights_only=False)
    assert payload['caption'] == captions[:6]
    assert len(list(out.glob('chunk_*.pt'))) == 1
    assert not (out / converter.COMPLETE_MARKER).exists()
