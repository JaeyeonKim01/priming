from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import time
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
COMPLETE_MARKER = "_cc12m_wds_to_chunks_complete.json"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_transform(image_size: int):
    # Matches the transform used everywhere else in the repo (see
    # mini_t2i/datasets/finetune.py): resize shorter side -> image_size with
    # BICUBIC, center crop to a square, keep uint8 pixels via PILToTensor.
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
        ]
    )


def to_sample(sample: dict, transform):
    image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected decoded PIL image, got {type(image)!r}")
    caption = sample.get("txt", "")
    if isinstance(caption, bytes):
        caption = caption.decode("utf-8", errors="replace")
    return transform(image.convert("RGB")), str(caption)


def write_chunk(images, captions, out_dir: Path, chunk_idx: int, tokenizer, prompt_length: int,
                timings: dict | None = None) -> Path:
    started = time.perf_counter()
    tokens = tokenizer(
        captions,
        max_length=prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized = time.perf_counter()
    pixels = torch.stack(images, dim=0)
    stacked = time.perf_counter()
    payload = {
        "pixel_values": pixels,
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "caption": captions,
    }
    path = out_dir / f"chunk_{chunk_idx:06d}.pt"
    # Write to a temp file then rename so an interrupted job never leaves a
    # half-written chunk that the loader would choke on.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    if timings is not None:
        timings['tokenize_seconds'] += tokenized - started
        timings['stack_seconds'] += stacked - tokenized
        timings['write_seconds'] += time.perf_counter() - stacked
    return path


def iter_decoded_samples(dataset, transform, workers: int = 0, prefetch: int | None = None):
    """Decode concurrently with bounded memory and retain exact input order."""
    if workers == 0:
        yield from dataset.decode('pil', handler=wds.warn_and_continue).map(
            lambda sample: to_sample(sample, transform), handler=wds.warn_and_continue)
        return
    if workers < 0 or (prefetch is not None and prefetch < 1):
        raise ValueError('workers must be nonnegative and prefetch must be positive')
    # This is the same stateless decoder used by WebDataset.decode('pil').
    decoder = wds.autodecode.Decoder([wds.autodecode.imagehandler('pil')])

    def decode_one(sample):
        try:
            return to_sample(decoder(sample), transform)
        except Exception as exc:
            if wds.warn_and_continue(exc):
                return None
            raise

    source = iter(dataset)
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(prefetch or workers * 4):
            try:
                pending.append(executor.submit(decode_one, next(source)))
            except StopIteration:
                break
        while pending:
            result = pending.popleft().result()
            try:
                pending.append(executor.submit(decode_one, next(source)))
            except StopIteration:
                pass
            if result is not None:
                yield result


def convert_samples(samples, out_dir: Path, tokenizer, prompt_length: int, chunk_size: int,
                    start_chunk: int = 0, skip_samples: int = 0, async_write: bool = False):
    """Optionally overlap one chunk write with decoding the following chunk."""
    timings = {key: 0.0 for key in ['decode_wait_seconds', 'tokenize_seconds',
                                   'stack_seconds', 'write_seconds']}
    started = time.perf_counter()
    images, captions = [], []
    chunk_idx, new_samples, written = start_chunk, 0, 0
    pending = None

    def commit(chunk_images, chunk_captions, index, total):
        path = write_chunk(chunk_images, chunk_captions, out_dir, index, tokenizer,
                           prompt_length, timings)
        log(f'wrote {path}; total_samples={total}, total_chunks={index + 1}')

    with (ThreadPoolExecutor(max_workers=1) if async_write else nullcontext()) as writer:
        source = iter(samples)
        while True:
            before_read = time.perf_counter()
            try:
                image, caption = next(source)
            except StopIteration:
                break
            timings['decode_wait_seconds'] += time.perf_counter() - before_read
            images.append(image)
            captions.append(caption)
            new_samples += 1
            if len(images) == chunk_size:
                if pending is not None:
                    pending.result()
                if writer is None:
                    commit(images, captions, chunk_idx, skip_samples + new_samples)
                else:
                    pending = writer.submit(commit, images, captions, chunk_idx,
                                            skip_samples + new_samples)
                images, captions = [], []
                chunk_idx += 1
                written += 1
        if pending is not None:
            pending.result()
        if images:
            commit(images, captions, chunk_idx, skip_samples + new_samples)
            chunk_idx += 1
            written += 1
    timings['elapsed_seconds'] = time.perf_counter() - started
    timings['images_per_second'] = new_samples / max(timings['elapsed_seconds'], 1e-9)
    return new_samples, written, chunk_idx, timings


def chunk_index(path: Path) -> int:
    return int(path.stem.removeprefix("chunk_"))


def chunk_sample_count(path: Path) -> int:
    chunk = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return int(chunk["pixel_values"].shape[0])


def existing_contiguous_chunks(out_dir: Path) -> list[Path]:
    chunks = sorted(out_dir.glob("chunk_*.pt"))
    for expected, path in enumerate(chunks):
        if chunk_index(path) != expected:
            raise SystemExit(
                f"found non-contiguous chunk sequence at {path}; expected chunk_{expected:06d}.pt. "
                "Move stale chunks aside or rebuild the output directory."
            )
    return chunks


def expected_marker(args, shards: list[str]) -> dict:
    return {
        "chunk_size": args.chunk_size,
        "image_size": args.image_size,
        "prompt_length": args.prompt_length,
        "tokenizer": args.tokenizer,
        "shards": [Path(s).name for s in shards],
    }


def marker_matches(marker: dict, expected: dict) -> bool:
    return all(marker.get(key) == value for key, value in expected.items())


def write_complete_marker(out_dir: Path, args, shards: list[str], total_samples: int, total_chunks: int) -> None:
    payload = expected_marker(args, shards)
    payload.update({"total_samples": total_samples, "total_chunks": total_chunks})
    path = out_dir / COMPLETE_MARKER
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.rename(path)


def resume_position(out_dir: Path, chunk_size: int) -> tuple[int, int]:
    chunks = existing_contiguous_chunks(out_dir)
    if not chunks:
        return 0, 0

    last = chunks[-1]
    last_count = chunk_sample_count(last)
    if last_count > chunk_size:
        raise SystemExit(f"{last} has {last_count} samples, more than chunk size {chunk_size}")
    if last_count < chunk_size:
        log(f"removing incomplete final chunk before resume: {last} ({last_count} samples)")
        last.unlink()
        chunks = chunks[:-1]

    return len(chunks), len(chunks) * chunk_size


def make_webdataset(shards: list[str]):
    try:
        return wds.WebDataset(
            shards,
            handler=wds.warn_and_continue,
            empty_check=False,
            shardshuffle=False,
        )
    except TypeError:
        return wds.WebDataset(shards, handler=wds.warn_and_continue, shardshuffle=False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert img2dataset WebDataset shards into MiniT2I chunk_*.pt tensor chunks "
        "for the local_folder pretraining backend."
    )
    parser.add_argument("--wds", required=True, help="Directory of img2dataset .tar shards.")
    parser.add_argument("--out", required=True, help="Destination dir for chunk_*.pt (CC12M_TENSOR_ROOT).")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many samples (0 = all).")
    parser.add_argument("--start-chunk", type=int, default=0, help="First output chunk index for appending.")
    parser.add_argument('--decode-workers', type=int, default=0,
                        help='Ordered JPEG decode/transform threads; 0 keeps the serial reference path.')
    parser.add_argument('--decode-prefetch', type=int, default=None)
    parser.add_argument('--async-write', action='store_true',
                        help='Overlap one chunk write with decoding the next chunk.')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete existing chunk_*.pt files and continue writing the same sequence.",
    )
    args = parser.parse_args()

    if args.decode_workers < 0 or (args.decode_prefetch is not None and args.decode_prefetch < 1):
        parser.error('decode-workers must be nonnegative and decode-prefetch positive')

    if args.resume and args.start_chunk != 0:
        raise SystemExit("--resume cannot be combined with --start-chunk")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(str(p) for p in Path(args.wds).glob("*.tar"))
    if not shards:
        raise SystemExit(f"no .tar shards found under {args.wds}")
    log(f"found {len(shards)} shard(s) under {args.wds}")

    marker_path = out_dir / COMPLETE_MARKER
    if args.resume and marker_path.exists():
        marker = json.loads(marker_path.read_text())
        expected = expected_marker(args, shards)
        if marker_matches(marker, expected):
            log(f"complete marker exists at {marker_path}; nothing to do")
            return
        raise SystemExit(
            f"{marker_path} does not match the current shard list or conversion settings. "
            "Move the old output aside or remove the marker and chunks to rebuild."
        )

    if args.resume:
        chunk_idx, skip_samples = resume_position(out_dir, args.chunk_size)
        if skip_samples:
            log(f"resuming at chunk {chunk_idx}; skipping {skip_samples} existing samples")
    else:
        chunk_idx = args.start_chunk
        skip_samples = 0

    log(
        f"chunk conversion status: start_chunk={chunk_idx}, "
        f"existing_chunks={chunk_idx}, skipped_samples={skip_samples}, chunk_size={args.chunk_size}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.prompt_length)
    transform = build_transform(args.image_size)

    dataset = make_webdataset(shards)
    samples = iter_decoded_samples(dataset, transform, args.decode_workers, args.decode_prefetch)
    if skip_samples or args.limit:
        # Chunks count successfully decoded samples. Apply the offset after
        # decoding so corrupt records cannot duplicate samples during resume.
        samples = islice(samples, skip_samples,
                         skip_samples + args.limit if args.limit else None)
    new_samples, written, chunk_idx, timings = convert_samples(
        samples, out_dir, tokenizer, args.prompt_length, args.chunk_size,
        chunk_idx, skip_samples, args.async_write)
    if args.profile:
        log('CONVERSION_PROFILE ' + json.dumps(timings, sort_keys=True))

    total_samples = skip_samples + new_samples
    if not args.limit:
        write_complete_marker(out_dir, args, shards, total_samples, chunk_idx)

    log(
        f"done: {total_samples} samples ({skip_samples} skipped, {new_samples} new) "
        f"-> {written} new chunks in {out_dir}"
    )


if __name__ == "__main__":
    main()
