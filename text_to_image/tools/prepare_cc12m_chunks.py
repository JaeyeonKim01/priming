from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def first_existing(row: dict, candidates: list[str]) -> str:
    for name in candidates:
        if name in row and row[name] is not None:
            return name
    raise KeyError(f"none of these columns were found: {', '.join(candidates)}")


def to_image(value, *, timeout: int = 20) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        request = urllib.request.Request(value, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    return Image.open(value).convert("RGB")


def to_image_with_retries(value, *, timeout: int, retries: int, retry_delay: float) -> Image.Image:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return to_image(value, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt < retries and retry_delay > 0:
                time.sleep(retry_delay)
    assert last_exc is not None
    raise last_exc


def write_chunk(images, captions, out_dir: Path, chunk_idx: int, tokenizer, prompt_length: int):
    tokens = tokenizer(
        captions,
        max_length=prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    payload = {
        "pixel_values": torch.stack(images, dim=0),
        "input_ids": tokens.input_ids,
        "attention_mask": tokens.attention_mask,
        "caption": captions,
    }
    path = out_dir / f"chunk_{chunk_idx:06d}.pt"
    torch.save(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare MiniT2I pretraining tensor chunks from a Hugging Face image/text dataset.")
    parser.add_argument("--dataset", default="CaptionEmporium/conceptual-captions-cc12m-llavanext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-column", default="")
    parser.add_argument("--caption-column", default="")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--tokenizer", default="google/flan-t5-large")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=0, help="Retry each failed image download this many times before skipping it.")
    parser.add_argument("--retry-delay", type=float, default=0.0, help="Seconds to sleep between retries.")
    parser.add_argument("--max-download-failures", type=int, default=0, help="Abort after this many image load failures. 0 means keep skipping bad samples.")
    parser.add_argument("--log-every", type=int, default=1000, help="Print progress every N input rows. 0 disables periodic progress logs.")
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.prompt_length)
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.PILToTensor(),
        ]
    )

    image_column = args.image_column
    caption_column = args.caption_column
    image_candidates = ["image", "jpg", "jpeg", "png", "url"]
    caption_candidates = ["caption_llava", "caption_llava_short", "text", "caption", "llava_caption", "recaption", "re_caption", "prompt"]

    images = []
    captions = []
    chunk_idx = 0
    total = 0
    kept = 0
    skipped = 0
    for row in dataset:
        if not image_column:
            image_column = first_existing(row, image_candidates)
        if not caption_column:
            caption_column = first_existing(row, caption_candidates)
        total += 1

        try:
            image = to_image_with_retries(
                row[image_column],
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
        except Exception as exc:
            skipped += 1
            key = row.get("key", total)
            print(f"skipping sample {key}: failed to load image from {row[image_column]!r}: {exc}", file=sys.stderr, flush=True)
            if args.max_download_failures and skipped >= args.max_download_failures:
                raise RuntimeError(f"reached --max-download-failures={args.max_download_failures}") from exc
            if args.limit and total >= args.limit:
                break
            continue

        images.append(transform(image))
        captions.append(str(row[caption_column]))
        kept += 1

        if len(images) == args.chunk_size:
            path = write_chunk(images, captions, out_dir, chunk_idx, tokenizer, args.prompt_length)
            print(f"wrote {path} ({kept} kept, {skipped} skipped, {total} seen)", flush=True)
            images.clear()
            captions.clear()
            chunk_idx += 1
        if args.log_every and total % args.log_every == 0:
            print(f"processed {total} rows ({kept} kept, {skipped} skipped)", flush=True)
        if args.limit and total >= args.limit:
            break

    if images:
        path = write_chunk(images, captions, out_dir, chunk_idx, tokenizer, args.prompt_length)
        print(f"wrote {path} ({kept} kept, {skipped} skipped, {total} seen)", flush=True)
    print(f"done: {kept} kept, {skipped} skipped, {total} seen", flush=True)


if __name__ == "__main__":
    main()
