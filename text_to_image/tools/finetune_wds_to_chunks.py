from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_t2i.config import TrainConfig
from mini_t2i.data import check_training_data, make_loader


COMPLETE_MARKER = "_finetune_wds_to_chunks_complete.json"


def chunk_index(path: Path) -> int:
    return int(path.stem.removeprefix("chunk_"))


def existing_chunks(out_dir: Path) -> list[Path]:
    chunks = sorted(out_dir.glob("chunk_*.pt"))
    for expected, path in enumerate(chunks):
        if chunk_index(path) != expected:
            raise RuntimeError(
                f"non-contiguous output at {path}; expected chunk_{expected:06d}.pt"
            )
    return chunks


def chunk_length(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["pixel_values"].shape[0])


def source_counts_for_chunks(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counts.update(str(source) for source in payload.get("source", []))
    return counts


def resume_position(out_dir: Path, chunk_size: int) -> tuple[int, int]:
    chunks = existing_chunks(out_dir)
    if not chunks:
        return 0, 0
    last_count = chunk_length(chunks[-1])
    if last_count > chunk_size:
        raise RuntimeError(f"{chunks[-1]} contains {last_count} samples, expected <= {chunk_size}")
    if last_count < chunk_size:
        # A partial chunk is only valid alongside the completion marker. If a
        # run stopped between its final rename and marker write, regenerate it.
        print(f"removing unmarked partial chunk {chunks[-1]} ({last_count} samples)", flush=True)
        chunks[-1].unlink()
        chunks.pop()
    return len(chunks), len(chunks) * chunk_size


def write_chunk(out_dir: Path, index: int, parts: list[dict]) -> Path:
    payload = {
        "pixel_values": torch.cat([part["pixel_values"] for part in parts], dim=0),
        "input_ids": torch.cat([part["input_ids"] for part in parts], dim=0),
        "attention_mask": torch.cat([part["attention_mask"] for part in parts], dim=0),
        "caption": [caption for part in parts for caption in part["caption"]],
        "source": [source for part in parts for source in part["source"]],
    }
    path = out_dir / f"chunk_{index:06d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the weighted fine-tuning WebDataset stream as resized uint8 images "
            "and token tensors consumable by the local_folder backend."
        )
    )
    parser.add_argument("--config-file", default=str(Path(__file__).resolve().parents[1] / "configs" / "finetune_data.yml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-samples", type=int, default=120_000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--loader-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=32,
        help=(
            "Per-source WebDataset shuffle buffer used only during conversion. "
            "Keep this modest because every loader worker buffers all three sources; "
            "the materialized chunks are shuffled again during training."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.total_samples <= 0 or args.chunk_size <= 0 or args.loader_batch_size <= 0:
        parser.error("sample and batch sizes must be positive")
    if args.chunk_size % args.loader_batch_size != 0:
        parser.error("--chunk-size must be divisible by --loader-batch-size")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_path = out_dir / COMPLETE_MARKER
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if int(marker.get("total_samples", -1)) == args.total_samples:
            print(f"complete marker exists at {marker_path}; nothing to do", flush=True)
            return
        raise RuntimeError(
            f"{marker_path} was generated for {marker.get('total_samples')} samples, "
            f"not {args.total_samples}"
        )

    chunks = existing_chunks(out_dir)
    if chunks and not args.resume:
        raise RuntimeError(f"{out_dir} already contains chunks; pass --resume to continue")
    chunk_idx, skip_samples = resume_position(out_dir, args.chunk_size) if args.resume else (0, 0)
    if skip_samples >= args.total_samples:
        raise RuntimeError(
            f"existing full chunks contain {skip_samples} samples, not fewer than target "
            f"{args.total_samples}"
        )

    cfg = TrainConfig().update_from_yaml(args.config_file)
    cfg.dataset_backend = "finetune_wds"
    cfg.micro_batch_size = args.loader_batch_size
    cfg.num_workers = args.num_workers
    if args.shuffle_buffer <= 0:
        parser.error("--shuffle-buffer must be positive")
    cfg.shuffle_buffer = args.shuffle_buffer
    check_training_data(cfg)
    loader = make_loader(cfg)
    iterator = iter(loader)

    source_counts = source_counts_for_chunks(existing_chunks(out_dir)) if skip_samples else Counter()
    if sum(source_counts.values()) != skip_samples:
        raise RuntimeError(
            f"existing chunks contain {sum(source_counts.values())} source labels for "
            f"{skip_samples} samples"
        )

    if skip_samples:
        print(f"resuming after {skip_samples} samples in {chunk_idx} chunks", flush=True)
        for _ in range(skip_samples // args.loader_batch_size):
            next(iterator)

    written_samples = skip_samples
    parts: list[dict] = []
    buffered = 0
    while written_samples < args.total_samples:
        batch = next(iterator)
        take = min(args.loader_batch_size, args.total_samples - written_samples - buffered)
        sources = [str(source) for source in batch["source"][:take]]
        parts.append(
            {
                "pixel_values": batch["pixel_values"][:take].clone(),
                "input_ids": batch["input_ids"][:take].clone(),
                "attention_mask": batch["attention_mask"][:take].clone(),
                "caption": list(batch["caption"][:take]),
                "source": sources,
            }
        )
        source_counts.update(sources)
        buffered += take
        if buffered == args.chunk_size or written_samples + buffered == args.total_samples:
            path = write_chunk(out_dir, chunk_idx, parts)
            written_samples += buffered
            chunk_idx += 1
            print(
                f"wrote {path} ({written_samples}/{args.total_samples} samples, "
                f"{chunk_idx} chunks)",
                flush=True,
            )
            parts = []
            buffered = 0

    marker = {
        "chunk_size": args.chunk_size,
        "config_file": str(Path(args.config_file).resolve()),
        "finetune_mix_weights": cfg.finetune_mix_weights,
        "finetune_sources": cfg.finetune_sources,
        "image_size": cfg.image_size,
        "prompt_length": cfg.prompt_length,
        "shuffle_buffer": cfg.shuffle_buffer,
        "source_counts": dict(sorted(source_counts.items())),
        "t5_name": cfg.t5_name,
        "total_chunks": chunk_idx,
        "total_samples": written_samples,
    }
    tmp_marker = marker_path.with_suffix(".json.tmp")
    tmp_marker.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    tmp_marker.rename(marker_path)
    print(f"done: {written_samples} samples -> {chunk_idx} chunks in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
