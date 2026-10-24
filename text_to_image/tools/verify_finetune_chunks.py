from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_t2i.config import TrainConfig
from mini_t2i.data import check_training_data
from mini_t2i.datasets.finetune import source_files_for_root
from tools.finetune_wds_to_chunks import COMPLETE_MARKER, chunk_index


VERIFIED_MARKER = "_finetune_chunks_verified.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every tensor chunk in a materialized fine-tuning mixture."
    )
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--config-file", default=str(Path(__file__).resolve().parents[1] / "configs" / "finetune_data.yml"))
    parser.add_argument("--expected-samples", type=int, default=120_000)
    parser.add_argument(
        "--max-source-z-score",
        type=float,
        default=6.0,
        help="Maximum binomial z-score allowed between realized and configured source counts.",
    )
    args = parser.parse_args()

    root = Path(args.chunks)
    require(root.is_dir(), f"chunk directory does not exist: {root}")
    require(args.expected_samples > 0, "--expected-samples must be positive")

    cfg = TrainConfig().update_from_yaml(args.config_file)
    cfg.dataset_backend = "finetune_wds"
    check_training_data(cfg)

    complete_path = root / COMPLETE_MARKER
    require(complete_path.is_file(), f"missing converter completion marker: {complete_path}")
    complete = json.loads(complete_path.read_text())
    require(int(complete.get("total_samples", -1)) == args.expected_samples, "wrong total_samples in marker")
    require(complete.get("finetune_sources") == cfg.finetune_sources, "marker source list differs from config")
    require(complete.get("finetune_mix_weights") == cfg.finetune_mix_weights, "marker weights differ from config")
    require(int(complete.get("image_size", -1)) == cfg.image_size, "marker image size differs from config")
    require(int(complete.get("prompt_length", -1)) == cfg.prompt_length, "marker prompt length differs from config")
    require(complete.get("t5_name") == cfg.t5_name, "marker tokenizer differs from config")

    chunk_size = int(complete["chunk_size"])
    expected_chunks = math.ceil(args.expected_samples / chunk_size)
    chunks = sorted(root.glob("chunk_*.pt"))
    require(len(chunks) == expected_chunks, f"found {len(chunks)} chunks, expected {expected_chunks}")
    require(int(complete.get("total_chunks", -1)) == expected_chunks, "wrong total_chunks in marker")
    require(not list(root.glob("*.tmp")), "temporary files remain in chunk directory")

    source_counts: Counter[str] = Counter()
    empty_captions = 0
    total_samples = 0
    total_bytes = 0
    allowed_sources = set(cfg.finetune_sources)

    for expected_index, path in enumerate(chunks):
        require(chunk_index(path) == expected_index, f"non-contiguous chunk name: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"pixel_values", "input_ids", "attention_mask", "caption", "source"}
        missing = required - set(payload)
        require(not missing, f"{path.name} is missing keys: {sorted(missing)}")

        expected_n = min(chunk_size, args.expected_samples - total_samples)
        pixels = payload["pixel_values"]
        input_ids = payload["input_ids"]
        attention_mask = payload["attention_mask"]
        captions = payload["caption"]
        sources = payload["source"]

        require(
            tuple(pixels.shape) == (expected_n, 3, cfg.image_size, cfg.image_size),
            f"{path.name} has wrong pixel shape {tuple(pixels.shape)}",
        )
        require(pixels.dtype == torch.uint8, f"{path.name} pixels have dtype {pixels.dtype}")
        require(
            tuple(input_ids.shape) == (expected_n, cfg.prompt_length),
            f"{path.name} has wrong input_ids shape {tuple(input_ids.shape)}",
        )
        require(
            tuple(attention_mask.shape) == (expected_n, cfg.prompt_length),
            f"{path.name} has wrong attention_mask shape {tuple(attention_mask.shape)}",
        )
        require(input_ids.dtype == torch.int64, f"{path.name} input_ids have dtype {input_ids.dtype}")
        require(
            attention_mask.dtype == torch.int64,
            f"{path.name} attention_mask has dtype {attention_mask.dtype}",
        )
        require(len(captions) == expected_n, f"{path.name} has {len(captions)} captions")
        require(len(sources) == expected_n, f"{path.name} has {len(sources)} source labels")
        require(all(isinstance(caption, str) for caption in captions), f"{path.name} has non-string captions")
        unknown_sources = set(sources) - allowed_sources
        require(not unknown_sources, f"{path.name} has unknown sources: {sorted(unknown_sources)}")

        empty_captions += sum(not caption.strip() for caption in captions)
        source_counts.update(str(source) for source in sources)
        total_samples += expected_n
        total_bytes += path.stat().st_size
        print(
            f"verified {path.name}: {total_samples}/{args.expected_samples} samples",
            flush=True,
        )

    require(total_samples == args.expected_samples, f"verified {total_samples} samples")
    require(sum(source_counts.values()) == args.expected_samples, "source counts do not sum to total")
    marker_counts = {str(key): int(value) for key, value in complete.get("source_counts", {}).items()}
    require(dict(source_counts) == marker_counts, "scanned source counts differ from completion marker")

    weight_sum = float(sum(cfg.finetune_mix_weights))
    probabilities = {
        source: float(weight) / weight_sum
        for source, weight in zip(cfg.finetune_sources, cfg.finetune_mix_weights)
    }
    expected_source_counts = {
        source: args.expected_samples * probability for source, probability in probabilities.items()
    }
    source_z_scores = {}
    for source, probability in probabilities.items():
        variance = args.expected_samples * probability * (1.0 - probability)
        z_score = (source_counts[source] - expected_source_counts[source]) / math.sqrt(variance)
        source_z_scores[source] = z_score
        require(
            abs(z_score) <= args.max_source_z_score,
            f"{source} count {source_counts[source]} is {z_score:.2f} standard deviations "
            "from its configured proportion",
        )

    source_root = Path(cfg.finetune_dataset_dir)
    source_shards = {
        source: [str(path.resolve()) for path in source_files_for_root(source_root, source)]
        for source in cfg.finetune_sources
    }
    require(all(source_shards.values()), "one or more configured sources have no local shards")

    result = {
        "chunk_size": chunk_size,
        "empty_captions": empty_captions,
        "expected_source_counts": expected_source_counts,
        "finetune_dataset_dir": str(source_root.resolve()),
        "finetune_mix_weights": cfg.finetune_mix_weights,
        "finetune_sources": cfg.finetune_sources,
        "image_size": cfg.image_size,
        "prompt_length": cfg.prompt_length,
        "source_counts": dict(source_counts),
        "source_shards": source_shards,
        "source_z_scores": source_z_scores,
        "total_bytes": total_bytes,
        "total_chunks": len(chunks),
        "total_samples": total_samples,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    verified_path = root / VERIFIED_MARKER
    tmp = verified_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.rename(verified_path)
    print("FINETUNE_CHUNKS_VERIFIED " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
