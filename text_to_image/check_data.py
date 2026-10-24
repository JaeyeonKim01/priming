"""Read-only structural check of prepared T2I training datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mini_t2i import settings


def check_chunks(root: Path, *, name: str, completion: str, verification: str, expected_samples: int | None = None) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(f"{name} dataset directory missing: {root}")
    completed = json.loads((root / completion).read_text())
    verified = json.loads((root / verification).read_text())
    samples = int(completed["total_samples"])
    chunk_size = int(completed["chunk_size"])
    chunks = sorted(root.glob("chunk_*.pt"))
    expected_chunks = math.ceil(samples / chunk_size)
    if samples < 1 or chunk_size < 1 or expected_samples is not None and samples != expected_samples:
        raise ValueError(f"{name}: unexpected sample count or chunk size")
    if int(completed["total_chunks"]) != expected_chunks or len(chunks) != expected_chunks:
        raise ValueError(f"{name}: completion marker and chunk count disagree")
    if [path.name for path in chunks] != [f"chunk_{index:06d}.pt" for index in range(expected_chunks)]:
        raise ValueError(f"{name}: chunk filenames are not contiguous")
    if any(root.glob("*.tmp")):
        raise ValueError(f"{name}: temporary files remain")
    if (int(completed["image_size"]), int(completed["prompt_length"])) != (512, 256):
        raise ValueError(f"{name}: expected 512-pixel images and 256-token captions")
    tokenizer = completed.get("tokenizer", completed.get("t5_name"))
    if tokenizer != "google/flan-t5-large":
        raise ValueError(f"{name}: unexpected tokenizer {tokenizer!r}")
    total_bytes = sum(path.stat().st_size for path in chunks)
    for field, value in (("total_samples", samples), ("total_chunks", expected_chunks), ("total_bytes", total_bytes)):
        if int(verified[field]) != value:
            raise ValueError(f"{name}: verification record disagrees on {field}")
    if "source_counts" in completed and completed["source_counts"] != verified.get("source_counts"):
        raise ValueError(f"{name}: source counts differ from the verification record")
    return {"path": str(root.resolve()), "samples": samples, "chunks": expected_chunks,
            "bytes": total_bytes, "verified_at": verified.get("verified_at")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=settings.STORAGE_ROOT)
    args = parser.parse_args()
    data = args.storage_root / "data"
    summary = {
        "cc12m": check_chunks(data / "cc12m_tensor_chunks", name="CC12M",
                              completion="_cc12m_wds_to_chunks_complete.json",
                              verification="_cc12m_chunks_verified.json"),
        "finetune_120k": check_chunks(data / "finetune_tensor_chunks_120k", name="120K mix",
                                      completion="_finetune_wds_to_chunks_complete.json",
                                      verification="_finetune_chunks_verified.json", expected_samples=120000),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
