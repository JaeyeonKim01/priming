from __future__ import annotations

import argparse
import ast
import base64
import io
import shutil
import tarfile
from math import ceil
from pathlib import Path


DATASETS = {
    "blip3_ft60k": "BLIP3o/BLIP3o-60k",
    "dalle3": "OpenDatasets/dalle-3-dataset",
    "sharegpt4o": "FreedomIntelligence/ShareGPT-4o-Image",
}

EXPECTED = {
    "blip3_ft60k": [
        "dalle3.tar",
        "geneval_train.tar",
        "human_gestures.tar",
        "journeyDB.tar",
        "mscoco_human.tar",
        "object_1.tar",
        "object_2.tar",
        "occupation_1.tar",
        "occupation_2.tar",
        "text_1.tar",
        "text_2.tar",
    ],
    "dalle3": ["shard-*.tar"],
    "sharegpt4o": ["text_to_image_part_*.tar", "text_to_image.json"],
}


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        shutil.copy2(src, tmp)
        tmp.rename(dst)
    else:
        dst.symlink_to(src.resolve())


def has_image_text_pair(path: Path) -> bool:
    if not tarfile.is_tarfile(path):
        return False
    try:
        with tarfile.open(path) as tar:
            names = [member.name.lower() for _, member in zip(range(512), tar) if member.isfile()]
    except Exception:
        return False
    return any(name.endswith((".jpg", ".jpeg", ".png")) for name in names) and any(
        name.endswith(".txt") for name in names
    )


def allow_patterns_for_source(source: str) -> list[str]:
    if source == "sharegpt4o":
        return ["text_to_image_part_*.tar", "text_to_image.json"]
    return ["*.tar", "**/*.tar"]


def tar_files_for_source(snapshot: Path, source: str) -> list[Path]:
    if source == "sharegpt4o":
        return sorted(snapshot.rglob("text_to_image_part_*.tar"))
    return sorted(snapshot.rglob("*.tar"))


def coerce_image_bytes(raw, base_dir: Path) -> bytes | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("bytes") is not None:
            return coerce_image_bytes(raw["bytes"], base_dir)
        if raw.get("path"):
            path = base_dir / raw["path"]
            return path.read_bytes() if path.exists() else None
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if isinstance(raw, str):
        if raw.startswith("data:image"):
            try:
                return base64.b64decode(raw.split(",", 1)[1], validate=False)
            except Exception:
                return None
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            pass
        try:
            literal = ast.literal_eval(raw)
            if isinstance(literal, (list, tuple)):
                return bytes(literal)
        except Exception:
            pass
    if isinstance(raw, (list, tuple)):
        try:
            return bytes(raw)
        except Exception:
            return None
    return None


def write_pair(tar: tarfile.TarFile, key: str, image_bytes: bytes, caption: str) -> None:
    image_info = tarfile.TarInfo(f"{key}.jpg")
    image_info.size = len(image_bytes)
    tar.addfile(image_info, io.BytesIO(image_bytes))

    text_bytes = (caption.strip() + "\n").encode("utf-8")
    text_info = tarfile.TarInfo(f"{key}.txt")
    text_info.size = len(text_bytes)
    tar.addfile(text_info, io.BytesIO(text_bytes))


def stage_dalle3_webdataset(repo_id: str, cache_dir: str, out_root: Path) -> None:
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq

    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["*.parquet", "**/*.parquet"],
        )
    )
    parquets = sorted(snapshot.rglob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"no .parquet files found in {repo_id}")

    dst_dir = out_root / "dalle3"
    dst_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    num_shards = ceil(len(parquets) / 5)
    for shard_idx in range(num_shards):
        shard_path = dst_dir / f"shard-{shard_idx:03d}.tar"
        done_path = shard_path.with_suffix(".tar.done")
        if shard_path.exists() and done_path.exists():
            if has_image_text_pair(shard_path):
                print(f"using existing dalle3 shard -> {shard_path}", flush=True)
                continue
            print(f"removing incomplete/corrupt dalle3 shard -> {shard_path}", flush=True)
            shard_path.unlink()
            done_path.unlink(missing_ok=True)
        elif shard_path.exists():
            print(f"removing unmarked dalle3 shard before rebuild -> {shard_path}", flush=True)
            shard_path.unlink()
        tmp_path = shard_path.with_suffix(".tar.tmp")
        if tmp_path.exists():
            print(f"removing stale temporary dalle3 shard -> {tmp_path}", flush=True)
            tmp_path.unlink()
        done_tmp_path = done_path.with_suffix(done_path.suffix + ".tmp")
        done_tmp_path.unlink(missing_ok=True)
        shard_parquets = parquets[shard_idx * 5 : (shard_idx + 1) * 5]
        written = 0
        with tarfile.open(tmp_path, "w") as tar:
            global_idx = shard_idx * 1_000_000
            for parquet_path in shard_parquets:
                parquet = pq.ParquetFile(parquet_path)
                row_offset = 0
                for batch in parquet.iter_batches(batch_size=1024):
                    rows = batch.to_pylist()
                    for local_offset, row in enumerate(rows):
                        image_bytes = coerce_image_bytes(row.get("image"), snapshot)
                        if not image_bytes:
                            errors.append(f"{parquet_path.name}:{row_offset + local_offset}:image")
                            continue
                        caption = row.get("caption") or row.get("synthetic_caption") or ""
                        key = f"{row.get('image_hash') or 'sample'}-{global_idx:012d}"
                        write_pair(tar, str(key), image_bytes, str(caption))
                        global_idx += 1
                        written += 1
                    row_offset += len(rows)
        tmp_path.rename(shard_path)
        done_tmp_path.write_text(f"samples={written}\n", encoding="utf-8")
        done_tmp_path.rename(done_path)
        print(f"staged {written} samples for dalle3 -> {shard_path}", flush=True)

    if errors:
        error_path = dst_dir / "errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"warning: wrote {len(errors)} dalle3 conversion errors to {error_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 120K mix WebDataset tar files in the layout expected by configs/finetune.yml.")
    parser.add_argument("--out", required=True, help="Destination directory used as FINETUNE_DATA_ROOT in mini_t2i/settings.py.")
    parser.add_argument("--cache-dir", default="hf_cache")
    parser.add_argument("--copy", action="store_true", help="Copy tar files instead of symlinking them from the Hugging Face cache.")
    parser.add_argument("--source", choices=sorted(DATASETS), action="append", default=None)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out_root = Path(args.out)
    sources = args.source or list(DATASETS)
    for name in sources:
        repo_id = DATASETS[name]
        if name == "dalle3":
            stage_dalle3_webdataset(repo_id, args.cache_dir, out_root)
            continue
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=args.cache_dir,
                allow_patterns=allow_patterns_for_source(name),
            )
        )
        tar_files = tar_files_for_source(snapshot, name)
        if not tar_files:
            raise RuntimeError(f"no .tar files found in {repo_id}")
        dst_dir = out_root / name
        for src in tar_files:
            link_or_copy(src, dst_dir / src.name, args.copy)
        if name == "sharegpt4o":
            json_files = sorted(snapshot.rglob("text_to_image.json"))
            if not json_files:
                raise RuntimeError(f"no text_to_image.json found in {repo_id}")
            link_or_copy(json_files[0], dst_dir / json_files[0].name, args.copy)
        print(f"staged {len(tar_files)} tar files for {name} -> {dst_dir}", flush=True)

        missing = []
        for pattern in EXPECTED[name]:
            if not list(dst_dir.glob(pattern)):
                missing.append(pattern)
        if missing:
            print(f"warning: {name} is missing expected pattern(s): {', '.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
