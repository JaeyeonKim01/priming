"""Verify release checkpoint/config checksums; optionally verify full sources."""

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="store_true",
                        help="Also hash the large original training checkpoints")
    options = parser.parse_args()
    manifest = json.loads((HERE / "checkpoints/manifest.json").read_text())
    assert len(manifest) == 4
    for key, item in sorted(manifest.items()):
        weights = HERE / item["checkpoint"]
        config = HERE / item["config"]
        assert weights.is_symlink() and weights.readlink() == Path(item["link_target"])
        assert weights.resolve() == Path(item["source_checkpoint"]).resolve()
        assert weights.stat().st_size == item["checkpoint_bytes"]
        assert digest(weights) == item["checkpoint_sha256"], key
        assert digest(config) == item["config_sha256"], key
        parsed = json.loads(config.read_text())
        assert parsed["completed_updates"] == 25000 and parsed["ema"] == "0.999"
        if options.sources:
            source = Path(item["source_checkpoint"])
            assert source.stat().st_size == item["source_checkpoint_bytes"]
            assert digest(source) == item["source_checkpoint_sha256"], key
            assert digest(Path(item["source_args"])) == item["source_args_sha256"], key
        print(f"{key}: verified", flush=True)
    print(f"All {len(manifest)} checkpoint packages verified")


if __name__ == "__main__":
    main()
