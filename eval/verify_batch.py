"""Validate a new sampling batch without changing its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DENOM = {"FID(ADM)": 1.68, "dinov2_cls": 14.19, "mae_cls": .04,
         "siglip_cls": .60, "clip_cls": 5.60}
REFERENCE_HASH = "412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_one(batch: Path, key: str, count: int, seeds: list[int]) -> dict:
    streams = set()
    results = []
    checkpoint = None
    for seed in seeds:
        out = batch / key / f"n{count}_seed{seed}"
        report = json.loads((out / "results.json").read_text())
        assert report["model"] == key and report["seed"] == seed
        assert report["num_images"] == count and report["script_sha256"] == digest(HERE / "evaluate.py")
        assert report["official_reference_sha256"] == REFERENCE_HASH
        assert set(report["fd_legacy"]) == set(DENOM)
        fdr5 = sum(report["fd_legacy"][name] / value for name, value in DENOM.items()) / 5
        assert math.isclose(report["fdr5"], fdr5, rel_tol=0, abs_tol=1e-10)
        assert all(math.isfinite(report[m]) and report[m] > 0 for m in ("fid", "fdr5"))
        if checkpoint is None:
            checkpoint = report["checkpoint"]
        assert report["checkpoint"] == checkpoint
        if key.startswith("priming"):
            manifest = json.loads((HERE.parent / "checkpoints/manifest.json").read_text())
            accepted = {manifest[key]["checkpoint_sha256"]}
            previous = manifest[key].get("previous_ema_export")
            if previous:
                accepted.add(previous["sha256"])
            assert report["checkpoint_sha256"] in accepted
        ids = np.concatenate([np.load(out / f"ids_rank{rank}.npy")
                              for rank in range(report["world_size"])])
        assert np.array_equal(np.sort(ids), np.arange(count))
        per_class = count // 1000
        assert np.array_equal(np.bincount(ids // per_class), np.full(1000, per_class))
        ranks = [json.loads((out / f"rank{rank}.json").read_text())
                 for rank in range(report["world_size"])]
        assert sum(r["num_images"] for r in ranks) == count
        stream = tuple(r["image_bytes_sha256"] for r in ranks)
        assert stream not in streams
        streams.add(stream)
        with np.load(out / "official_generated_stats.npz") as moments:
            assert moments["mu"].shape == (2048,)
            assert moments["sigma"].shape == (2048, 2048)
            assert np.isfinite(moments["mu"]).all() and np.isfinite(moments["sigma"]).all()
        results.append(report)
    summary = {"model": key, "replicates": len(results), "num_images_per_seed": count,
               "status": "pass"}
    if len(results) > 1:
        for metric in ("fid", "fdr5"):
            summary[f"{metric}_mean"] = statistics.mean(r[metric] for r in results)
            summary[f"{metric}_sd"] = statistics.stdev(r[metric] for r in results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    options = parser.parse_args()
    assert options.num_images % 1000 == 0 and len(set(options.seeds)) == len(options.seeds)
    print(json.dumps(verify_one(options.output_dir, options.model,
                                options.num_images, options.seeds), indent=2))


if __name__ == "__main__":
    main()
