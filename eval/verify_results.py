"""Read-only verification of the released three-seed ImageNet result package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DENOM = {"FID(ADM)": 1.68, "dinov2_cls": 14.19, "mae_cls": .04,
         "siglip_cls": .60, "clip_cls": 5.60}
EXPECTED_REFERENCE = "412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c"
KEYS = ("wflow_xl", "drifting_b", "drifting_l", "pmf_l", "pmf_h", "jit_h",
        "imf_xl", "priming_b_80", "priming_b_160", "priming_l_80", "priming_l_160")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(scratch_root: Path | None = None) -> dict:
    reports = HERE / "reports"
    summary = {r["model"]: r for r in csv.DictReader((reports / "summary.csv").open())}
    per_seed = {(r["model"], int(r["seed"])): r
                for r in csv.DictReader((reports / "per_seed.csv").open())}
    paper = list(csv.DictReader((HERE / "paper_results.csv").open()))
    assert len(summary) == len(paper) == 11 and len(per_seed) == 33
    paper_by_key = dict(zip(KEYS, paper, strict=True))
    identities = json.loads((HERE / 'provenance/evaluator_identity.json').read_text())
    for name, identity in identities.items():
        assert digest(HERE / 'archive' / name) == identity['public_sha256'], name
    frozen_hashes = {identity['executed_sha256'] for identity in identities.values()}
    artifacts = sorted(reports.glob("*_n50000_seed*.json"))
    assert len(artifacts) == 33
    checked = []
    for key in KEYS:
        values = []
        streams = set()
        checkpoint = None
        for seed in (1, 2, 3):
            result = json.loads((reports / f"{key}_n50000_seed{seed}.json").read_text())
            assert result["model"] == key and result["seed"] == seed
            assert result["num_images"] == 50000
            assert result["official_reference_sha256"] == EXPECTED_REFERENCE
            assert result["script_sha256"] in frozen_hashes
            assert set(result["fd_legacy"]) == set(DENOM)
            fdr5 = sum(result["fd_legacy"][name] / divisor
                       for name, divisor in DENOM.items()) / 5
            assert math.isclose(result["fdr5"], fdr5, rel_tol=0, abs_tol=1e-10)
            for metric in ("fid", "fdr5"):
                assert math.isfinite(result[metric]) and result[metric] > 0
                assert math.isclose(result[metric], float(per_seed[key, seed][metric]),
                                    rel_tol=0, abs_tol=1e-12)
            assert checkpoint is None or checkpoint == result["checkpoint"]
            checkpoint = result["checkpoint"]
            values.append(result)
            if scratch_root:
                out = scratch_root / key / f"n50000_seed{seed}"
                assert json.loads((out / "results.json").read_text()) == result
                ids = np.concatenate([np.load(out / f"ids_rank{rank}.npy")
                                      for rank in range(result["world_size"])])
                assert np.array_equal(np.sort(ids), np.arange(50000))
                assert np.array_equal(np.bincount(ids // 50), np.full(1000, 50))
                rank_records = [json.loads((out / f"rank{rank}.json").read_text())
                                for rank in range(result["world_size"])]
                assert sum(r["num_images"] for r in rank_records) == 50000
                stream = tuple(r["image_bytes_sha256"] for r in rank_records)
                assert stream not in streams
                streams.add(stream)
                with np.load(out / "official_generated_stats.npz") as moments:
                    assert moments["mu"].shape == (2048,)
                    assert moments["sigma"].shape == (2048, 2048)
                    assert np.isfinite(moments["mu"]).all()
                    assert np.isfinite(moments["sigma"]).all()
        row = summary[key]
        published = paper_by_key[key]
        assert published["checkpoint"] == checkpoint
        assert int(row["completed"]) == 3 and int(published["num_images_per_seed"]) == 50000
        for metric in ("fid", "fdr5"):
            mean = statistics.mean(v[metric] for v in values)
            sd = statistics.stdev(v[metric] for v in values)
            assert math.isclose(mean, float(row[f"{metric}_mean"]), rel_tol=0, abs_tol=1e-12)
            assert math.isclose(sd, float(row[f"{metric}_sd"]), rel_tol=0, abs_tol=1e-12)
            assert math.isclose(mean, float(published[f"{metric}_mean"]), rel_tol=0, abs_tol=1e-12)
            assert math.isclose(sd, float(published[f"{metric}_sd"]), rel_tol=0, abs_tol=1e-12)
        checked.append(key)
    return {"models": len(checked), "replicates": len(artifacts),
            "sampling_images": len(artifacts) * 50000,
            "scratch_checked": bool(scratch_root), "status": "pass"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path,
                        help="Retained rank artifacts for IDs, byte streams, and moments")
    options = parser.parse_args()
    print(json.dumps(verify(options.scratch_root), indent=2))


if __name__ == "__main__":
    main()
