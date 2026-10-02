"""Recompute official FID from each saved generated-moment file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "isolated_experiments/fid_protocol_audit_20260925/vendor/torch-fidelity"))
from torch_fidelity.metric_fid import fid_statistics_to_metric  # noqa: E402

REFERENCE = Path("/n/home10/jaykim/woobin/JiT/fid_stats/jit_in256_stats.npz")
REFERENCE_SHA256 = "412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="Limit the audit to one model")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--report", type=Path, help="Optional JSON audit output")
    options = parser.parse_args()
    assert digest(options.reference) == REFERENCE_SHA256
    with np.load(options.reference) as data:
        reference = {key: data[key] for key in ("mu", "sigma")}
    reports = sorted(options.output_dir.glob("*/n50000_seed*/results.json"))
    if options.model:
        reports = [p for p in reports if p.parent.parent.name == options.model]
    assert reports, "No completed 50,000-image runs found"
    checked = {}
    for report_path in reports:
        report = json.loads(report_path.read_text())
        with np.load(report_path.parent / "official_generated_stats.npz") as saved:
            generated = {key: saved[key].astype(reference[key].dtype)
                         for key in ("mu", "sigma")}
        value = float(fid_statistics_to_metric(generated, reference, verbose=False)
                      ["frechet_inception_distance"])
        delta = value - float(report["fid"])
        assert abs(delta) < 1e-6, (report_path, delta)
        name = str(report_path.parent.relative_to(options.output_dir))
        checked[name] = {"reported_fid": report["fid"], "recomputed_fid": value,
                         "difference": delta, "status": "pass"}
        print(name, value, delta, flush=True)
    summary = {"status": "pass", "reference_sha256": REFERENCE_SHA256,
               "batch": str(options.output_dir.resolve()), "runs_checked": len(checked),
               "runs": checked}
    if options.report:
        options.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Recomputed saved-moment FID for {len(checked)} runs")


if __name__ == "__main__":
    main()
