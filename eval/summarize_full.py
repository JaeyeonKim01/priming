"""Validate a fresh 33-run batch and compare its metrics with the paper batch."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import statistics
from pathlib import Path

from verify_batch import verify_one


HERE = Path(__file__).resolve().parent
KEYS = ("wflow_xl", "drifting_b", "drifting_l", "pmf_l", "pmf_h", "jit_h",
        "imf_xl", "priming_b_80", "priming_b_160", "priming_l_80", "priming_l_160")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Scratch root of the fresh batch")
    parser.add_argument("--tag", default="20260930")
    parser.add_argument("--paper-scratch", type=Path,
                        default=Path("/n/netscratch/sham_lab/Lab/jaykim/three_seed_eval_20260925"),
                        help="Retained paper-batch artifacts for rank-stream comparison")
    options = parser.parse_args()
    paper_rows = list(csv.DictReader((HERE / "paper_results.csv").open()))
    assert len(paper_rows) == len(KEYS) == 11

    summaries, new_rows, seeds = {}, [], []
    for key, paper in zip(KEYS, paper_rows, strict=True):
        check = verify_one(options.output_dir, key, 50000, [1, 2, 3])
        reports = [json.loads((options.output_dir / key / f"n50000_seed{seed}" /
                               "results.json").read_text()) for seed in (1, 2, 3)]
        assert len({r["script_sha256"] for r in reports}) == 1
        assert len({r["checkpoint"] for r in reports}) == 1
        paper_stream_matches = []
        for report in reports:
            seed = report["seed"]
            for rank in range(report["world_size"]):
                path = Path(key) / f"n50000_seed{seed}" / f"rank{rank}.json"
                paper_rank = options.paper_scratch / path
                fresh_rank = options.output_dir / path
                if paper_rank.exists():
                    original = json.loads(paper_rank.read_text())["image_bytes_sha256"]
                    reproduced = json.loads(fresh_rank.read_text())["image_bytes_sha256"]
                    paper_stream_matches.append(original == reproduced)
        updated = paper.copy()
        updated["checkpoint"] = reports[0]["checkpoint"]
        deltas = {}
        for metric in ("fid", "fdr5"):
            values = [float(r[metric]) for r in reports]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            assert abs(mean - check[f"{metric}_mean"]) < 1e-12
            assert abs(sd - check[f"{metric}_sd"]) < 1e-12
            updated[f"{metric}_mean"] = repr(mean)
            updated[f"{metric}_sd"] = repr(sd)
            deltas[metric] = {"paper_mean": float(paper[f"{metric}_mean"]),
                              "fresh_mean": mean,
                              "fresh_minus_paper": mean - float(paper[f"{metric}_mean"]),
                              "paper_table_value": f"{float(paper[f'{metric}_mean']):.2f}",
                              "fresh_table_value": f"{mean:.2f}",
                              "two_decimal_table_match": f"{mean:.2f}" == f"{float(paper[f'{metric}_mean']):.2f}",
                              "paper_sd": float(paper[f"{metric}_sd"]),
                              "fresh_sd": sd,
                              "per_seed": values}
        updated["metric_source"] = str(options.output_dir.resolve())
        new_rows.append(updated)
        for report in reports:
            seeds.append({"model": key, "seed": report["seed"], "fid": report["fid"],
                          "fdr5": report["fdr5"], "checkpoint": report["checkpoint"],
                          "script_sha256": report["script_sha256"]})
        summaries[key] = {"label": paper["model"], "artifact_validation": check,
                          "rank_streams_matching_paper": sum(paper_stream_matches),
                          "rank_streams_compared": len(paper_stream_matches),
                          "comparison": deltas}

    tag = options.tag
    with (HERE / f"full_reproduction_{tag}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paper_rows[0].keys())
        writer.writeheader()
        writer.writerows(new_rows)
    with (HERE / f"full_reproduction_{tag}_per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=seeds[0].keys())
        writer.writeheader()
        writer.writerows(seeds)
    result = {"status": "pass", "completed_at_utc": datetime.datetime.now(
        datetime.timezone.utc).isoformat(), "scratch_root": str(options.output_dir.resolve()),
        "count_models": len(summaries), "count_replicates": len(seeds),
        "images_per_replicate": 50000,
        "comparison_note": "Fresh 50,000-image sampling, FID and FDr5 for each of three seeds; differences are fresh minus September 25–26 paper batch",
        "models": summaries}
    (HERE / f"full_reproduction_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    same_table = all(entry["comparison"][metric]["two_decimal_table_match"]
                     for entry in summaries.values() for metric in ("fid", "fdr5"))
    lines = [f"# Fresh FID / FDr⁵ reproduction — {tag}", "",
             "All 11 checkpoints × three independent sampling seeds × 50,000 class-balanced ImageNet-256 images completed and passed artifact validation (33 runs; 1,650,000 images). FID and FDr⁵ were computed as separate metrics under the final paper protocol.", "",
             f"The fresh and paper means {'match' if same_table else 'do not all match'} at the manuscript's two-decimal precision. Δ is fresh minus the September 25–26 paper batch.", "",
             "| Model | Fresh FID mean ± SD | Δ FID | Fresh FDr⁵ mean ± SD | Δ FDr⁵ | Matching rank streams |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in KEYS:
        entry = summaries[key]
        fid, fdr5 = (entry["comparison"][metric] for metric in ("fid", "fdr5"))
        lines.append(f"| {entry['label']} | {fid['fresh_mean']:.4f} ± {fid['fresh_sd']:.4f} | "
                     f"{fid['fresh_minus_paper']:+.6f} | "
                     f"{fdr5['fresh_mean']:.4f} ± {fdr5['fresh_sd']:.4f} | "
                     f"{fdr5['fresh_minus_paper']:+.6f} | "
                     f"{entry['rank_streams_matching_paper']}/{entry['rank_streams_compared']} |")
    lines += ["", "Byte-stream counts compare each GPU rank and seed with the retained paper batch; differing streams can still yield close aggregate metrics. FDr⁵ uses the five frozen normalized feature distances and excludes ConvNeXt. The three seeds are sampling replicates of one checkpoint per model, not training replicates.", "",
              f"Full-precision results: [CSV](full_reproduction_{tag}.csv), [per-seed CSV](full_reproduction_{tag}_per_seed.csv), and [validation/comparison JSON](full_reproduction_{tag}.json). Raw artifacts: `{options.output_dir.resolve()}`.", ""]
    (HERE / f"full_reproduction_{tag}.md").write_text("\n".join(lines))
    print(f"Validated {len(seeds)} fresh runs across {len(summaries)} models")
    for key, row in summaries.items():
        comp = row["comparison"]
        print(f"{row['label']:<18} FID {comp['fid']['fresh_mean']:.4f} ± {row['artifact_validation']['fid_sd']:.4f} "
              f"(Δ {comp['fid']['fresh_minus_paper']:+.4f}); "
              f"FDr5 {comp['fdr5']['fresh_mean']:.4f} ± {row['artifact_validation']['fdr5_sd']:.4f} "
              f"(Δ {comp['fdr5']['fresh_minus_paper']:+.4f})")


if __name__ == "__main__":
    main()
