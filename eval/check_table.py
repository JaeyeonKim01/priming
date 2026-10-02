"""Recompute the 11 headline rows and compare them with the manuscript table."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = {
    "wflow_xl": ("WF-XL/2", "WF-XL/2"),
    "drifting_b": ("Drifting-B/16", "Drifting-B"),
    "drifting_l": ("Drifting-L/16", "Drifting-L"),
    "pmf_l": ("pMF-L/16", "pMF-L"),
    "pmf_h": ("pMF-H/16", "pMF-H"),
    "jit_h": ("JiT-H/16", "JiT-H"),
    "imf_xl": ("iMF-XL/2", "iMF-XL/2"),
    "priming_b_80": ("Priming-B-80", "Priming-B (80 ep.)"),
    "priming_b_160": ("Priming-B-160", "Priming-B (160 ep.)"),
    "priming_l_80": ("Priming-L-80", "Priming-L (80 ep.)"),
    "priming_l_160": ("Priming-L-160", "Priming-L (160 ep.)"),
}


def manuscript_rows(path) -> dict[str, tuple[float, float, float]]:
    text = path.read_text()
    table = text.split(r"\begin{tabular}{lrrr}", 1)[1].split(r"\end{tabular}", 1)[0]
    table = re.sub(r"\\tablecolor\{[^{}]*\}", "", table)
    for command in ("textbf", "underline"):
        table = re.sub(r"\\" + command + r"\{([^{}]*)\}", r"\1", table)
    rows = {}
    for chunk in table.split(r"\\"):
        parts = [" ".join(part.split()) for part in chunk.split("&")]
        if len(parts) != 4:
            continue
        name = parts[0].replace(r"\toprule", "").replace(r"\midrule", "").strip()
        if name == "Method":
            continue
        rows[name] = tuple(float(value) for value in parts[1:])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=HERE / "provenance/manuscript_table.tex")
    options = parser.parse_args()
    paper = {r["model"]: r for r in csv.DictReader((HERE / "paper_results.csv").open())}
    table = manuscript_rows(options.manuscript)
    status = (HERE.parent / "REPRODUCTION_STATUS.md").read_text()
    assert len(paper) == len(table) == len(LABELS) == 11
    checked = {}
    for key, (csv_label, tex_label) in LABELS.items():
        values = [json.loads((HERE / "reports" / f"{key}_n50000_seed{seed}.json").read_text())
                  for seed in (1, 2, 3)]
        assert all(row["num_images"] == 50000 and row["seed"] == seed
                   for seed, row in zip((1, 2, 3), values, strict=True))
        fid = statistics.mean(row["fid"] for row in values)
        fid_sd = statistics.stdev(row["fid"] for row in values)
        fdr5 = statistics.mean(row["fdr5"] for row in values)
        fdr5_sd = statistics.stdev(row["fdr5"] for row in values)
        row = paper[csv_label]
        for value, column in ((fid, "fid_mean"), (fid_sd, "fid_sd"),
                              (fdr5, "fdr5_mean"), (fdr5_sd, "fdr5_sd")):
            assert abs(value - float(row[column])) < 1e-12, (key, column)
        tex_fid, tex_fdr5, tex_compute = table[tex_label]
        assert f"{fid:.2f}" == f"{tex_fid:.2f}", key
        assert f"{fdr5:.2f}" == f"{tex_fdr5:.2f}", key
        assert f"{float(row['eflops']):.2f}" == f"{tex_compute:.2f}", key
        expected_status = (f"| {csv_label} | 3/3 | {fid:.4f} ± {fid_sd:.4f} | "
                           f"{fdr5:.4f} ± {fdr5_sd:.4f} |")
        assert expected_status in status, key
        checked[key] = {"model": csv_label, "fid_mean": fid, "fid_sd": fid_sd,
                        "fdr5_mean": fdr5, "fdr5_sd": fdr5_sd,
                        "manuscript_fid": tex_fid, "manuscript_fdr5": tex_fdr5,
                        "status": "match"}
    result = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source": "33 individual saved 50,000-image result JSON files",
              "compared_with": ["release full-precision CSV", "final paper ImageNet LaTeX table",
                                "OST_public reproduction status table"],
              "models_checked": len(checked), "status": "pass", "rows": checked,
              "scope_note": "Recomputed summary arithmetic from saved evaluations; no new 50,000-image sampling"}
    output = HERE / "reports/table_recheck_20260930.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Matched all {len(checked)} models against CSV, paper table, and status table")


if __name__ == "__main__":
    main()
