"""Publish verified metrics and fixed samples; no server or network required."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import html
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.dashboard_data import atomic_write, read_json, terminal_metrics, write_json


def register_run(run_dir, step, *, figures=None, pretrain_eflops=None,
                 step_tflops=None, label=None, compute_source=None):
    figures = Path(figures or ROOT / "figures")
    run_dir = Path(run_dir)
    metrics = terminal_metrics(run_dir / "final_eval_summary.csv", int(step))
    identifier = run_dir.name
    if not identifier or identifier in {".", ".."}:
        raise ValueError("A run directory with a stable name is required")
    post = None if step_tflops is None else float(step_tflops) * step / 1e6
    total = None if post is None or pretrain_eflops is None else float(pretrain_eflops) + post
    record = dict(metrics, id=identifier, model=label or identifier, family="OST",
        space="pixel", eflops=total, pretrain_eflops=pretrain_eflops,
        posttrain_eflops=post, compute_source=compute_source,
        metric_source="Local frozen evaluator · 50k terminal EMA",
        completed_at=datetime.now(timezone.utc).isoformat())
    write_json(figures / "benchmarks/runs" / f"{identifier}.json", record)
    write_json(figures / "benchmarks/latest.json", {"run": identifier})
    return record


def _metric(row, excluded):
    return row.get("fdr5_no_inception" if excluded == "FID(ADM)" else "fdr5_no_convnext")


def render_benchmark(figures, *, excluded=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ost_dashboard_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    figures = Path(figures)
    output = figures / "benchmarks"
    if excluded is None:
        settings = read_json(ROOT / "configs/dashboard_metrics.json")
        excluded = settings["fdr5_excluded_encoder"]
    if excluded not in {"FID(ADM)", "convnext"}:
        raise ValueError("Dashboard FDr5 definition must be explicit")
    excluded_label = "Inception" if excluded == "FID(ADM)" else "ConvNeXt"
    rows = [read_json(path) for path in sorted((output / "runs").glob("*.json"))]
    rows += read_json(output / "baselines.json")
    latest_id = read_json(output / "latest.json")["run"]
    latest = next(row for row in rows if row["id"] == latest_id)
    rows.sort(key=lambda row: (row["family"] != "OST", row.get("eflops") or float("inf")))
    for row in rows:
        row["fdr5"] = _metric(row, excluded)
        row["fdr5_excluded_encoder"] = excluded
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    styles = {"OST": ("#26745a", "*"), "pMF": ("#777777", "o"),
        "Drifting pixel": ("#555555", "s"), "Drifting latent": ("#999999", "D"),
        "JiT": ("#aaaaaa", "^")}
    for ax, metric, title in zip(axes, ["fid", "fdr5"], ["FID ↓", f"FDr5 · without {excluded_label} ↓"]):
        for family, (color, marker) in styles.items():
            available = sorted([row for row in rows if row["family"] == family
                and row.get(metric) is not None and row.get("eflops") is not None], key=lambda row: row["eflops"])
            if family != "OST" and len(available) > 1:
                ax.plot([r["eflops"] for r in available], [r[metric] for r in available], color=color, alpha=.3, lw=1)
            for row in available:
                current = row["id"] == latest_id
                ax.scatter(row["eflops"], row[metric], s=230 if current else 65,
                    color=color, marker=marker, zorder=5, alpha=1 if current or family != "OST" else .55)
                label = row.get("short_label", row["model"]) if current or family == "OST" else row.get("size", row["model"])
                ax.annotate(label, (row["eflops"], row[metric]), xytext=(7, 8),
                    textcoords="offset points", color=color, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(40, 3000)
        ax.set_xticks([50, 100, 300, 1000, 2000])
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        ax.set_xlabel("Total generator-training compute (EFLOPs, log scale)")
        ax.set_title(title, loc="left", fontweight="bold", pad=15)
        ax.grid(True, color="#e7e9e6", zorder=0)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=color, marker=marker, lw=1, markersize=8, label=family)
        for family, (color, marker) in styles.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.text(.06, .015, "ImageNet-256 · 50k metrics · OST includes reconstruction pretraining · recipe-level comparison; training encoders and counter coverage differ", fontsize=9, color="#667069")
    fig.tight_layout(rect=(0, .05, 1, .91))
    for suffix in ("png", "svg"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=suffix, dpi=180, facecolor="white")
        atomic_write(output / f"quality_compute.{suffix}", buffer.getvalue())
    plt.close(fig)
    columns = ["model", "space", "nfe", "eflops", "fid", "fdr5", "fdr5_excluded_encoder", "precision", "recall", "metric_source", "source", "compute_source"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(output / "comparison.csv", buffer.getvalue())
    write_json(output / "comparison.json", {"fdr5_excluded_encoder": excluded, "models": rows})
    def number(value, digits=4):
        return "—" if value is None else f"{value:,.{digits}f}"
    table = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
        [row["model"], row["space"], row.get("nfe", 1), number(row.get("eflops"), 3),
         number(row.get("fid")), number(row.get("fdr5")), row["metric_source"]]) + "</tr>" for row in rows)
    svg = (output / "quality_compute.svg").read_text()
    svg = svg[svg.index("<svg"):]
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OST · training compute and quality</title><style>body{{font:15px system-ui,sans-serif;background:#f4f5f3;color:#25372e;margin:0}}main{{max-width:1400px;margin:30px auto;padding:30px;background:white;border-radius:12px}}h1{{font-size:30px}}p{{line-height:1.6}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:12px;border-bottom:1px solid #e3e7e3;text-align:left}}a{{color:#26745a}}.scroll{{overflow:auto}}.note{{color:#66746b}}@media(max-width:800px){{main{{margin:0;padding:16px}}}}</style><main>
<h1>Training compute versus image quality</h1><p><strong>{html.escape(latest['model'])}</strong> · {latest['step']:,} OT updates · terminal EMA 0.999 · FID <strong>{latest['fid']:.4f}</strong> · FDr5 <strong>{number(_metric(latest, excluded))}</strong> · {number(latest.get('eflops'), 3)} total EFLOPs.</p>
{svg}<p><a href="quality_compute.png">Open metric PNG</a> · <a href="comparison.csv">CSV</a> · <a href="comparison.json">Full metrics and provenance</a> · <a href="../samples/index.html">25-class sample dashboard</a></p>
<p class="note">FDr5 excludes <strong>{excluded_label}</strong> consistently. Missing components are shown as unavailable, never replaced by another five-encoder average. PMF, Drifting and JiT are gray; OST is green. JiT uses its reported 200-NFE guided recipe; the one-step models use 1 NFE. Lines connect family members for readability.</p>
<div class="scroll"><table><thead><tr><th>Model</th><th>Space</th><th>NFE</th><th>EFLOPs</th><th>FID ↓</th><th>FDr5 ↓</th><th>Metric source</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="note">Compute includes the successful generator-training lineage, frozen training encoder forward/backward work, and online reference encoding where used. It excludes foundation encoder/VAE pretraining, offline preparation, optimizer/EMA, communication, evaluation and search. PyTorch and XLA coverage differ; unsupported Sinkhorn operations are omitted. This is a descriptive recipe comparison, not an equal-compute objective ablation. Evaluation-seed repeats are not independent training seeds.</p></main></html>'''
    atomic_write(output / "index.html", page)
    return rows


def refresh(figures=None):
    from src.eval.sample_dashboard import build_sample_dashboard
    figures = Path(figures or ROOT / "figures")
    figures.mkdir(parents=True, exist_ok=True)
    with (figures / ".dashboards.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = render_benchmark(figures)
        latest = read_json(figures / "benchmarks/latest.json")["run"]
        current = figures / "samples" / latest / "manifest.json"
        baselines = [figures / "samples" / name / "manifest.json" for name in ("drifting_b", "drifting_l")]
        if current.is_file():
            build_sample_dashboard(figures / "samples", [current, *[p for p in baselines if p.is_file()]])
        for row in rows:
            if row["family"] != "OST":
                continue
            sample = figures / "samples" / row["id"] / "manifest.json"
            if sample.is_file():
                build_sample_dashboard(sample.parent, [sample, *[p for p in baselines if p.is_file()]])
            run_output = figures / "runs" / row["id"]
            write_json(run_output / "metrics.json", row)
            sample_link = f'../../samples/{row["id"]}/index.html'
            value = "unavailable" if row["fdr5"] is None else f'{row["fdr5"]:.4f}'
            page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(row["model"])}</title>
<main style="font:16px system-ui;max-width:1000px;margin:50px auto;line-height:1.7"><h1>{html.escape(row["model"])}</h1>
<p>Completed {row["step"]:,} updates · EMA {row["weights"]} · 50,000 evaluation images.</p>
<p>FID {row["fid"]:.4f} · FDr5 {value} (excluded encoder: {row["fdr5_excluded_encoder"]}).</p>
<p><a href="../../benchmarks/index.html">Compute and metric dashboard</a> · <a href="metrics.json">Exact run metrics</a> · <a href="{html.escape(sample_link)}">Fixed sample comparison</a></p></main></html>'''
            atomic_write(run_output / "index.html", page)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--label")
    parser.add_argument("--pretrain-eflops", type=float)
    parser.add_argument("--step-tflops", type=float)
    parser.add_argument("--compute-source")
    cli = parser.parse_args()
    if cli.run_dir:
        if cli.step is None:
            parser.error("--step is required to register a terminal evaluation")
        register_run(cli.run_dir, cli.step, label=cli.label, pretrain_eflops=cli.pretrain_eflops,
            step_tflops=cli.step_tflops, compute_source=cli.compute_source)
    rows = refresh()
    print(json.dumps([dict(model=r["model"], fid=r["fid"], fdr5=r["fdr5"], eflops=r["eflops"]) for r in rows], indent=2))


if __name__ == "__main__":
    main()
