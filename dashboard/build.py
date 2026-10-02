"""Build an offline dashboard from the frozen full-precision release CSVs."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
COLORS = {"Priming": "#315ad8", "Drifting": "#6b7280", "pMF": "#a06a35",
          "WF": "#07939a", "JiT": "#7447a8", "iMF": "#b64779"}


def family(model: str) -> str:
    return next(k for k in COLORS if model.startswith(k))


def chart(rows: list[dict], metric: str, label: str) -> str:
    width, height, left, right, top, bottom = 650, 350, 70, 22, 28, 58
    vals = [float(r[f"{metric}_mean"]) for r in rows]
    ymin, ymax = min(vals) * .83, max(vals) * 1.08
    xmin, xmax = 70, 4000
    x = lambda value: left + (math.log(value) - math.log(xmin)) / (math.log(xmax) - math.log(xmin)) * (width-left-right)
    y = lambda value: top + (ymax-value) / (ymax-ymin) * (height-top-bottom)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)} versus training compute">']
    for tick in (100, 300, 1000, 3000):
        px=x(tick)
        svg.append(f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{height-bottom}"/>')
        svg.append(f'<text class="tick" x="{px:.1f}" y="{height-bottom+20}" text-anchor="middle">{tick:,}</text>')
    for i in range(5):
        value=ymin+(ymax-ymin)*i/4
        py=y(value)
        svg.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{py:.1f}" y2="{py:.1f}"/>')
        svg.append(f'<text class="tick" x="{left-10}" y="{py+4:.1f}" text-anchor="end">{value:.1f}</text>')
    for row in rows:
        px, py = x(float(row["eflops"])), y(float(row[f"{metric}_mean"]))
        model = html.escape(row["model"])
        color=COLORS[family(row["model"])]
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"><title>{model}: {float(row[f"{metric}_mean"]):.4f} ± {float(row[f"{metric}_sd"]):.4f}; {float(row["eflops"]):.2f} EFLOPs</title></circle>')
    svg.append(f'<text class="axis" x="{width/2}" y="{height-8}" text-anchor="middle">Training EFLOPs · log scale</text>')
    svg.append('</svg>')
    return "".join(svg)


def main() -> None:
    fresh_csv = EVAL / "full_reproduction_20260930.csv"
    fresh = fresh_csv.exists()
    rows=list(csv.DictReader((fresh_csv if fresh else EVAL / "paper_results.csv").open()))
    seeds=list(csv.DictReader(((EVAL / "full_reproduction_20260930_per_seed.csv")
                               if fresh else (EVAL / "reports" / "per_seed.csv")).open()))
    paper={row["model"]: row for row in csv.DictReader((EVAL / "paper_results.csv").open())}
    keys=("wflow_xl", "drifting_b", "drifting_l", "pmf_l", "pmf_h", "jit_h",
          "imf_xl", "priming_b_80", "priming_b_160", "priming_l_80", "priming_l_160")
    assert len(rows)==len(keys)==11 and len(seeds)==33
    table=[]
    for key,row in zip(keys,rows,strict=True):
        color=COLORS[family(row["model"])]
        sample=" · ".join(f's{r["seed"]}: {float(r["fid"]):.4f} / {float(r["fdr5"]):.4f}'
                          for r in seeds if r["model"]==key)
        prior=paper[row["model"]]
        delta=(f'<td>{float(row["fid_mean"])-float(prior["fid_mean"]):+.4f} / '
               f'{float(row["fdr5_mean"])-float(prior["fdr5_mean"]):+.4f}</td>') if fresh else ''
        table.append(f'<tr><td><span class="dot" style="background:{color}"></span>{html.escape(row["model"])}<small>{html.escape(row["space"])} · {row["nfe"]} NFE</small></td><td>{float(row["fid_mean"]):.4f} <span class="sd">± {float(row["fid_sd"]):.4f}</span></td><td>{float(row["fdr5_mean"]):.4f} <span class="sd">± {float(row["fdr5_sd"]):.4f}</span></td>{delta}<td>{float(row["eflops"]):,.2f}</td><td><details><summary>View</summary><small>{html.escape(sample)}</small></details></td></tr>')
    legend=''.join(f'<span><i style="background:{color}"></i>{name}</span>' for name,color in COLORS.items())
    previews=''.join(f'<a class="preview" href="previews/{key}.png"><img loading="lazy" src="previews/{key}.png" alt="Fixed sample overview for {label}"><strong>{label}</strong></a>'
                     for key,label in (("priming_b_80","Priming-B-80"),
                                       ("priming_b_160","Priming-B-160"),
                                       ("priming_l_80","Priming-L-80"),
                                       ("priming_l_160","Priming-L-160")))
    batch_label="September 30, 2026 fresh reproduction" if fresh else "September 25–26, 2026 paper batch"
    best=min(rows,key=lambda row: float(row["fid_mean"]))
    delta_header='<th>Δ vs paper<br><small>FID / FDr⁵</small></th>' if fresh else ''
    result_link='full_reproduction_20260930.csv' if fresh else 'paper_results.csv'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Priming Models · ImageNet evaluation</title>
<style>body{{margin:0;background:#f4f6fa;color:#17233b;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:40px 24px 80px}}h1{{font-size:32px;letter-spacing:-.025em;margin:0 0 6px}}h2{{font-size:20px;margin:0 0 18px}}p{{max-width:1000px}}a{{color:#315ad8}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.1em;color:#315ad8;text-transform:uppercase}}.intro{{color:#52617a;margin:0 0 28px}}.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:28px}}.card,.panel{{background:white;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 2px 12px #182d5410}}.card{{padding:16px 22px;min-width:150px}}.card strong{{display:block;font-size:23px}}.card small{{color:#52617a}}.plots{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.panel{{padding:22px;margin-bottom:18px}}svg{{width:100%;height:auto}}.grid{{stroke:#e5eaf2;stroke-width:1}}.tick{{fill:#6a7890;font-size:12px}}.axis{{fill:#53637e;font-size:13px}}.legend{{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 0}}.legend span{{white-space:nowrap;color:#52617a;font-size:13px}}.legend i,.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}.previews{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.preview{{display:block;text-decoration:none;color:#17233b;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}.preview img{{width:100%;display:block}}.preview strong{{display:block;padding:9px 12px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:790px}}th,td{{padding:13px 9px;border-bottom:1px solid #ebeff5;text-align:left}}th{{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#53637e}}td small{{display:block;color:#6a7890;margin-left:16px}}.sd{{color:#6a7890}}summary{{cursor:pointer;color:#315ad8}}details small{{min-width:200px;margin:5px 0}}.note{{color:#53637e;font-size:13px}}@media(max-width:900px){{.plots,.previews{{grid-template-columns:1fr}}main{{padding:24px 14px}}}}</style>
<main><div class="eyebrow">ImageNet-256 · final paper evaluation</div><h1>Priming Models and comparison baselines</h1>
<p class="intro">Fixed checkpoints, 50,000 class-balanced images per sampling seed, seeds 1–3. Values show mean ± sample SD across separate evaluations. Displayed batch: {batch_label}.</p>
<div class="cards"><div class="card"><strong>11 / 11</strong><small>models complete</small></div><div class="card"><strong>33 / 33</strong><small>sampling replicates validated</small></div><div class="card"><strong>1.65M</strong><small>generated images</small></div><div class="card"><strong>{float(best['fid_mean']):.4f}</strong><small>best FID · {html.escape(best['model'])}</small></div></div>
<div class="plots"><section class="panel"><h2>FID ↓</h2>{chart(rows,"fid","FID")}</section><section class="panel"><h2>FDr⁵ ↓</h2>{chart(rows,"fdr5","FDr5")}</section></div>
<div class="legend">{legend}</div><section class="panel" style="margin-top:22px"><h2>Production model samples</h2><p class="note">Fixed 25-class sample overviews from the four terminal EMA checkpoints. These are qualitative panels, separate from the 50,000-image metric draws.</p><div class="previews">{previews}</div></section><section class="panel"><h2>Results</h2><div class="scroll"><table><thead><tr><th>Model</th><th>FID ↓</th><th>FDr⁵ ↓</th>{delta_header}<th>Training EFLOPs</th><th>Seed scores<br><small>FID / FDr⁵</small></th></tr></thead><tbody>{''.join(table)}</tbody></table></div></section>
<section class="panel"><h2>Metric contract</h2><p>FID: rounded uint8 images, pinned LTH14 torch-fidelity Inception features in FP32 with TF32 disabled, and the JiT ImageNet-256 real reference. FDr⁵: a separate frozen evaluator on float images, using normalized Inception (ADM), DINOv2, ViT-MAE, SigLIP, and CLIP distances; ConvNeXt is excluded.</p><p class="note">The three seeds resample one checkpoint per model; they are not training replicates. JiT-H uses 198 actual guided network calls (200 by paper convention). Training compute excludes evaluation. This table supersedes older dashboard headline values.</p><p><a href="../eval/{result_link}">Displayed full-precision CSV</a> · <a href="../eval/paper_results.csv">Paper batch CSV</a> · <a href="../eval/FID_SETUP_AND_RESULTS.md">Protocol and comparison notes</a> · <a href="../REPRODUCTION_STATUS.md">Reproduction status</a></p></section></main></html>'''
    (HERE / "index.html").write_text(page)
    print(f"Built {HERE / 'index.html'} from {len(rows)} models and {len(seeds)} replicates")


if __name__ == "__main__":
    main()
