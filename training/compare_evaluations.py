"""Compare complete evaluation draws against the original production runs."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parent
sys.path[:0] = [str(TASK / 'runtime'), str(TASK / 'scripts')]
from audit_evaluations import audit_series
from paths import load_config
from src.eval.dashboard_data import evaluation_groups


def metrics(draw):
    result = {k: draw[k] for k in ('fid', 'fdr5_no_convnext', 'fdr6', 'precision', 'recall')}
    result.update({'fd_' + k: v for k, v in draw['fd'].items()})
    inception = next(r for r in draw['source_rows'] if r['model'] == 'FID(ADM)')
    result['inception_score'] = float(inception['is'])
    assert all(math.isfinite(v) for v in result.values())
    return result


def protocol(draw):
    r = draw['source_rows'][0]
    return (draw['cfg'], draw['n'], draw['nfe'], float(r['interval_min']), float(r['interval_max']))


def main():
    reports = TASK / 'reports'
    snapshots = reports / 'comparison_inputs'
    snapshots.mkdir(exist_ok=True)
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                  protocol='Historical BF16 online evaluation; 50k images, CFG 1, NFE 1. '
                           'FDr5 excludes ConvNeXt. These are not final release-protocol FIDs.',
                  delta_definition='new minus historical; negative favors new for FID/FD/FDr',
                  arms={}, rows=[])
    for arm in ('b160', 'l160'):
        sources, series = {}, {}
        for label, relative in [('historical', f'provenance/{arm}_production.json'),
                                ('new', f'configs/{arm}.json')]:
            cfg = load_config(arm)['logging']
            source = snapshots / f'{arm}_historical.csv' if label == 'historical' else Path(cfg['output_dir']) / cfg['project'] / cfg['exp_name'] / 'final_eval_summary.csv'
            raw = source.read_bytes()
            snapshot = snapshots / f'{arm}_{label}.csv'
            snapshot.write_bytes(raw)
            sources[label] = dict(path=str(source), snapshot=str(snapshot),
                                  sha256=hashlib.sha256(raw).hexdigest())
            draws = [r for w in ('online', '0.999') for r in evaluation_groups(snapshot, weights=w)]
            # Latest complete draw at each step/weight, never the best metric.
            series[label] = {(r['step'], r['weights']): r for r in draws}
        through = min(max(s for s, w in series['new'] if w == weight)
                      for weight in ('online', '0.999'))
        audits = {label: audit_series(Path(info['snapshot']), through) for label, info in sources.items()}
        result['arms'][arm] = dict(through=through, sources=sources, audits=audits)
        for (step, weight), new in sorted(series['new'].items()):
            if step > through:
                continue
            old = series['historical'][(step, weight)]
            assert protocol(new) == protocol(old), (arm, step, weight)
            row = dict(arm=arm, step=step, weights=weight, n=new['n'],
                       new_timestamp=new['timestamp'], historical_timestamp=old['timestamp'])
            for name, value in metrics(new).items():
                historical = metrics(old)[name]
                row.update({name + '_new': value, name + '_historical': historical,
                            name + '_delta': value - historical,
                            name + '_delta_pct': 100 * (value / historical - 1) if historical else None})
            result['rows'].append(row)
    (reports / 'evaluation_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (reports / 'evaluation_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = {'online': '#0072B2', '0.999': '#D55E00'}
    for col, arm in enumerate(('b160', 'l160')):
        for row_index, (metric, label) in enumerate([('fid', 'FID'), ('fdr5_no_convnext', 'FDr5 (excludes ConvNeXt)')]):
            ax = axes[row_index, col]
            for weight in ('online', '0.999'):
                rows = sorted((r for r in result['rows'] if r['arm'] == arm and r['weights'] == weight), key=lambda r: r['step'])
                for source, style, marker in [('historical', '--', 'o'), ('new', '-', 'x')]:
                    ax.plot([r['step'] / 1000 for r in rows], [r[f'{metric}_{source}'] for r in rows],
                            linestyle=style, marker=marker, markersize=5, color=colors[weight],
                            label=f"{'EMA' if weight == '0.999' else 'Online'} · {source}")
            ax.set_title(f"OST-{'B' if arm == 'b160' else 'L'} · {label}")
            ax.set_xlabel('OT updates (thousands)')
            ax.set_ylabel(label + ' ↓ (log scale)')
            ax.set_yscale('log')
            ax.grid(alpha=.2, which='both')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Migrated runs vs original production · 50k online evaluation protocol')
    fig.savefig(reports / 'evaluation_comparison.png', dpi=180)
    fig.savefig(reports / 'evaluation_comparison.pdf')
    plt.close(fig)
    print(json.dumps({arm: {'through': info['through'], 'paired_draws': sum(r['arm'] == arm for r in result['rows'])}
                      for arm, info in result['arms'].items()}))
    for arm in result['arms']:
        print(arm, 'step | weights | FID historical -> new | FDr5 historical -> new')
        for r in result['rows']:
            if r['arm'] == arm:
                print(f"{r['step']:5d} | {r['weights']:6s} | {r['fid_historical']:.4f} -> {r['fid_new']:.4f} | "
                      f"{r['fdr5_no_convnext_historical']:.4f} -> {r['fdr5_no_convnext_new']:.4f}")


if __name__ == '__main__':
    main()
