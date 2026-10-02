"""Require a complete, finite 50k draw at every production evaluation milestone.

Unlike grouping only by step and EMA label, this retains the evaluator's draw
timestamp and interval. Fragments from different draws cannot fill a milestone.
"""
import argparse
import json
import math
from pathlib import Path
import sys


def audit_series(path, through=25000, *, weights=('online', '0.999')):
    from src.eval.dashboard_data import evaluation_groups
    if through < 1250 or through % 1250:
        raise ValueError('through must be a positive 1,250-update milestone')
    required = list(range(1250, through+1, 1250))
    selected = []
    for label in weights:
        draws = evaluation_groups(path, weights=label)
        by_step = {}
        for draw in draws:
            if draw['step'] in required:
                if not all(math.isfinite(draw[name]) and 0 <= draw[name] <= 1
                           for name in ('precision', 'recall')):
                    raise ValueError(f'Invalid precision/recall: {path}, {draw["step"]}, {label}')
                by_step.setdefault(draw['step'], []).append(draw)
        missing = sorted(set(required) - set(by_step))
        if missing:
            raise ValueError(f'Missing complete 50k draws: {path}, weights={label}, steps={missing}')
        for step in required:
            # Repeated complete evaluations are retained in the raw CSV.
            # This audit selects the latest complete draw, never the best FID.
            draw = by_step[step][-1]
            selected.append({key: draw[key] for key in (
                'step', 'timestamp', 'weights', 'n', 'cfg', 'nfe', 'fid',
                'fdr5_no_convnext', 'fdr6', 'precision', 'recall', 'source')})
    return dict(passed=True, through=through, required_milestones=required,
        weights=list(weights), complete_draws=len(selected), draws=selected,
        fdr5_excluded_encoder='convnext', grouping='step, weight, timestamp, CFG, interval')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--through', type=int, default=25000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    task = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(task / 'runtime'))
    from src.eval.dashboard_data import write_json
    result = audit_series(Path(args.csv), args.through)
    write_json(args.output, result)
    print(json.dumps({key:value for key,value in result.items() if key != 'draws'}), flush=True)


if __name__ == '__main__':
    main()
