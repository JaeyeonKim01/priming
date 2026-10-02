"""Regression: six rows split across draws must not satisfy a milestone."""
import csv
import importlib.util
from pathlib import Path
import sys

import pytest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'runtime'))
from src.eval.dashboard_data import DENOMINATORS

spec = importlib.util.spec_from_file_location('ost_l_series_audit', TASK / 'scripts/audit_evaluations.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def rows(timestamp, weights='online', step=1250):
    return [dict(model=name,fd=denom*(i+1),fdr6=3.5,n=50000,ema_label=weights,cfg=1,
        num_sampling_steps=1,step=step,timestamp=timestamp,interval_min=.1,interval_max=1,
        precision=.8,recall=.6) for i,(name,denom) in enumerate(DENOMINATORS.items())]


def write(path, data):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def test_fragments_from_different_draws_cannot_fill_one_milestone(tmp_path):
    path = tmp_path / 'evaluation.csv'
    write(path, rows('draw-a')[:3] + rows('draw-b')[3:] + rows('ema', '0.999'))
    with pytest.raises(ValueError, match='Missing complete 50k draws'):
        audit.audit_series(path, 1250)


def test_requires_each_weight_at_each_milestone(tmp_path):
    path = tmp_path / 'evaluation.csv'
    data = rows('online') + rows('ema', '0.999')
    write(path, data)
    result = audit.audit_series(path, 1250)
    assert result['complete_draws'] == 2
    assert all(draw['fdr5_no_convnext'] == 3.8 for draw in result['draws'])
    with pytest.raises(ValueError, match='2500'):
        audit.audit_series(path, 2500)


def test_invalid_precision_fails_even_when_fd_components_are_complete(tmp_path):
    path = tmp_path / 'evaluation.csv'
    data = rows('online') + rows('ema', '0.999')
    data[0]['precision'] = float('nan')
    write(path, data)
    with pytest.raises(ValueError, match='precision/recall'):
        audit.audit_series(path, 1250)
