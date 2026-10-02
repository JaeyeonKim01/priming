"""Strict, CPU-only readers for the public run dashboards."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DENOMINATORS = {
    "FID(ADM)": 1.68, "convnext": 56.87, "dinov2_cls": 14.19,
    "mae_cls": 0.04, "siglip_cls": 0.60, "clip_cls": 5.60,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content.encode() if isinstance(content, str) else content)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_fd(fd, excluded):
    if excluded not in DENOMINATORS:
        raise ValueError(f"Unknown excluded encoder: {excluded}")
    names = [name for name in DENOMINATORS if name != excluded]
    if any(name not in fd for name in names):
        return None
    values = [float(fd[name]) / DENOMINATORS[name] for name in names]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Nonfinite or negative FD")
    return sum(values) / len(values)


def evaluation_groups(path, *, weights="0.999", cfg=1.0):
    """Never combine different draws, protocols, weights or partial evaluations."""
    groups = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["ema_label"] != weights or float(row["cfg"]) != cfg:
                continue
            if int(row["n"]) != 50000 or int(row["num_sampling_steps"]) != 1:
                continue
            key = (int(row["step"]), row["timestamp"], row["interval_min"], row["interval_max"])
            groups.setdefault(key, []).append(row)
    result = []
    for (step, timestamp, _, _), rows in groups.items():
        if len(rows) != 6 or {row["model"] for row in rows} != set(DENOMINATORS):
            continue
        fd = {row["model"]: float(row["fd"]) for row in rows}
        if not all(math.isfinite(v) and v >= 0 for v in fd.values()):
            continue
        fdr6 = sum(fd[name] / denom for name, denom in DENOMINATORS.items()) / 6
        if any(not math.isclose(float(row["fdr6"]), fdr6, rel_tol=1e-6) for row in rows):
            raise ValueError(f"FDr6 does not match component FDs: {path}, step {step}")
        inc = next(row for row in rows if row["model"] == "FID(ADM)")
        result.append(dict(step=step, timestamp=timestamp, weights=weights, cfg=cfg,
            n=50000, nfe=1, fd=fd, fid=fd["FID(ADM)"], fdr6=fdr6,
            fdr5_no_inception=aggregate_fd(fd, "FID(ADM)"),
            fdr5_no_convnext=aggregate_fd(fd, "convnext"),
            precision=float(inc["precision"]), recall=float(inc["recall"]),
            source=str(path), source_rows=rows))
    return sorted(result, key=lambda row: (row["step"], row["timestamp"]))


def terminal_metrics(path, step, *, weights="0.999", cfg=1.0):
    rows = [row for row in evaluation_groups(path, weights=weights, cfg=cfg) if row["step"] == step]
    if not rows:
        raise ValueError(f"No complete 50k evaluation at step {step}, EMA {weights}: {path}")
    return rows[-1]


def panel_contract(path=None):
    contract = read_json(path or ROOT / "configs/dashboard_panel.json")
    ids = [row["id"] for row in contract["classes"]]
    if len(ids) != 25 or len(set(ids)) != 25 or any(not 0 <= i < 1000 for i in ids):
        raise ValueError("Dashboard requires 25 distinct ImageNet class IDs")
    if contract["samples_per_class"] != 8:
        raise ValueError("Dashboard requires eight inputs per class")
    return contract


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
