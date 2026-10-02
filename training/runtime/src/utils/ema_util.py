"""
EMA (Exponential Moving Average) utilities for model weight averaging.

two schedule modes via ema_type:
  - "const": fixed decay rate (e.g. 0.9999)
  - "edm":   step-dependent decay via halflife ramp-up (Karras et al. 2024)

usage:
    ema = EMAModel(model, ema_type="const", values=[0.9999, 0.9996])
    ema = EMAModel(model, ema_type="edm", values=[500, 1000], batch_size=1024)

    # training: call after optimizer.step()
    ema.step(model)

    # evaluation: context manager swaps in EMA weights, auto-restores on exit
    with ema.swap(model):                       # default (first) copy
        evaluate(model)
    with ema.swap(model, label="0.9996"):       # specific copy
        evaluate(model)
    with ema.swap(model, label="online"):        # no-op (online model, no swap)
        evaluate(model)

    # checkpoint
    torch.save({"ema": ema.state_dict()}, path)
    ema.load_state_dict(checkpoint["ema"])
"""

import logging
from contextlib import contextmanager

import torch

from src.utils.runtime_util import normalize_param_name

logger = logging.getLogger("FD_loss")


def const_schedule(step: int, batch_size: int, value: float) -> float:
    """constant decay — returns value as-is."""
    return value


def edm_schedule(step: int, batch_size: int, halflife_kimg: float) -> float:
    """edm-style decay (Karras et al. 2024): halflife ramps up during first 5% of training."""
    halflife_nimg = halflife_kimg * 1000
    rampup_ratio = 0.05
    halflife_nimg = min(halflife_nimg, step * batch_size * rampup_ratio)
    return 0.5 ** (batch_size / max(halflife_nimg, 1e-8))


SCHEDULES = {"const": const_schedule, "edm": edm_schedule}


class EMAModel:
    """
    exponential moving average of model parameters.

    maintains one or more shadow copies identified by string labels.
    supports constant and step-dependent decay schedules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ema_type: str = "const",
        values: list[float] | None = None,
        batch_size: int = 1,
    ):
        """
        args:
            model:      model whose parameters will be tracked.
            ema_type:   "const" or "edm".
            values:     for "const": list of decay rates (default [0.9999]).
                        for "edm": list of halflife in kimg (default [500, 1000, 2000]).
            batch_size: global batch size, used by edm schedule.
        """
        if ema_type not in SCHEDULES:
            raise ValueError(f"unknown ema_type '{ema_type}'. use one of {list(SCHEDULES.keys())}.")

        self.schedule_fn = SCHEDULES[ema_type]
        self.batch_size = batch_size
        self.step_count: int = 0

        # build (label, value) pairs
        if ema_type == "const":
            vals = values or [0.9999, 0.9996]
            self.schedules = [(str(v), float(v)) for v in vals]
        else:
            vals = values or [500, 1000, 2000]
            self.schedules = [(f"edm_{v}", float(v)) for v in vals]

        self.labels = [label for label, _ in self.schedules]

        # shadow parameters: {label: {normalized_name: tensor}}
        self.shadows: dict[str, dict[str, torch.Tensor]] = {}
        for label, _ in self.schedules:
            self.shadows[label] = {
                normalize_param_name(n): (p.clone().detach() if p.requires_grad else p)
                for n, p in model.named_parameters()
            }

    @property
    def default_label(self) -> str:
        return self.labels[0]

    @torch.no_grad()
    def step(self, model: torch.nn.Module) -> None:
        """update all EMA copies from current model weights. call once per training step."""
        self.step_count += 1
        for label, value in self.schedules:
            decay = self.schedule_fn(self.step_count, self.batch_size, value)
            shadow = self.shadows[label]
            for name, param in model.named_parameters():
                n = normalize_param_name(name)
                if param.requires_grad:
                    shadow[n].lerp_(param.data, 1.0 - decay)
                else:
                    shadow[n].copy_(param.data)

    @contextmanager
    def swap(self, model: torch.nn.Module, label: str | None = None):
        """temporarily replace model weights with EMA weights; restores on exit.

        - ``swap(model)``                 -> swap in the default (first) EMA copy.
        - ``swap(model, label="0.9999")`` -> swap in the named EMA copy.
        - ``swap(model, label="online")`` -> no-op (online model, no swap).
        """
        if label == "online":
            yield
            return
        label = label or self.default_label
        if label not in self.shadows:
            raise ValueError(f"unknown ema label '{label}'. available: {self.labels}")

        # save current weights (stay on device to avoid slow D2H copy)
        stored = {normalize_param_name(n): p.data.clone() for n, p in model.named_parameters()}
        shadow = self.shadows[label]
        for name, param in model.named_parameters():
            param.data.copy_(shadow[normalize_param_name(name)].to(param.device))
        try:
            yield
        finally:
            for name, param in model.named_parameters():
                param.data.copy_(stored[normalize_param_name(name)])

    def to(self, device=None, dtype=None) -> "EMAModel":
        """move all shadow parameters to the given device and/or dtype."""
        for shadow in self.shadows.values():
            for name, param in shadow.items():
                shadow[name] = param.to(device=device, dtype=dtype)
        return self

    def __repr__(self) -> str:
        return f"EMAModel(labels={self.labels}, step_count={self.step_count}, schedule={self.schedule_fn.__name__})"

    # -- serialization ---------------------------------------------------------

    def state_dict(self, label: str | None = None) -> dict:
        """full checkpoint dict, or shadow params for a single label."""
        if label is not None:
            return dict(self.shadows[label])
        return {
            "step_count": self.step_count,
            "schedule": self.schedule_fn.__name__,
            "batch_size": self.batch_size,
            "schedules": self.schedules,
            "shadows": {label: dict(s) for label, s in self.shadows.items()},
        }

    def load_state_dict(self, state_dict: dict, label: str | None = None) -> None:
        """
        load shadow parameters. two formats:
          1. full:        {"shadows": {label: {name: tensor}}, "step_count": ...}
          2. single-copy: {name: tensor}  -> loaded into specified/default label
        """
        if not isinstance(state_dict, dict):
            raise ValueError(f"state_dict must be a dict, got {type(state_dict)}")

        # full format
        if "shadows" in state_dict:
            self.step_count = state_dict.get("step_count", 0)
            available_labels = list(state_dict["shadows"].keys())
            for lbl in self.labels:
                if lbl in state_dict["shadows"]:
                    _copy_params(self.shadows[lbl], state_dict["shadows"][lbl])
                elif available_labels:
                    fallback = available_labels[0]
                    logger.warning(
                        f"ema label '{lbl}' not found in checkpoint, "
                        f"copying from '{fallback}' instead"
                    )
                    _copy_params(self.shadows[lbl], state_dict["shadows"][fallback])
                else:
                    logger.warning(f"ema label '{lbl}' not found and no fallback available in checkpoint")
            return

        # single-copy format: {name: tensor}
        target = label or self.default_label
        _copy_params(self.shadows[target], state_dict)


def _copy_params(shadow: dict[str, torch.Tensor], params: dict) -> None:
    """copy params into shadow dict, matching by normalized name with suffix fallback."""
    normed = {normalize_param_name(k): v for k, v in params.items()}

    # suffix index: fallback when exact match fails (e.g. "backbone.X" vs "X")
    suffix_map: dict[str, torch.Tensor] = {}
    for k, v in normed.items():
        suffix = k.split(".", 1)[-1]
        suffix_map[suffix] = v

    matched, skipped = 0, []
    for name in shadow:
        src = normed.get(name)
        if src is None:
            suffix = name.split(".", 1)[-1]
            src = suffix_map.get(suffix) or suffix_map.get(name)
        if src is not None:
            shadow[name].data.copy_(src.to(shadow[name].device))
            matched += 1
        else:
            skipped.append(name)

    if skipped:
        logger.warning(f"ema load: {matched} matched, {len(skipped)} skipped (e.g. {skipped[0]})")
