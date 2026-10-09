"""Load generator weights from the existing full-checkpoint format.

Training-state restoration belongs to the training loop. These helpers select
online or EMA weights without modifying optimizer, RNG, or checkpoint counters.
"""
import inspect
import json
from pathlib import Path

import torch

from models import Priming
from .files import sha256


def load_weights(model, path, *, stage="posttraining", weights="ema", terminal=False):
    """Load online/EMA weights, preserving their dtype. Call before making an optimizer."""
    if stage not in {"pretraining", "posttraining"}:
        raise ValueError("stage must be pretraining or posttraining")
    if weights not in {"online", "ema"}:
        raise ValueError("weights must be online or ema")
    if terminal and stage != "posttraining":
        raise ValueError("terminal evaluation requires posttraining")
    state = torch.load(Path(path), map_location="cpu", mmap=True, weights_only=False)
    label = "0.9999" if stage == "pretraining" else "0.999"
    if terminal:
        if state.get("step") != 24999 or state.get("current_step") != 25000:
            raise ValueError("Expected 25,000 completed post-training updates")
        if state.get("model_ema", {}).get("schedules") != [(label, float(label))]:
            raise ValueError("Expected terminal EMA 0.999")
    selected = state["model"] if weights == "online" else state["model_ema"]["shadows"][label]
    expected = model.state_dict()
    if set(selected) != set(expected):
        raise ValueError("Checkpoint tensor names differ from the model")
    for name, value in selected.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape:
            raise ValueError(f"Checkpoint tensor shape differs: {name}")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"Checkpoint tensor is not finite floating-point data: {name}")
    device = next(model.parameters()).device
    model.load_state_dict(selected, strict=True, assign=True)
    model.to(device)


def load_released_model(key, *, root=None, checkpoint=None, stage="posttraining",
                        weights="ema", device="cpu", verify_release=True, terminal=False):
    """Load B/L with its pinned inference config; explicit custom weights may be used."""
    if stage not in {"pretraining", "posttraining"}:
        raise ValueError("stage must be pretraining or posttraining")
    root = Path(root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    manifest = "pretraining_manifest.json" if stage == "pretraining" else "manifest.json"
    item = json.loads((root / "old/checkpoints" / manifest).read_text())[key]
    path = Path(checkpoint).expanduser() if checkpoint is not None else Path(item["checkpoint"])
    if not path.is_absolute():
        path = root / path
    if verify_release and (path.stat().st_size != item["checkpoint_bytes"] or sha256(path) != item["checkpoint_sha256"]):
        raise ValueError(f"Checkpoint does not match released {stage}/{key}")
    config_path = root / "old/checkpoints" / f"{key}.json"
    config_record = json.loads((root / "old/checkpoints/manifest.json").read_text())[key]
    if sha256(config_path) != config_record["config_sha256"]:
        raise ValueError("Released inference configuration changed")
    args = json.loads(config_path.read_text())["sampling_args"]
    kwargs = {name: args[name] for name in inspect.signature(Priming).parameters if name in args}
    kwargs.update(backbone=args["model"], in_channels=args["token_channels"],
                  tokenizer_patch_size=1, corruption="denoising", t_dist="lognorm",
                  t_min=0., t_max=1.)
    model = Priming(**kwargs)
    load_weights(model, path, stage=stage, weights=weights, terminal=terminal)
    return model.to(device).eval().requires_grad_(False), args
