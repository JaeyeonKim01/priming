import json
import logging
import math
import os
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.runtime_util import normalize_param_name
from src.utils.distributed_util import is_main_process


logger = logging.getLogger("FD_loss")

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_MODEL_NAME = "adapter_model.pth"
ADAPTER_DIR_PREFIX = "lora_adapter_step_"
LATEST_ADAPTER_NAME = "lora_adapter_latest"

LORA_TARGETS = {
    "attention": (
        "attn.q_proj.linear",
        "attn.k_proj.linear",
        "attn.v_proj.linear",
        "attn.out_proj.linear",
        "attn.qkv",
        "attn.proj",
    ),
    "attention_mlp": (
        "attn.q_proj.linear",
        "attn.k_proj.linear",
        "attn.v_proj.linear",
        "attn.out_proj.linear",
        "attn.qkv",
        "attn.proj",
        "mlp.w1.linear",
        "mlp.w2.linear",
        "mlp.w3.linear",
        "mlp.w12",
        "mlp.w3",
    ),
}


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for this repo's plain PyTorch models."""

    r: int = 8
    lora_alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
    lora_dropout: float = 0.0
    bias: str = "none"
    lora_mode: str = "attention"
    base_model_name_or_path: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        kwargs = dict(data)
        if "rank" in kwargs and "r" not in kwargs:
            kwargs["r"] = kwargs.pop("rank")
        if "alpha" in kwargs and "lora_alpha" not in kwargs:
            kwargs["lora_alpha"] = kwargs.pop("alpha")
        if "dropout" in kwargs and "lora_dropout" not in kwargs:
            kwargs["lora_dropout"] = kwargs.pop("dropout")
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in kwargs.items() if k in valid})


def lora_enabled(args) -> bool:
    return bool(getattr(args, "use_lora", False) or getattr(args, "lora_adapter_path", None))


class LoRALinear(nn.Linear):
    """Linear layer with LoRA adapter matrices stored under a default adapter."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        if config.r <= 0:
            raise ValueError(f"lora rank must be positive, got {config.r}")
        if config.bias != "none":
            raise ValueError("Only bias='none' is supported for LoRA adapters")
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight.data.copy_(base.weight.data)
        if base.bias is not None:
            self.bias.data.copy_(base.bias.data)

        self.r = int(config.r)
        self.lora_alpha = float(config.lora_alpha)
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.ModuleDict({
            "default": nn.Dropout(float(config.lora_dropout))
            if config.lora_dropout > 0 else nn.Identity()
        })
        self.lora_A = nn.ModuleDict({
            "default": nn.Linear(base.in_features, self.r, bias=False,
                                 device=base.weight.device, dtype=base.weight.dtype)
        })
        self.lora_B = nn.ModuleDict({
            "default": nn.Linear(self.r, base.out_features, bias=False,
                                 device=base.weight.device, dtype=base.weight.dtype)
        })
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B["default"].weight)

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.train(base.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        lora_x = self.lora_dropout["default"](x)
        lora_out = self.lora_B["default"](self.lora_A["default"](lora_x))
        return out + lora_out * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        delta = self.lora_B["default"].weight @ self.lora_A["default"].weight
        return self.weight + delta * self.scaling


def create_lora_config(args) -> LoRAConfig:
    mode = getattr(args, "lora_mode", "attention")
    if mode not in LORA_TARGETS:
        raise ValueError(f"unknown lora_mode={mode!r}; expected one of {list(LORA_TARGETS)}")
    return LoRAConfig(
        r=int(getattr(args, "lora_rank", 8)),
        lora_alpha=float(getattr(args, "lora_alpha", 16.0)),
        target_modules=list(LORA_TARGETS[mode]),
        lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
        bias="none",
        lora_mode=mode,
        base_model_name_or_path=getattr(args, "model", None),
    )


def apply_lora_config(model: nn.Module, lora_config: LoRAConfig) -> nn.Module:
    """Attach LoRA adapters in-place using a LoRAConfig."""
    replacements = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in lora_config.target_modules):
            continue
        parent, child_name = _parent_and_child(model, name)
        setattr(parent, child_name, LoRALinear(module, lora_config))
        replacements.append(name)

    if not replacements:
        raise ValueError(
            f"LoRA config did not match any nn.Linear modules. "
            f"Targets: {', '.join(lora_config.target_modules)}"
        )

    mark_only_lora_as_trainable(model)
    model.lora_config = lora_config
    model.lora_matched_modules = replacements

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[LoRA] apply_lora_config: mode={lora_config.lora_mode}, r={lora_config.r}, "
        f"alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}: "
        f"wrapped {len(replacements)} modules, trainable={trainable:,}/{total:,} "
        f"({100.0 * trainable / max(total, 1):.4f}%)"
    )
    for name in replacements:
        logger.info(f"[LoRA] wrapped {name}")
    return model


def prepare_lora_args(args):
    """Enable LoRA from an adapter directory and hydrate adapter config onto args."""
    adapter_path = getattr(args, "lora_adapter_path", None)
    if adapter_path:
        config = read_lora_config(adapter_path)
        setattr(args, "use_lora", True)
        setattr(args, "lora_mode", config.lora_mode)
        setattr(args, "lora_rank", config.r)
        setattr(args, "lora_alpha", config.lora_alpha)
        setattr(args, "lora_dropout", config.lora_dropout)
        logger.info(f"[LoRA] Loaded adapter config from {adapter_path}: {config.to_dict()}")
    return args


def save_lora_outputs(args, model: nn.Module, step: int) -> None:
    if not lora_enabled(args) or not is_main_process():
        return
    config = getattr(model, "lora_config", None) or create_lora_config(args)
    adapter_root = lora_adapter_save_root(args)
    adapter_path = os.path.join(adapter_root, f"{ADAPTER_DIR_PREFIX}{step:07d}")
    save_lora_adapter(model, adapter_path, config)
    cleanup_lora_adapters(
        adapter_root,
        keep_num=getattr(args, "keep_n_ckpts", 3),
        milestone_every=getattr(args, "milestone_every", 0),
    )
    if getattr(args, "merge_lora_on_save", False):
        merged_path = os.path.join(args.ckpt_dir, "merged_lora_latest.pth")
        save_merged_lora_checkpoint(model, merged_path, config, step=step)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def read_lora_config(path: str) -> LoRAConfig:
    config_path = os.path.join(path, ADAPTER_CONFIG_NAME)
    with open(config_path, "r") as f:
        return LoRAConfig.from_dict(json.load(f))


def save_lora_adapter(model: nn.Module, path: str, config: LoRAConfig) -> None:
    state = collect_lora_state_dict(model, to_cpu=True)
    if not state:
        raise ValueError("cannot save LoRA adapter: model has no LoRA parameters")

    os.makedirs(path, exist_ok=True)
    config_dict = config.to_dict()
    config_dict["matched_target_modules"] = sorted({k.split(".lora_", 1)[0] for k in state})
    with open(os.path.join(path, ADAPTER_CONFIG_NAME), "w") as f:
        json.dump(config_dict, f, indent=2, sort_keys=True)
    torch.save(state, os.path.join(path, ADAPTER_MODEL_NAME))
    logger.info(f"[LoRA] Saved adapter directory: {path}")


def lora_adapter_save_root(args) -> str:
    requested = getattr(args, "save_lora_adapter", None)
    if requested in (None, "", "auto"):
        return args.ckpt_dir
    return requested


def cleanup_lora_adapters(root: str, keep_num: int = 5, milestone_every: int = 50000) -> None:
    """Keep latest N LoRA adapter dirs plus milestone adapters, like checkpoint cleanup."""
    if not os.path.isdir(root):
        return

    adapters = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or os.path.islink(path):
            continue
        step = _adapter_step(name)
        if step is not None:
            adapters.append((step, path))

    if not adapters:
        return

    adapters.sort(key=lambda item: item[0])
    newest_keep = {path for _, path in adapters[-keep_num:]}
    milestone_keep = {
        path for step, path in adapters
        if milestone_every > 0 and step % milestone_every == 0
    }
    keep_set = newest_keep.union(milestone_keep)

    for _, path in adapters:
        if path not in keep_set:
            shutil.rmtree(path)
            logger.info(f"[LoRA] Removed adapter directory: {path}")

    newest_adapter = os.path.abspath(adapters[-1][1])
    latest_path = os.path.join(root, LATEST_ADAPTER_NAME)
    _replace_latest_adapter_link(latest_path, newest_adapter)


def _adapter_step(name: str) -> int | None:
    if not name.startswith(ADAPTER_DIR_PREFIX):
        return None
    try:
        return int(name[len(ADAPTER_DIR_PREFIX):])
    except ValueError:
        return None


def _replace_latest_adapter_link(latest_path: str, target: str) -> None:
    try:
        if os.path.islink(latest_path) or os.path.isfile(latest_path):
            os.remove(latest_path)
        elif os.path.isdir(latest_path):
            shutil.rmtree(latest_path)
    except FileNotFoundError:
        pass

    os.symlink(target, latest_path)
    logger.info(f"[LoRA] Created symlink: {latest_path} -> {target}")


def load_lora_adapter(model: nn.Module, path: str) -> LoRAConfig:
    config = read_lora_config(path)
    model_path = os.path.join(path, ADAPTER_MODEL_NAME)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} does not contain a non-empty LoRA state dict")

    current = collect_lora_state_dict(model, to_cpu=False)
    missing = sorted(set(state) - set(current))
    if missing:
        raise ValueError(f"adapter has keys that do not exist in this model, e.g. {missing[0]}")

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))
    logger.info(f"[LoRA] Loaded adapter from {path} ({len(state)} tensors)")
    return config


def collect_lora_state_dict(model: nn.Module, to_cpu: bool = True) -> OrderedDict:
    state = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        tensor = param.detach()
        state[name] = tensor.cpu() if to_cpu else tensor
    return state


def sync_lora_to_ema(model_ema, model: nn.Module) -> None:
    if model_ema is None or not hasattr(model_ema, "shadows"):
        return
    lora_params = {
        normalize_param_name(name): param.detach()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    if not lora_params:
        return
    for shadow in model_ema.shadows.values():
        for name, param in lora_params.items():
            if name in shadow:
                shadow[name].copy_(param.to(device=shadow[name].device, dtype=shadow[name].dtype))
    logger.info(f"[LoRA] Synced {len(lora_params)} LoRA tensors into EMA shadows")


def merged_lora_state_dict(model: nn.Module, to_cpu: bool = True) -> OrderedDict:
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    state = OrderedDict()
    for key, tensor in model.state_dict().items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        value = tensor
        if key.endswith(".weight"):
            module_name = key[:-len(".weight")]
            module = modules.get(module_name)
            if module is not None:
                value = module.merged_weight()
        value = value.detach()
        state[key] = value.cpu() if to_cpu else value
    return state


def save_merged_lora_checkpoint(model: nn.Module, path: str, config: LoRAConfig, step: int | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": merged_lora_state_dict(model, to_cpu=True),
        "lora_config": config.to_dict(),
        "merged_from_lora": True,
    }
    if step is not None:
        payload["step"] = step
    torch.save(payload, path)
    logger.info(f"[LoRA] Saved merged checkpoint: {path}")


def _parent_and_child(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
