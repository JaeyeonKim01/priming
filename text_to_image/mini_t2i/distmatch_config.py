"""Separate configuration for one-step matching (not diffusion training)."""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path

import yaml

from . import settings
from .config import TrainConfig
from .multilayer_features import (
    DEFAULT_LAYERS, DINOV3_REVISION, FEATURE_SET as MULTILAYER_FEATURE_SET,
    multilayer_feature_weights, validate_layers,
)
from .sim_features import SIM_RECIPES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DINOV3_REPO = str(settings.ASSET_ROOT / "dinov3" / "source")
DEFAULT_DINOV3_WEIGHTS = str(settings.ASSET_ROOT / "dinov3" / "dinov3_vitl16.pth")
DEFAULT_SIGLIP_WEIGHTS = str(settings.ASSET_ROOT / "siglip" / "model.safetensors")
DEFAULT_VISUALIZATION_PROMPTS = [
    "A red panda wearing a blue baseball cap.",
    "A watercolor painting of a lighthouse in a storm.",
    "A glass teapot filled with yellow flowers on a wooden table.",
    "An astronaut riding a horse through a snowy forest.",
]


@dataclass
class DistMatchTrainConfig:
    init_from: str = str(settings.CHECKPOINT_ROOT / "minit2i-b-16.pt")
    init_weights: str = "model"
    resume_from: str | None = None
    output_dir: str = str(settings.OUTPUT_ROOT / "train" / "distmatch_b16_sim")
    dataset_dir: str = str(settings.FINETUNE_TENSOR_ROOT)
    expected_samples: int = 120000
    n_c: int = 4                    # Distinct caption/example IDs PER RANK.
    n_pos: int = 1                  # T2I: exactly one real image per caption.
    b: int = 4                     # Each of the two generated populations.
    input_times: list[float] | None = None
    samples_per_time: list[int] | None = None
    time_matching: str = "joint"
    use_replay: bool = True        # False: retain the full live graph and call loss.backward().
    replay_microbatch_size: int = 4
    verify_replay: bool = False
    mode: str = "conditional"
    objective: str = "mmd"
    kernel: str = "rbf"
    bandwidth: float = 0.05         # Dimensionless R, NOT Gaussian sigma.
    sinkhorn_epsilon: float = 0.05
    sinkhorn_max_iter: int = 10
    sinkhorn_tolerance: float = 1e-6
    sinkhorn_strict: bool = False
    feature_set: str = "sim"
    feature_pooling: str = "sim"
    feature_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(SIM_RECIPES, 1.0))
    sim_weights: dict[str, str] = field(default_factory=dict)
    feature_layers: dict[str, list[int]] = field(
        default_factory=lambda: {name: list(values) for name, values in DEFAULT_LAYERS.items()}
    )
    feature_family_weights: dict[str, float] = field(
        default_factory=lambda: {"dinov3": 0.5, "siglip": 0.5}
    )
    feature_microbatch_size: int = 8
    feature_checkpointing: bool = True
    dinov3_repo: str = DEFAULT_DINOV3_REPO
    dinov3_weights: str = DEFAULT_DINOV3_WEIGHTS
    dinov3_revision: str = DINOV3_REVISION
    siglip_weights: str = DEFAULT_SIGLIP_WEIGHTS
    max_pairwise_elements: int = 16000000
    seed: int = 42
    num_steps: int = 40000
    warmup_steps: int = 1000
    learning_rate: float = 0.00004
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ema_decay: float = 0.99995
    amp_dtype: str = "bf16"
    device: str = "cuda"
    attention_impl: str = "sdpa"
    mean_flow_omega: float = 1.0
    num_workers: int = 1
    dataloader_prefetch: int = 2
    log_every: int = 10
    ckpt_every: int = 1000
    vis_every: int = 0              # Opt-in; the main YAML enables every 1000 steps.
    vis_at_start: bool = True
    visualization_seed: int = 123
    visualization_samples_per_prompt: int = 4
    visualization_batch_size: int = 4  # Inference only, unrelated to training replay.
    visualization_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_VISUALIZATION_PROMPTS))

    def time_groups(self) -> tuple[tuple[float, int], ...]:
        """Return ordered (input time, population size) groups.

        Older recipes specify only ``b`` and retain their exact t=1 behavior.
        Multi-time recipes opt in by specifying both lists explicitly.
        """
        if self.input_times is None:
            return ((1.0, self.b),)
        return tuple(zip(self.input_times, self.samples_per_time, strict=True))

    @property
    def total_b(self) -> int:
        return sum(size for _, size in self.time_groups())

    def validate(self):
        for name in ("use_replay", "verify_replay", "vis_at_start", "sinkhorn_strict",
                     "feature_checkpointing"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.verify_replay and not self.use_replay:
            raise ValueError("verify_replay requires use_replay=true")
        for name in ("expected_samples", "n_c", "n_pos", "b", "replay_microbatch_size",
                     "feature_microbatch_size", "num_steps",
                     "max_pairwise_elements", "sinkhorn_max_iter", "dataloader_prefetch", "log_every", "ckpt_every",
                     "visualization_samples_per_prompt", "visualization_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_pos != 1:
            raise ValueError("T2I matching requires n_pos=1")
        if (self.input_times is None) != (self.samples_per_time is None):
            raise ValueError("input_times and samples_per_time must be specified together")
        if self.input_times is not None:
            if (not isinstance(self.input_times, list) or not self.input_times
                    or not isinstance(self.samples_per_time, list)
                    or len(self.input_times) != len(self.samples_per_time)):
                raise ValueError("input_times and samples_per_time must be nonempty lists of equal length")
            if any(isinstance(t, bool) or not isinstance(t, (int, float))
                   or not math.isfinite(t) or not 0 < t <= 1 for t in self.input_times):
                raise ValueError("input_times must contain finite values in (0,1]")
            if any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for size in self.samples_per_time):
                raise ValueError("samples_per_time must contain positive integers")
            if len(set(float(t) for t in self.input_times)) != len(self.input_times):
                raise ValueError("input_times must be unique")
        for name in ("num_workers", "warmup_steps", "seed", "vis_every", "visualization_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (not isinstance(self.visualization_prompts, list)
                or any(not isinstance(p, str) or not p.strip() for p in self.visualization_prompts)
                or (self.vis_every and not self.visualization_prompts)):
            raise ValueError("visualization_prompts must be a list of nonempty strings, nonempty when visualization is enabled")
        for name in ("bandwidth", "sinkhorn_epsilon", "sinkhorn_tolerance", "learning_rate", "mean_flow_omega"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("ema_decay and adam_beta2 must be in [0,1)")
        for name, choices in {"mode": {"conditional", "marginal"}, "objective": {"mmd", "sinkhorn"},
                              "kernel": {"rbf", "laplacian", "energy"}, "amp_dtype": {"fp32", "bf16"},
                              "device": {"cpu", "cuda"}, "init_weights": {"model", "ema"},
                              "attention_impl": {"einsum", "sdpa"},
                              "time_matching": {"joint", "separate"},
                              "feature_set": {"sim", MULTILAYER_FEATURE_SET},
                              "feature_pooling": {"sim", "none"}}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if self.feature_set == "sim":
            if set(self.feature_weights) != set(SIM_RECIPES):
                raise ValueError("feature_weights must specify siglip, mae, inception")
            if set(self.sim_weights) - set(SIM_RECIPES):
                raise ValueError("unknown sim_weights branch")
            feature_values = self.feature_weights.values()
            paths = self.sim_weights.values()
        else:
            if self.feature_pooling != "none":
                raise ValueError(f"{MULTILAYER_FEATURE_SET} requires feature_pooling=none")
            layers = validate_layers(self.feature_layers)
            automatic = multilayer_feature_weights(layers, self.feature_family_weights)
            if self.feature_weights and set(self.feature_weights) != set(automatic):
                raise ValueError("explicit feature_weights must specify all multi-layer branches")
            feature_values = (self.feature_weights or automatic).values()
            if self.sim_weights:
                raise ValueError("sim_weights is only valid for feature_set=sim")
            if self.dinov3_revision != DINOV3_REVISION:
                raise ValueError(f"dinov3_revision must be the pinned revision {DINOV3_REVISION}")
            repo = Path(self.dinov3_repo)
            if not (repo / "hubconf.py").is_file():
                raise FileNotFoundError(f"DINOv3 source has no hubconf.py: {repo}")
            paths = (self.dinov3_weights, self.siglip_weights)
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 for value in feature_values)
                or sum(feature_values) <= 0):
            raise ValueError("feature weights must be nonnegative with positive sum")
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        return self

    def resolved_feature_weights(self):
        if self.feature_set == "sim" or self.feature_weights:
            return dict(self.feature_weights)
        return multilayer_feature_weights(self.feature_layers, self.feature_family_weights)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open() as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("config must be a YAML mapping")
        values = settings.expand_setting(values)
        # Explicit paths in YAML are relative to the repository, not caller cwd.
        for key in ("init_from", "resume_from", "output_dir", "dataset_dir", "dinov3_repo",
                    "dinov3_weights", "siglip_weights"):
            if values.get(key):
                value = Path(values[key]).expanduser()
                values[key] = str(value if value.is_absolute() else ROOT / value)
        for key, value in values.get("sim_weights", {}).items():
            value = Path(value).expanduser()
            values["sim_weights"][key] = str(value if value.is_absolute() else ROOT / value)
        return cls(**values).validate()


def model_config(checkpoint, cfg):
    values = checkpoint.get("config")
    if not isinstance(values, dict):
        raise ValueError("checkpoint must include native MiniT2I architecture config")
    required = {"image_size", "prompt_length", "t5_name", "noise_scale", "patch_size", "hidden_size", "t5_hidden_size", "depth_double",
                "text_preamble_depth", "num_heads", "head_dim", "mlp_ratio", "pca_channels"}
    if required - set(values):
        raise ValueError(f"missing checkpoint architecture fields: {sorted(required - set(values))}")
    known = {f.name for f in fields(TrainConfig)}
    model_cfg = TrainConfig(**{k: v for k, v in values.items() if k in known})
    if model_cfg.prediction != "x" or model_cfg.model_variant not in {"diffusion", "mean_flow"}:
        raise ValueError("one-step training requires an x-predicting MiniT2I checkpoint")
    model_cfg.attention_impl = cfg.attention_impl
    model_cfg.n_T, model_cfg.cfg_scale = 1, cfg.mean_flow_omega if model_cfg.model_variant == "mean_flow" else 1.0
    model_cfg.output_dir = cfg.output_dir
    return model_cfg
