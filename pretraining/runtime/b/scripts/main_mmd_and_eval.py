"""Train and evaluate OST with the canonical full-MAE featurewise OT recipe.

Historical SIM OT, queue-MMD and frozen rank interventions remain explicit
configurations for reproducing the scientific controls.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "ot_full_mae.json"
DISTMATCH_ROOT = REPO_ROOT / "packages" / "distmatch"
DISTMATCH_SRC = REPO_ROOT / "packages" / "distmatch" / "src"
STATS_DIR = REPO_ROOT / "data" / "fid_stats"

for root in (REPO_ROOT, DISTMATCH_SRC, DISTMATCH_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
import torch.distributed as torch_dist
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

from distmatch import (
    BalancedSinkhornOT,
    Feature,
    FrechetEvaluator,
    MMD,
    SharedForwardXM,
    feature_stats,
    gradcache_backward,
)
from distmatch.distributed import sync_gradients
from distmatch.objectives.loss_ablation import LossAblationOT
from tools import RealFeatureBank
from src.eval.inception import (
    _accumulate_inception_batch,
    _prepare_inception_evaluation,
    run_inception_evaluation,
)
from src.eval.results import _append_eval_csv
from src.utils.builders import create_generation_model, create_tokenizer
from src.utils.checkpoint_util import ckpt_resume, save_checkpoint
from src.utils.distributed_util import (
    all_reduce_mean,
    get_global_rank,
    get_world_size,
    is_enabled,
    preempt_requested,
    register_preempt_handler,
)
from src.utils.eval_util import _prepare_eval_classes, get_start_end_indices
from src.utils.feature_intervention import (
    FrozenJointSIMIntervention,
    verify_artifact_integrity,
)
from src.utils.grad_util import get_grad_norm
from src.utils.logging_util import MetricLogger, SmoothedValue
from src.utils.optimizer_util import create_optimizer
from src.utils.ot_util import ClasswiseOTBatch, ClasswiseOTStream
from src.utils.rng_util import RNGStateManager
from src.utils.sampling_util import generate_images, sample_generation_latents
from src.utils.schedule_util import adjust_learning_rate
from src.utils.setup_util import setup
from src.utils.vis_util import visualize
from src.utils.xm_util import ClasswiseXMStream


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if os.environ.get("OT_DETERMINISTIC_DIAGNOSTIC", "0") == "1":
    # A validation-only control separates checkpoint state from autotuner
    # kernel selection across fresh process lifetimes. Production retains
    # the original baseline CUDA settings above.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

# setup() configures the shared OST training logger under this name.
logger = logging.getLogger("FD_loss")
MMD_QUEUE_STATE_KEY = "mmd_repulsion_queue_state"
MMD_RNG_STATE_KEY = "mmd_rng_state"
MMD_RNG_STATE_VERSION = 1
XM_STREAM_STATE_KEY = "forward_xm_stream_state"
XM_STREAM_STATE_VERSION = 1
OT_STREAM_STATE_KEY = "sinkhorn_ot_stream_state"
OT_STREAM_STATE_VERSION = 2
FEATURE_INTERVENTION_STATE_KEY = "frozen_joint_sim_intervention_state"
MODELS = {
    "siglip": "vit_so400m_patch16_siglip_256.v2_webli",
    "mae": "vit_large_patch16_224.mae",
    "dinov2": "vit_large_patch14_dinov2.lvd142m",
    "clip": "vit_large_patch14_clip_224.openai",
    "convnext": "convnextv2_base.fcmae_ft_in22k_in1k",
}

TRAINING_SPECS = (
    (
        "siglip",
        MODELS["siglip"],
        "cls",
        224,
        "vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
    ),
    (
        "mae",
        MODELS["mae"],
        "cls",
        224,
        "vit_large_patch16_224_mae_in256_t224_stats.npz",
    ),
    ("inception", "inception", "cls", None, "guided_diffusion_stats.npz"),
)

FDR6_SPECS = (
    ("FID(ADM)", "inception", "cls", None, "guided_diffusion_stats.npz"),
    (
        "convnext",
        MODELS["convnext"],
        "cls",
        224,
        "convnext_in256_t224_stats.npz",
    ),
    (
        "dinov2_cls",
        MODELS["dinov2"],
        "cls",
        256,
        "vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz",
    ),
    (
        "mae_cls",
        MODELS["mae"],
        "cls",
        224,
        "vit_large_patch16_224_mae_in256_t224_stats.npz",
    ),
    (
        "siglip_cls",
        MODELS["siglip"],
        "cls",
        224,
        "vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
    ),
    (
        "clip_cls",
        MODELS["clip"],
        "cls",
        256,
        "vit_large_patch14_clip_224_openai_in256_t256_stats.npz",
    ),
)

FDR6_VALIDATION_FD = {
    "FID(ADM)": 1.68,
    "convnext": 56.87,
    "dinov2_cls": 14.19,
    "mae_cls": 0.04,
    "siglip_cls": 0.60,
    "clip_cls": 5.60,
}

INTERVENED_EVAL_FEATURE_KEYS = {
    "siglip": "siglip_cls",
    "mae": "mae_cls",
    "inception": "FID(ADM)",
}
SIM3_EVAL_NAMES = ("FID(ADM)", "mae_cls", "siglip_cls")

CONFIG_ARG_SECTIONS = (
    "train",
    "model",
    "tokenizer",
    "optimizer",
    "diffusion",
    "sampling",
    "data",
    "checkpointing",
    "evaluation",
    "mmd",
    "xm",
    "ot",
    "feature_intervention",
    "dashboards",
    "logging",
    "system",
)


def _features(specs) -> list[Feature]:
    return [
        Feature(model=model, name=name, pool=pool, target_size=target_size)
        for name, model, pool, target_size, _ in specs
    ]


def _cached_stats(specs, *, device: torch.device):
    """Load existing OST .npz files through feature_stats(source=None, ...)."""
    features = _features(specs)
    cache_features = [
        replace(feature, name=Path(spec[-1]).stem)
        for feature, spec in zip(features, specs)
    ]
    cached = feature_stats(
        features=cache_features,
        stats_dir=STATS_DIR,
        device=device,
        distributed="auto",
    )
    return {
        feature.name: cached[cache_feature.name]
        for feature, cache_feature in zip(features, cache_features)
    }


def _training_weights(args: argparse.Namespace) -> dict[str, float]:
    values = args.mmd_repr_weights or [1.0] * len(TRAINING_SPECS)
    if len(values) == 1:
        values = list(values) * len(TRAINING_SPECS)
    if len(values) != len(TRAINING_SPECS):
        raise ValueError(
            f"mmd_repr_weights must have length 1 or {len(TRAINING_SPECS)}"
        )
    return {spec[0]: float(weight) for spec, weight in zip(TRAINING_SPECS, values)}


def _feature_bank_dir(args: argparse.Namespace) -> Path:
    path = Path(args.mmd_feature_bank_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _feature_bank_summary(args: argparse.Namespace) -> dict[str, Any]:
    path = _feature_bank_dir(args) / "build_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"feature-bank build summary does not exist: {path}")
    with path.open() as handle:
        summary = json.load(handle)
    if summary.get("format") != "distmatch.tools.imagenet_feature_banks":
        raise ValueError(f"unsupported feature-bank build summary: {path}")
    if summary.get("version") != 1:
        raise ValueError(
            f"unsupported feature-bank summary version {summary.get('version')!r}"
        )
    if not isinstance(summary.get("features"), dict):
        raise ValueError(f"feature-bank summary has no feature metadata: {path}")
    return summary


def _mmd_sigmas(
    args: argparse.Namespace,
    features: Sequence[Feature],
) -> Mapping[str, float]:
    """Resolve per-representation attraction-kernel bandwidths.

    Existing feature-bank manifests call this quantity ``bandwidth``; for the
    RBF implementation it is sigma in ``exp(-d^2 / (2 sigma^2))`` and for
    L2-Laplacian it is ell in ``exp(-d / ell)``.
    """
    configured = args.mmd_sigmas
    if configured != "feature_bank":
        if not isinstance(configured, Mapping):
            raise TypeError(
                "mmd_sigmas must be 'feature_bank' or a feature-name mapping"
            )
        return {str(name): float(value) for name, value in configured.items()}
    summary = _feature_bank_summary(args)
    metadata = summary["features"]
    missing = [feature.name for feature in features if feature.name not in metadata]
    if missing:
        raise KeyError(f"feature-bank summary is missing bandwidths for {missing}")
    return {
        feature.name: float(metadata[feature.name]["bandwidth"])
        for feature in features
    }


def _build_feature_intervention(
    args: argparse.Namespace,
    model,
    sigmas: Mapping[str, float],
    checkpoint_extra: Mapping[str, Any] | None,
) -> FrozenJointSIMIntervention | None:
    enabled = bool(getattr(args, "feature_intervention_enabled", False))
    saved_state = (
        checkpoint_extra.get(FEATURE_INTERVENTION_STATE_KEY)
        if checkpoint_extra
        else None
    )
    if not enabled:
        if saved_state is not None and int(args.current_step) > 0:
            raise ValueError(
                "cannot resume an intervened checkpoint with the intervention disabled"
            )
        return None

    configured_path = getattr(args, "feature_intervention_artifact", None)
    if configured_path is None:
        raise ValueError("feature_intervention_artifact is required when enabled")
    artifact_path = Path(configured_path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = REPO_ROOT / artifact_path
    verification_error = None
    if get_global_rank() == 0:
        try:
            verify_artifact_integrity(artifact_path)
        except Exception as error:  # broadcast the failure instead of stranding peers
            verification_error = f"{type(error).__name__}: {error}"
    if is_enabled():
        verification = [verification_error]
        torch_dist.broadcast_object_list(verification, src=0)
        verification_error = verification[0]
    if verification_error is not None:
        raise RuntimeError(
            f"feature-intervention artifact integrity check failed: {verification_error}"
        )
    latent_shape = (model.in_channels, model.input_size, model.input_size)
    intervention = FrozenJointSIMIntervention(
        artifact_path,
        rank=int(args.feature_intervention_rank),
        strength=float(args.feature_intervention_strength),
        device=next(model.parameters()).device,
        expected_num_classes=int(args.num_classes),
        expected_latent_shape=latent_shape,
        expected_feature_sigmas=sigmas,
    )
    if int(args.current_step) > 0:
        if not isinstance(saved_state, Mapping):
            raise RuntimeError(
                "resumed intervened checkpoint has no frozen intervention identity"
            )
        intervention.validate_checkpoint_state(saved_state)
    if get_global_rank() == 0:
        logger.info(
            "Loaded frozen joint-SIM intervention: artifact=%s id=%s rank=%s "
            "strength=%s expected_energy=lambda^2*s_y^2 latent_projection=fixed_coordinates",
            intervention.root,
            intervention.artifact_id,
            intervention.rank,
            intervention.strength,
        )
    return intervention


def _mmd_queue_bandwidths(args: argparse.Namespace) -> Mapping[str, float] | str:
    """Resolve queue bandwidths without reusing a cross-kernel scale silently."""
    configured = args.mmd_repulsion_queue_bandwidths
    if configured == "attraction":
        return "attraction"
    if not isinstance(configured, Mapping):
        raise TypeError(
            "mmd_repulsion_queue_bandwidths must be 'attraction' or a "
            "feature-name mapping"
        )
    return {str(name): float(value) for name, value in configured.items()}


def _feature_bank_device(
    args: argparse.Namespace,
    objective_device: torch.device,
) -> torch.device:
    configured = str(args.mmd_feature_bank_device).strip().lower()
    if configured in {"objective", "model"}:
        return objective_device
    if configured == "cuda" and objective_device.type == "cuda":
        return objective_device
    return torch.device(configured)


def _load_feature_banks(
    args: argparse.Namespace,
    objective: MMD,
) -> dict[str, RealFeatureBank]:
    bank_dir = _feature_bank_dir(args)
    summary = _feature_bank_summary(args)
    summary_features = summary["features"]
    bank_device = _feature_bank_device(args, objective.device)
    expected_classes = set(range(int(args.num_classes)))
    banks: dict[str, RealFeatureBank] = {}
    for resolved in objective.resolved:
        path = bank_dir / f"{resolved.name}.pt"
        if resolved.name not in summary_features:
            raise KeyError(
                f"feature-bank summary is missing metadata for {resolved.name!r}"
            )
        metadata = summary_features[resolved.name]
        if metadata.get("feature_id") != resolved.spec.feature_id:
            raise ValueError(
                f"feature bank {path} was built with feature_id "
                f"{metadata.get('feature_id')!r}, expected "
                f"{resolved.spec.feature_id!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"real feature bank does not exist: {path}; build class-conditional "
                "banks with tools.build_real_feature_banks first"
            )
        bank = RealFeatureBank.load(path, device=bank_device)
        if bank.name not in {None, resolved.name}:
            raise ValueError(
                f"feature bank {path} is named {bank.name!r}, expected "
                f"{resolved.name!r}"
            )
        if bank.feature_dim != resolved.spec.feature_dim:
            raise ValueError(
                f"feature bank {path} has dimension {bank.feature_dim}, expected "
                f"{resolved.spec.feature_dim}"
            )
        if int(metadata.get("count", -1)) != bank.count:
            raise ValueError(
                f"feature bank {path} has {bank.count} rows but its build "
                f"summary records {metadata.get('count')!r}"
            )
        if bank.labels is None:
            raise ValueError(
                f"feature bank {path} has no class labels; rebuild it with "
                "class_conditional=True"
            )
        actual_classes = set(bank.classes)
        if actual_classes != expected_classes:
            missing = sorted(expected_classes - actual_classes)
            unexpected = sorted(actual_classes - expected_classes)
            raise ValueError(
                f"feature bank {path} class IDs do not match ImageNet: "
                f"missing={missing}, unexpected={unexpected}"
            )
        banks[resolved.name] = bank
        if get_global_rank() == 0:
            logger.info(
                "Loaded class-conditional feature bank %s: path=%s count=%s "
                "classes=%s dim=%s device=%s",
                resolved.name,
                path,
                bank.count,
                len(bank.classes),
                bank.feature_dim,
                bank.device,
            )
    return banks


RolloutChunk = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FullClassAttraction:
    """Replicated real-feature references for every ImageNet class."""

    features: Mapping[str, torch.Tensor]
    labels: torch.Tensor
    anchors: Mapping[str, torch.Tensor]
    class_starts: torch.Tensor
    class_counts: torch.Tensor


def _build_full_class_attraction(
    args: argparse.Namespace,
    feature_banks: Mapping[str, RealFeatureBank],
) -> FullClassAttraction:
    class_ids = torch.arange(int(args.num_classes), dtype=torch.int64)

    features: dict[str, torch.Tensor] = {}
    anchors: dict[str, torch.Tensor] = {}
    common_labels = None
    common_counts = None
    common_starts = None
    total_bytes = 0
    for name, bank in feature_banks.items():
        rows, labels = bank.rows_for_classes(class_ids)
        if common_labels is None:
            common_labels = labels
        elif not torch.equal(labels, common_labels):
            raise RuntimeError("feature banks disagree on ImageNet class labels")
        unique_labels, counts = torch.unique_consecutive(
            labels, return_counts=True
        )
        starts = torch.cat(
            [
                counts.new_zeros(1),
                counts.cumsum(dim=0)[:-1],
            ]
        )
        if common_counts is None:
            common_counts = counts.detach().cpu()
            common_starts = starts.detach().cpu()
        elif not torch.equal(counts.detach().cpu(), common_counts):
            raise RuntimeError("feature banks disagree on ImageNet class populations")
        anchor_table = rows.new_zeros((int(args.num_classes), rows.shape[1]))
        anchor_table.index_copy_(
            0,
            unique_labels,
            rows.index_select(0, starts),
        )
        features[name] = rows
        anchors[name] = anchor_table
        total_bytes += rows.numel() * rows.element_size()
        if get_global_rank() == 0:
            logger.info(
                "Prepared full-class attraction bank %s: rows=%s classes=%s "
                "dim=%s device=%s",
                name,
                rows.shape[0],
                unique_labels.numel(),
                rows.shape[1],
                rows.device,
            )
    assert common_labels is not None
    assert common_counts is not None
    assert common_starts is not None
    if get_global_rank() == 0:
        logger.info(
            "Full-ImageNet within-class attraction references ready: rows=%s "
            "classes=%s total_storage=%.2f GiB",
            common_labels.shape[0],
            torch.unique(common_labels).numel(),
            total_bytes / (1024**3),
        )
    return FullClassAttraction(
        features,
        common_labels,
        anchors,
        common_starts,
        common_counts,
    )


def sample_rollout_chunks(args: argparse.Namespace, model) -> list[RolloutChunk]:
    """Sample the fixed inputs for one complete logical MMD rollout."""
    device = next(model.parameters()).device
    shape = (model.in_channels, model.input_size, model.input_size)
    return [
        (
            torch.randn(args.batch_size, *shape, device=device) * args.noise_scale,
            torch.randint(
                0,
                args.num_classes,
                (args.batch_size,),
                device=device,
            ),
        )
        for _ in range(args.grad_accum)
    ]


def sample_ot_rollout_chunks(
    args: argparse.Namespace,
    model,
    ot_stream: ClasswiseOTStream,
) -> tuple[list[RolloutChunk], ClasswiseOTBatch]:
    """Sample the main rollout with complete rank-local OT class groups."""
    batch = ot_stream.next_batch()
    if batch.generated_labels.shape[0] != int(args.local_rollout_size):
        raise RuntimeError(
            "Sinkhorn OT generated labels do not fill the local rollout: "
            f"labels={batch.generated_labels.shape[0]}, "
            f"local_rollout={args.local_rollout_size}"
        )
    device = next(model.parameters()).device
    if batch.generated_labels.device != device:
        raise RuntimeError(
            "Sinkhorn OT labels and model must share a device: "
            f"{batch.generated_labels.device} != {device}"
        )
    shape = (model.in_channels, model.input_size, model.input_size)
    label_chunks = batch.generated_labels.split(int(args.batch_size))
    if len(label_chunks) != int(args.grad_accum) or any(
        labels.shape[0] != int(args.batch_size) for labels in label_chunks
    ):
        raise RuntimeError("OT class groups cannot be split into configured chunks")
    chunks = [
        (
            torch.randn(args.batch_size, *shape, device=device) * args.noise_scale,
            labels,
        )
        for labels in label_chunks
    ]
    return chunks, batch


def extract_chunk_features(
    args: argparse.Namespace,
    model,
    tokenizer,
    sampling_args: Mapping[str, object],
    objective: MMD,
    chunk: RolloutChunk,
    *,
    compiled_extractors: bool = True,
    feature_intervention: FrozenJointSIMIntervention | None = None,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    noise, labels = chunk
    with torch.autocast(
        device_type=device.type,
        dtype=args.amp_dtype,
        enabled=args.enable_amp,
    ):
        sampled = model.sample_images_with_grad(
            noise, labels, sampling_args=dict(sampling_args)
        )
        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        images = sampled * 0.5 + 0.5
        features = objective.extract_features(images, compiled=compiled_extractors)
    if feature_intervention is not None:
        features = feature_intervention.transform(features, noise, labels)
    return features


def backward_rollout(
    args: argparse.Namespace,
    model,
    tokenizer,
    sampling_args: Mapping[str, object],
    objective: MMD,
    full_class_attraction: FullClassAttraction,
    chunks: Sequence[RolloutChunk],
    *,
    replay: bool,
    synchronize: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backpropagate one complete rollout, directly or with feature replay."""
    if not chunks:
        raise ValueError("an MMD rollout must contain at least one chunk")

    def encode(chunk: RolloutChunk) -> dict[str, torch.Tensor]:
        return extract_chunk_features(
            args,
            model,
            tokenizer,
            sampling_args,
            objective,
            chunk,
        )

    rollout_class_ids = torch.cat([chunk[1] for chunk in chunks], dim=0)

    def score(
        generated_features: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selected_real = None
        if not objective.local_generated_scoring:
            selected_real = {
                name: full_class_attraction.anchors[name].index_select(
                    0, rollout_class_ids
                )
                for name in generated_features
            }
        return objective.score_features(
            generated_features,
            selected_real,
            generated_labels=rollout_class_ids,
            attraction_features=full_class_attraction.features,
            attraction_labels=full_class_attraction.labels,
        )

    if replay:
        loss, raw_mmd = gradcache_backward(
            chunks,
            encode,
            score,
        )
    else:
        feature_chunks = [encode(chunk) for chunk in chunks]
        names = tuple(feature_chunks[0])
        if any(set(rows) != set(names) for rows in feature_chunks[1:]):
            raise RuntimeError("MMD feature names changed between rollout chunks")
        complete_rows = {
            name: torch.cat([rows[name] for rows in feature_chunks], dim=0)
            for name in names
        }
        loss, raw_mmd = score(complete_rows)
        loss.backward()

    if synchronize:
        # Legacy scoring replicates the global loss but backpropagates only the
        # rank-local generated rows. Optimized scoring computes a globally
        # normalized local contribution, scalar-SUMs it across ranks with a
        # local-only backward, and likewise backpropagates only local rows. In
        # both cases, this final SUM combines disjoint local model-gradient
        # contributions into the exact global-rollout gradient.
        sync_gradients(model.parameters(), reduction="sum")
    return loss, raw_mmd


def _parameter_gradients(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    """Return one branch VJP without accumulating into ``parameter.grad``."""
    if not loss.requires_grad:
        return (None,) * len(parameters)
    return tuple(
        None if gradient is None else gradient.detach()
        for gradient in torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
        )
    )


@torch.no_grad()
def _sum_parameter_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    """Match the training step's cross-rank SUM for diagnostic VJPs.

    Each rank differentiates only its globally normalized local class
    contribution.  The optimizer sees the SUM of those contributions, so a
    mathematically meaningful branch norm must use that same synchronized
    vector rather than the norm from one rank.  Missing local gradients become
    zeros so every rank executes an identical collective sequence.
    """
    if len(gradients) != len(parameters):
        raise ValueError("gradient and parameter collections must have equal length")
    if not is_enabled():
        return tuple(gradients)

    synchronized: list[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters):
        value = (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient.detach().contiguous()
        )
        torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
        synchronized.append(value)
    return tuple(synchronized)


def _gradient_norm(
    gradients: Sequence[torch.Tensor | None],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    total = like.new_zeros((), dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            total = total + gradient.float().square().sum()
    return total.sqrt()


def _combined_gradient_norm(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if len(left) != len(right):
        raise ValueError("gradient collections must have the same length")
    total = like.new_zeros((), dtype=torch.float32)
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is None:
            combined = right_gradient
        elif right_gradient is None:
            combined = left_gradient
        else:
            combined = left_gradient.float() + right_gradient.float()
        if combined is not None:
            total = total + combined.float().square().sum()
    return total.sqrt()


def _gradient_dot(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if len(left) != len(right):
        raise ValueError("gradient collections must have the same length")
    total = like.new_zeros((), dtype=torch.float32)
    for left_gradient, right_gradient in zip(left, right):
        if left_gradient is not None and right_gradient is not None:
            total = total + (
                left_gradient.float() * right_gradient.float()
            ).sum()
    return total


def _ot_gradient_metrics(
    *,
    attraction_gradients: Sequence[torch.Tensor | None],
    repulsion_gradients: Sequence[torch.Tensor | None],
    ot_gradients: Sequence[torch.Tensor | None],
    like: torch.Tensor,
    lambda_ot: float,
    baseline_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Summarize independent branch VJPs using the optimizer's convention."""
    attr_norm = _gradient_norm(attraction_gradients, like=like)
    repel_norm = _gradient_norm(repulsion_gradients, like=like)
    baseline_norm = _combined_gradient_norm(
        attraction_gradients,
        repulsion_gradients,
        like=like,
    )
    ot_norm = _gradient_norm(ot_gradients, like=like)
    scaled_baseline_norm = abs(float(baseline_weight)) * baseline_norm
    scaled_ot_norm = abs(float(lambda_ot)) * ot_norm
    baseline_gradients = tuple(
        right if left is None else left if right is None else left.float() + right.float()
        for left, right in zip(attraction_gradients, repulsion_gradients)
    )
    baseline_ot_dot = _gradient_dot(
        baseline_gradients, ot_gradients, like=like
    )
    attraction_ot_dot = _gradient_dot(
        attraction_gradients, ot_gradients, like=like
    )
    repulsion_ot_dot = _gradient_dot(
        repulsion_gradients, ot_gradients, like=like
    )
    combined_gradients = tuple(
        (
            None
            if baseline is None and ot is None
            else float(baseline_weight) * (
                torch.zeros_like(ot) if baseline is None else baseline
            )
            + float(lambda_ot) * (
                torch.zeros_like(baseline) if ot is None else ot
            )
        )
        for baseline, ot in zip(baseline_gradients, ot_gradients)
    )
    combined_norm = _gradient_norm(combined_gradients, like=like)
    return {
        "attraction_grad_norm": attr_norm.detach(),
        "repulsion_grad_norm": repel_norm.detach(),
        "baseline_grad_norm": baseline_norm.detach(),
        "baseline_weight": baseline_norm.new_tensor(float(baseline_weight)),
        "scaled_baseline_grad_norm": scaled_baseline_norm.detach(),
        "ot_unscaled_grad_norm": ot_norm.detach(),
        "ot_scaled_grad_norm": scaled_ot_norm.detach(),
        "ot_to_baseline_grad_norm_ratio": (
            scaled_ot_norm / scaled_baseline_norm.clamp_min(1e-20)
        ).detach(),
        "ot_baseline_grad_cosine": (
            baseline_ot_dot / (baseline_norm * ot_norm).clamp_min(1e-20)
        ).detach(),
        "ot_attraction_grad_cosine": (
            attraction_ot_dot / (attr_norm * ot_norm).clamp_min(1e-20)
        ).detach(),
        "ot_repulsion_grad_cosine": (
            repulsion_ot_dot / (repel_norm * ot_norm).clamp_min(1e-20)
        ).detach(),
        "combined_grad_norm": combined_norm.detach(),
        "gradient_cancellation_ratio": (
            combined_norm
            / (scaled_baseline_norm + scaled_ot_norm).clamp_min(1e-20)
        ).detach(),
    }


def backward_ot_rollout(
    args: argparse.Namespace,
    model,
    tokenizer,
    sampling_args: Mapping[str, object],
    objective: MMD,
    full_class_attraction: FullClassAttraction,
    ot_objective: BalancedSinkhornOT,
    ot_self_objective: BalancedSinkhornOT | None,
    ot_stream: ClasswiseOTStream,
    chunks: Sequence[RolloutChunk],
    ot_batch: ClasswiseOTBatch,
    feature_intervention: FrozenJointSIMIntervention | None = None,
    *,
    replay: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Score MMD and OT on one shared generated rollout, then backpropagate."""
    if not chunks:
        raise ValueError("a Sinkhorn OT rollout must contain at least one chunk")
    if replay and bool(args.ot_log_grad_norms):
        raise ValueError(
            "OT parameter-gradient diagnostics require grad_accum=1"
        )

    rollout_class_ids = torch.cat([chunk[1] for chunk in chunks], dim=0)
    if not torch.equal(rollout_class_ids, ot_batch.generated_labels):
        raise RuntimeError("OT rollout labels changed after class sampling")
    if getattr(objective, "representation_swap", False):
        target_features = objective.target_features(ot_batch)
    else:
        target_features = {
            name: rows.index_select(0, ot_batch.target_indices.to(rows.device))
            for name, rows in full_class_attraction.features.items()
        }

    def encode(chunk: RolloutChunk) -> dict[str, torch.Tensor]:
        return extract_chunk_features(
            args,
            model,
            tokenizer,
            sampling_args,
            objective,
            chunk,
            feature_intervention=feature_intervention,
        )

    def complete_features() -> dict[str, torch.Tensor]:
        feature_chunks = [encode(chunk) for chunk in chunks]
        names = tuple(feature_chunks[0])
        if any(set(rows) != set(names) for rows in feature_chunks[1:]):
            raise RuntimeError("OT feature names changed between rollout chunks")
        return {
            name: torch.cat([rows[name] for rows in feature_chunks], dim=0)
            for name in names
        }

    negative_features = None
    if ot_self_objective is not None:
        # An independent detached q' is required for an unbiased one-sided
        # estimator of 1/2 d_theta OT(q_theta, q_theta).  Reusing q itself
        # would make the diagonal identity coupling erase the self-repulsion.
        shape = (model.in_channels, model.input_size, model.input_size)
        with torch.no_grad():
            negative_chunks = [
                (
                    torch.randn(
                        int(args.batch_size),
                        *shape,
                        device=rollout_class_ids.device,
                        generator=ot_stream.negative_noise_generator,
                    )
                    * float(args.noise_scale),
                    labels,
                )
                for labels in ot_batch.generated_labels.split(int(args.batch_size))
            ]
            negative_feature_chunks = [
                extract_chunk_features(
                    args,
                    model,
                    tokenizer,
                    sampling_args,
                    objective,
                    chunk,
                    compiled_extractors=False,
                    feature_intervention=feature_intervention,
                )
                for chunk in negative_chunks
            ]
            names = tuple(negative_feature_chunks[0])
            negative_features = {
                name: torch.cat(
                    [rows[name] for rows in negative_feature_chunks], dim=0
                ).detach()
                for name in names
            }

    def score_ot(generated_features: Mapping[str, torch.Tensor]):
        cross_result = ot_objective.score_features(
            generated_features,
            target_features,
            generated_labels=rollout_class_ids,
            target_labels=ot_batch.target_labels,
        )
        self_result = None
        ot_loss = cross_result.loss
        if ot_self_objective is not None:
            assert negative_features is not None
            self_result = ot_self_objective.score_features(
                generated_features,
                negative_features,
                generated_labels=rollout_class_ids,
                target_labels=rollout_class_ids,
            )
            # With independent q and q', symmetry makes this one-sided
            # gradient equal to d_theta [1/2 OT(q_theta, q_theta)]. OT(p,p)
            # is constant in generator parameters and need not be evaluated.
            ot_loss = cross_result.loss - self_result.loss
        return cross_result, self_result, ot_loss

    def score_mmd(generated_features: Mapping[str, torch.Tensor]):
        selected_real = None
        if not objective.local_generated_scoring:
            selected_real = {
                name: full_class_attraction.anchors[name].index_select(
                    0, rollout_class_ids
                )
                for name in generated_features
            }
        baseline_loss, raw_mmd, components = objective.score_features(
            generated_features,
            selected_real,
            generated_labels=rollout_class_ids,
            attraction_features=full_class_attraction.features,
            attraction_labels=full_class_attraction.labels,
            update_repulsion_queue=False,
            return_components=True,
        )
        return baseline_loss, raw_mmd, components

    def score(generated_features: Mapping[str, torch.Tensor]):
        baseline_weight = float(args.ot_baseline_weight)
        if baseline_weight > 0:
            baseline_loss, raw_mmd, components = score_mmd(generated_features)
            weighted_baseline = baseline_weight * baseline_loss
        else:
            # A pure OT-first stage must not quietly update the Q4 queue or pay
            # for the unused full-reference MMD kernels.  The MMD objective is
            # still the owner of the shared feature extractors and real banks.
            raw_mmd = {}
            components = {}
            weighted_baseline = next(iter(generated_features.values())).new_zeros(())
        ot_result, self_result, ot_loss = score_ot(generated_features)
        weighted_ot = float(args.lambda_ot) * ot_loss
        total_loss = weighted_baseline + weighted_ot
        # The queue is mutated only now: both differentiable loss branches and
        # their sum were formed from the pre-enqueue Q4 snapshot.
        if baseline_weight > 0:
            objective.enqueue_repulsion_queue_rows(
                generated_features, rollout_class_ids
            )
        ot_metrics = dict(ot_result.metrics)
        ot_metrics.update(
            {
                "loss_ot": ot_loss.detach(),
                "loss_cross": ot_result.loss.detach(),
                "objective_is_sinkhorn_divergence": ot_result.loss.new_tensor(
                    float(self_result is not None)
                ),
                "lambda_ot": ot_result.loss.new_tensor(float(args.lambda_ot)),
                "baseline_weight": ot_result.loss.new_tensor(baseline_weight),
                "weighted_loss_baseline": weighted_baseline.detach(),
                "weighted_loss_ot": weighted_ot.detach(),
                "loss_total": total_loss.detach(),
                "local_classes": ot_result.loss.new_tensor(
                    float(ot_objective.local_classes_per_step)
                ),
                "cost_rows": ot_result.loss.new_tensor(
                    float(ot_objective.samples_per_class)
                ),
                "cost_columns": ot_result.loss.new_tensor(
                    float(ot_objective.real_references_per_class)
                ),
                "plan_requires_grad": ot_result.loss.new_tensor(
                    float(ot_result.transport_plan.requires_grad)
                ),
            }
        )
        if self_result is not None:
            ot_metrics.update(
                {
                    "loss_self": self_result.loss.detach(),
                    **{
                        f"self_{name}": value
                        for name, value in self_result.metrics.items()
                    },
                }
            )
        return total_loss, (raw_mmd, ot_metrics, components, ot_loss)

    if replay:
        total_loss, payload = gradcache_backward(chunks, encode, score)
        raw_mmd, ot_metrics, _, _ = payload
    else:
        gradient_metrics: dict[str, torch.Tensor] = {}
        if bool(args.ot_log_grad_norms):
            # Production MMD kernels use activation checkpointing. Reusing one
            # retained graph for several VJPs changes later gradients on that
            # graph, so calibration uses one freshly replayed graph per branch.
            # Restore every observable training state before the real forward;
            # diagnostics then have exactly zero effect on the optimizer update.
            parameters = tuple(
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            queue_state = objective.running_state_dict()
            cpu_rng_state = torch.get_rng_state()
            device = next(model.parameters()).device
            cuda_rng_state = (
                torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None
            )
            buffer_state = {
                name: buffer.detach().clone()
                for name, buffer in model.named_buffers()
            }

            def restore_replay_state() -> None:
                objective.load_running_state_dict(queue_state, strict=True)
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, device)
                with torch.no_grad():
                    current_buffers = dict(model.named_buffers())
                    for name, saved in buffer_state.items():
                        current_buffers[name].copy_(saved)

            def mmd_branch_gradients(
                component_name: str,
            ) -> tuple[torch.Tensor | None, ...]:
                restore_replay_state()
                replay_rows = complete_features()
                _, _, replay_components = score_mmd(replay_rows)
                return _sum_parameter_gradients(
                    _parameter_gradients(
                        replay_components[component_name], parameters
                    ),
                    parameters,
                )

            try:
                attraction_gradients = mmd_branch_gradients("loss_attr")
                repulsion_gradients = mmd_branch_gradients("loss_repel")
                restore_replay_state()
                ot_rows = complete_features()
                replay_cross, replay_self, replay_ot_loss = score_ot(ot_rows)
                ot_gradients = _sum_parameter_gradients(
                    _parameter_gradients(replay_ot_loss, parameters),
                    parameters,
                )
                gradient_metrics = _ot_gradient_metrics(
                    attraction_gradients=attraction_gradients,
                    repulsion_gradients=repulsion_gradients,
                    ot_gradients=ot_gradients,
                    like=replay_ot_loss,
                    lambda_ot=float(args.lambda_ot),
                    baseline_weight=float(args.ot_baseline_weight),
                )
                del (
                    attraction_gradients,
                    repulsion_gradients,
                    ot_gradients,
                    replay_cross,
                    replay_self,
                    replay_ot_loss,
                    ot_rows,
                )
            finally:
                restore_replay_state()

        if getattr(objective, "representation_swap", False):
            if len(chunks) != 1 or tokenizer is not None:
                raise ValueError("Pixel representation replay expects one full OST rollout")
            noise, labels = chunks[0]
            with torch.autocast(device_type=noise.device.type, dtype=args.amp_dtype, enabled=args.enable_amp):
                sampled = model.sample_images_with_grad(noise, labels, sampling_args=dict(sampling_args))
                images = sampled * 0.5 + 0.5
            with torch.no_grad():
                complete_rows = objective.extract_features(images, compiled=False)
            complete_rows = {k:v.detach().requires_grad_(True) for k,v in complete_rows.items()}
            total_loss, payload = score(complete_rows)
            raw_mmd, ot_metrics, _, _ = payload
            cotangents = torch.autograd.grad(total_loss, tuple(complete_rows.values()))
            feature_gradients = dict(zip(complete_rows, cotangents))
            del complete_rows, cotangents
            image_gradients = objective.input_vjp(images.detach(), feature_gradients)
            del feature_gradients
            images.backward(image_gradients)
        else:
            complete_rows = complete_features()
            total_loss, payload = score(complete_rows)
            raw_mmd, ot_metrics, _, _ = payload
            total_loss.backward()
        ot_metrics.update(gradient_metrics)

    # MMD and OT scalar SUMs retain only rank-local backward paths. The one
    # existing final parameter-gradient SUM yields the exact global-class mean.
    sync_gradients(model.parameters(), reduction="sum")
    return total_loss, raw_mmd, ot_metrics


def _backward_with_branch_diagnostics(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Backpropagate one branch and measure its contribution before accumulation."""
    mmd_grad_norm = get_grad_norm(parameters).float()
    xm_squared: list[torch.Tensor] = []
    cross_products: list[torch.Tensor] = []
    handles = []

    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        reference = None if parameter.grad is None else parameter.grad.detach()

        def capture(
            gradient: torch.Tensor,
            reference_gradient: torch.Tensor | None = reference,
        ) -> torch.Tensor:
            detached = gradient.detach().float()
            xm_squared.append(detached.square().sum())
            if reference_gradient is not None:
                cross_products.append(
                    (detached * reference_gradient.detach().float()).sum()
                )
            return gradient

        handles.append(parameter.register_hook(capture))

    try:
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    if not xm_squared:
        raise RuntimeError("Forward-XM backward produced no parameter gradients")

    xm_grad_norm = torch.stack(xm_squared).sum().sqrt()
    dot = (
        torch.stack(cross_products).sum()
        if cross_products
        else xm_grad_norm.new_zeros(())
    )
    denominator = (mmd_grad_norm * xm_grad_norm).clamp_min(1e-20)
    return {
        "mmd_local_grad_norm_before_xm": mmd_grad_norm.detach(),
        "xm_weighted_local_grad_norm": xm_grad_norm.detach(),
        "xm_to_mmd_local_grad_ratio": (
            xm_grad_norm / mmd_grad_norm.clamp_min(1e-20)
        ).detach(),
        "mmd_xm_local_grad_cosine": (dot / denominator).detach(),
    }


def backward_xm(
    args: argparse.Namespace,
    model,
    tokenizer,
    sampling_args: Mapping[str, object],
    feature_extractor: MMD,
    full_class_attraction: FullClassAttraction,
    xm_objective: SharedForwardXM,
    xm_stream: ClasswiseXMStream,
    *,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backpropagate one auxiliary shared-pool Forward-XM rollout."""
    shape = (model.in_channels, model.input_size, model.input_size)
    batch = xm_stream.next_batch(
        noise_shape=shape,
        noise_scale=args.noise_scale,
    )
    generated_features = extract_chunk_features(
        args,
        model,
        tokenizer,
        sampling_args,
        feature_extractor,
        (batch.noise, batch.candidate_labels),
        # The production MMD batch retains its compiled extractor path.  The
        # smaller auxiliary shape runs eagerly through the identical frozen
        # extractors: PyTorch 2.6 Inductor currently fails while compiling the
        # antialiased bicubic backward for this second batch shape.
        compiled_extractors=False,
    )
    target_features = {
        name: rows.index_select(0, batch.target_indices.to(rows.device))
        for name, rows in full_class_attraction.features.items()
    }
    result = xm_objective.score_features(
        generated_features,
        target_features,
        generated_labels=batch.candidate_labels,
        target_labels=batch.target_labels,
    )
    weighted_loss = float(args.xm_weight) * result.loss
    metrics = dict(result.metrics)
    metrics["weight"] = result.loss.new_tensor(float(args.xm_weight))
    metrics["weighted_loss"] = weighted_loss.detach()
    if bool(args.xm_log_branch_grad_norms):
        metrics.update(
            _backward_with_branch_diagnostics(weighted_loss, parameters)
        )
    else:
        weighted_loss.backward()
    return result.loss.detach(), metrics


def train_step(
    args: argparse.Namespace,
    model,
    tokenizer,
    sampling_args: Mapping[str, object],
    objective: MMD,
    full_class_attraction: FullClassAttraction,
    xm_objective: SharedForwardXM | None = None,
    xm_stream: ClasswiseXMStream | None = None,
    ot_objective: BalancedSinkhornOT | None = None,
    ot_self_objective: BalancedSinkhornOT | None = None,
    ot_stream: ClasswiseOTStream | None = None,
    feature_intervention: FrozenJointSIMIntervention | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    if (xm_objective is None) != (xm_stream is None):
        raise ValueError("xm_objective and xm_stream must be configured together")
    if (ot_objective is None) != (ot_stream is None):
        raise ValueError("ot_objective and ot_stream must be configured together")
    if ot_self_objective is not None and ot_objective is None:
        raise ValueError("an OT self objective requires the cross objective")
    if xm_objective is not None and ot_objective is not None:
        raise ValueError("Forward-XM and Sinkhorn OT cannot be enabled together")
    if ot_objective is not None:
        assert ot_stream is not None
        chunks, ot_batch = sample_ot_rollout_chunks(args, model, ot_stream)
        return backward_ot_rollout(
            args,
            model,
            tokenizer,
            sampling_args,
            objective,
            full_class_attraction,
            ot_objective,
            ot_self_objective,
            ot_stream,
            chunks,
            ot_batch,
            feature_intervention,
            replay=len(chunks) > 1,
        )

    chunks = sample_rollout_chunks(args, model)
    loss, raw_mmd = backward_rollout(
        args,
        model,
        tokenizer,
        sampling_args,
        objective,
        full_class_attraction,
        chunks,
        replay=len(chunks) > 1,
        synchronize=xm_objective is None,
    )
    if xm_objective is None:
        return loss, raw_mmd, {}

    parameters = tuple(model.parameters())
    xm_loss, xm_metrics = backward_xm(
        args,
        model,
        tokenizer,
        sampling_args,
        objective,
        full_class_attraction,
        xm_objective,
        xm_stream,
        parameters=parameters,
    )
    sync_gradients(parameters, reduction="sum")
    total_loss = loss.detach() + float(args.xm_weight) * xm_loss
    return total_loss, raw_mmd, xm_metrics


def compile_training_modules(args, model, objective: MMD) -> None:
    if not args.compile:
        return
    logger.info("Compiling OST backbone and DistMatch feature extractors")
    model.net.compile(fullgraph=True)
    objective.compile(fullgraph=True)


def measure_training_flops(
    args,
    model,
    tokenizer,
    objective: MMD,
    full_class_attraction: FullClassAttraction,
    ot_objective: BalancedSinkhornOT | None = None,
    ot_self_objective: BalancedSinkhornOT | None = None,
    ot_stream: ClasswiseOTStream | None = None,
    feature_intervention: FrozenJointSIMIntervention | None = None,
) -> None:
    """Measure one eager MMD backward step without changing model state."""
    if args.current_step != 0:
        logger.info("Skipping FLOPs measurement when resuming training")
        return

    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    buffer_states = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    objective_state = objective.running_state_dict()
    ot_stream_state = None if ot_stream is None else ot_stream.state_dict()
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    logger.info("Measuring one eager MMD training step's forward/backward FLOPs")
    try:
        with FlopCounterMode(display=False) as counter:
            train_step(
                args,
                model,
                tokenizer,
                sampling_args,
                objective,
                full_class_attraction,
                None,
                None,
                ot_objective,
                ot_self_objective,
                ot_stream,
                feature_intervention,
            )
        local_step_flops = counter.get_total_flops()
        grad_norm = get_grad_norm(model.parameters())
        if not torch.isfinite(grad_norm) or grad_norm.item() <= 0:
            raise RuntimeError(
                f"eager FLOP measurement produced invalid grad_norm={grad_norm.item()}"
            )
        logger.info("Eager FLOP measurement grad_norm=%.6f", grad_norm.item())
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        with torch.no_grad():
            current_buffers = dict(model.named_buffers())
            for name, saved_buffer in buffer_states.items():
                current_buffers[name].copy_(saved_buffer)
        objective.load_running_state_dict(objective_state, strict=True)
        if ot_stream is not None:
            assert ot_stream_state is not None
            ot_stream.load_state_dict(ot_stream_state)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    global_step_flops = torch.tensor(local_step_flops, dtype=torch.int64, device="cuda")
    if is_enabled():
        torch_dist.all_reduce(global_step_flops, op=torch_dist.ReduceOp.SUM)
    global_step_flops = int(global_step_flops.item())

    if args.rank == 0:
        if getattr(objective, "representation_swap", False):
            Path(args.log_dir, "training_flops.json").write_text(json.dumps({
                "global_step_flops": global_step_flops,
                "planned_updates": args.total_steps,
                "post_training_eflops": global_step_flops * args.total_steps / 1e18,
                "reference_encoding": "online",
                "encoder_input_checkpointing": args.ot_encoder_checkpointing,
                "feature_gradient_replay": True,
                "per_rank_supported_flops_by_module": {
                    name: {str(op): int(count) for op, count in counts.items()}
                    for name, counts in counter.get_flop_counts().items()
                },
                "module_count_note": "Inclusive module hierarchy; do not sum parent and child counts. Global is the local total before the rank SUM.",
                "scope": "supported matmul/conv/SDPA incl. checkpoint recomputation; excludes optimizer/EMA/eval/unsupported elementwise"
            }, indent=2) + "\n")
        device_step_flops = global_step_flops / get_world_size()
        per_sample_flops = global_step_flops / args.global_bsz
        total_flops = global_step_flops * args.total_steps
        logger.info(
            "Training FLOPs estimate (supported matmul/conv/SDPA operations; "
            "optimizer, model EMA, communication, and unsupported operations excluded): "
            "device_step=%.3f TFLOPs, global_step=%.3f TFLOPs, "
            "sample=%.3f GFLOPs, planned_total=%.3f EFLOPs, "
            "planned_total=%.3f PF-days",
            device_step_flops / 1e12,
            global_step_flops / 1e12,
            per_sample_flops / 1e9,
            total_flops / 1e18,
            total_flops / (1e15 * 86400),
        )


def _checkpoint_rng_state(args, rng: RNGStateManager) -> dict[str, Any] | None:
    """Collect the rank-local RNG streams for an exact later continuation."""
    local_state = rng.snapshot()
    if is_enabled():
        rank_states = [None] * args.world_size if get_global_rank() == 0 else None
        torch_dist.gather_object(local_state, rank_states, dst=0)
    else:
        rank_states = [local_state]

    if get_global_rank() != 0:
        return None
    return {
        "version": MMD_RNG_STATE_VERSION,
        "world_size": int(args.world_size),
        "rank_states": rank_states,
    }


def _restore_checkpoint_rng_state(
    args,
    rng: RNGStateManager,
    checkpoint_extra: Mapping[str, Any] | None,
) -> bool:
    """Restore this rank's saved RNG stream after startup has finished."""
    if not checkpoint_extra or MMD_RNG_STATE_KEY not in checkpoint_extra:
        return False
    payload = checkpoint_extra[MMD_RNG_STATE_KEY]
    if not isinstance(payload, Mapping):
        raise TypeError("saved MMD RNG state must be a mapping")
    if int(payload.get("version", -1)) != MMD_RNG_STATE_VERSION:
        raise ValueError(
            f"unsupported MMD RNG state version {payload.get('version')!r}"
        )
    saved_world_size = int(payload.get("world_size", -1))
    if saved_world_size != int(args.world_size):
        raise ValueError(
            "MMD RNG resume requires the original world size: "
            f"checkpoint={saved_world_size}, current={args.world_size}"
        )
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, Sequence) or len(rank_states) != saved_world_size:
        raise ValueError("saved MMD RNG state has invalid rank_states")
    rng.load(rank_states[int(args.rank)])
    return True


def _checkpoint_xm_stream_state(
    args,
    xm_stream: ClasswiseXMStream | None,
) -> dict[str, Any] | None:
    if xm_stream is None:
        return None
    local_state = xm_stream.state_dict()
    if is_enabled():
        rank_states = [None] * args.world_size if get_global_rank() == 0 else None
        torch_dist.gather_object(local_state, rank_states, dst=0)
    else:
        rank_states = [local_state]
    if get_global_rank() != 0:
        return None
    return {
        "version": XM_STREAM_STATE_VERSION,
        "world_size": int(args.world_size),
        "rank_states": rank_states,
    }


def _restore_checkpoint_xm_stream_state(
    args,
    xm_stream: ClasswiseXMStream | None,
    checkpoint_extra: Mapping[str, Any] | None,
) -> bool:
    if xm_stream is None or not checkpoint_extra:
        return False
    if XM_STREAM_STATE_KEY not in checkpoint_extra:
        return False
    payload = checkpoint_extra[XM_STREAM_STATE_KEY]
    if not isinstance(payload, Mapping):
        raise TypeError("saved Forward-XM stream state must be a mapping")
    if int(payload.get("version", -1)) != XM_STREAM_STATE_VERSION:
        raise ValueError(
            f"unsupported Forward-XM stream version {payload.get('version')!r}"
        )
    saved_world_size = int(payload.get("world_size", -1))
    if saved_world_size != int(args.world_size):
        raise ValueError(
            "Forward-XM resume requires the original world size: "
            f"checkpoint={saved_world_size}, current={args.world_size}"
        )
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, Sequence) or len(rank_states) != saved_world_size:
        raise ValueError("saved Forward-XM stream has invalid rank_states")
    xm_stream.load_state_dict(rank_states[int(args.rank)])
    return True


def _checkpoint_ot_stream_state(
    args,
    ot_stream: ClasswiseOTStream | None,
) -> dict[str, Any] | None:
    if ot_stream is None:
        return None
    local_state = ot_stream.state_dict()
    if is_enabled():
        rank_states = [None] * args.world_size if get_global_rank() == 0 else None
        torch_dist.gather_object(local_state, rank_states, dst=0)
    else:
        rank_states = [local_state]
    if get_global_rank() != 0:
        return None
    return {
        "version": OT_STREAM_STATE_VERSION,
        "world_size": int(args.world_size),
        "rank_states": rank_states,
    }


def _restore_checkpoint_ot_stream_state(
    args,
    ot_stream: ClasswiseOTStream | None,
    checkpoint_extra: Mapping[str, Any] | None,
    *,
    require_negative_noise_state: bool = False,
) -> bool:
    if ot_stream is None or not checkpoint_extra:
        return False
    if OT_STREAM_STATE_KEY not in checkpoint_extra:
        return False
    payload = checkpoint_extra[OT_STREAM_STATE_KEY]
    if not isinstance(payload, Mapping):
        raise TypeError("saved Sinkhorn OT stream state must be a mapping")
    payload_version = int(payload.get("version", -1))
    if payload_version not in {1, OT_STREAM_STATE_VERSION}:
        raise ValueError(
            f"unsupported Sinkhorn OT stream version {payload.get('version')!r}"
        )
    saved_world_size = int(payload.get("world_size", -1))
    if saved_world_size != int(args.world_size):
        raise ValueError(
            "Sinkhorn OT resume requires the original world size: "
            f"checkpoint={saved_world_size}, current={args.world_size}"
        )
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, Sequence) or len(rank_states) != saved_world_size:
        raise ValueError("saved Sinkhorn OT stream has invalid rank_states")
    class_rng_states = [
        state.get("class_generator_state")
        if isinstance(state, Mapping)
        else None
        for state in rank_states
    ]
    if not all(isinstance(state, torch.Tensor) for state in class_rng_states):
        raise TypeError("saved Sinkhorn OT class RNG states are invalid")
    if any(
        not torch.equal(state, class_rng_states[0])
        for state in class_rng_states[1:]
    ):
        raise ValueError("saved Sinkhorn OT global class RNG diverged across ranks")
    if require_negative_noise_state and any(
        not isinstance(state, Mapping)
        or int(state.get("version", -1)) < 2
        or not isinstance(state.get("negative_noise_generator_state"), torch.Tensor)
        for state in rank_states
    ):
        raise ValueError(
            "resumed Sinkhorn-divergence checkpoint predates the private q' "
            "noise stream and cannot be resumed exactly"
        )
    ot_stream.load_state_dict(rank_states[int(args.rank)])
    return True


def _save(
    args,
    step,
    model,
    optimizer,
    ema_model,
    objective,
    started,
    rng,
    xm_stream: ClasswiseXMStream | None = None,
    ot_stream: ClasswiseOTStream | None = None,
    feature_intervention: FrozenJointSIMIntervention | None = None,
) -> None:
    elapsed = time.time() - started + args.last_elapsed_time
    rng_state = _checkpoint_rng_state(args, rng)
    xm_state = _checkpoint_xm_stream_state(args, xm_stream)
    ot_state = _checkpoint_ot_stream_state(args, ot_stream)
    extra = None
    if (objective.repulsion_queue_size > 0 or getattr(objective, "representation_swap", False)) and get_global_rank() == 0:
        extra = {
            MMD_QUEUE_STATE_KEY: objective.running_state_dict(),
            MMD_RNG_STATE_KEY: rng_state,
        }
        if xm_state is not None:
            extra[XM_STREAM_STATE_KEY] = xm_state
        if ot_state is not None:
            extra[OT_STREAM_STATE_KEY] = ot_state
        if feature_intervention is not None:
            extra[FEATURE_INTERVENTION_STATE_KEY] = (
                feature_intervention.checkpoint_state()
            )
    save_checkpoint(
        args,
        step,
        model,
        optimizer,
        ema_model,
        elapsed,
        extra=extra,
    )
    if is_enabled():
        torch_dist.barrier()


def run_training(
    args,
    model,
    ema_model,
    optimizer,
    tokenizer,
    objective: MMD,
    full_class_attraction: FullClassAttraction,
    xm_objective: SharedForwardXM | None,
    xm_stream: ClasswiseXMStream | None,
    ot_objective: BalancedSinkhornOT | None,
    ot_self_objective: BalancedSinkhornOT | None,
    ot_stream: ClasswiseOTStream | None,
    feature_intervention: FrozenJointSIMIntervention | None,
    *,
    wandb_logger,
    rng,
) -> bool:
    if args.current_step > args.total_steps:
        raise ValueError(
            f"checkpoint step {args.current_step} exceeds training horizon "
            f"{args.total_steps}"
        )
    if args.current_step == args.total_steps:
        logger.info(
            "Training is already at the exclusive horizon step %s; "
            "skipping optimizer updates",
            args.total_steps,
        )
        return True

    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    started = time.time()
    step_start = time.perf_counter()
    last_saved_step = args.current_step
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        f"training from step {args.current_step:,} -> {args.total_steps:,} "
        f"({args.start_epoch} -> {args.epochs} epochs)"
    )

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    def _infinite():
        while True:
            yield None

    for step, _ in metric_logger.log_every(
        _infinite(),
        args.print_freq,
        header="Train:",
        start_iteration=args.current_step,
        n_iterations=args.total_steps,
    ):
        model.train()
        adjust_learning_rate(optimizer, step, args)
        loss, raw_mmd, branch_metrics = train_step(
            args,
            model,
            tokenizer,
            sampling_args,
            objective,
            full_class_attraction,
            xm_objective,
            xm_stream,
            ot_objective,
            ot_self_objective,
            ot_stream,
            feature_intervention,
        )
        xm_metrics = branch_metrics if xm_objective is not None else {}
        ot_metrics = branch_metrics if ot_objective is not None else {}

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.grad_clip > 0
            else get_grad_norm(model.parameters())
        )
        audit_update = step < 3 or (step + 1) % 1250 == 0
        before_update = [p.detach().clone() for p in model.parameters() if p.requires_grad] if audit_update else None
        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning("Non-finite gradients at step %s; update skipped", step)
            if getattr(objective, "representation_swap", False):
                raise RuntimeError(f"Non-finite generator gradient at update {step}; stop for diagnosis")
        update_norm = None
        if before_update is not None:
            update_norm = float(sum((p.detach().float() - old.float()).square().sum()
                for p, old in zip((p for p in model.parameters() if p.requires_grad), before_update)).sqrt())
            del before_update
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += args.rollout_size

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        loss_value = all_reduce_mean(loss.item())
        mmd_values = {
            name: all_reduce_mean(float(value.detach()))
            for name, value in raw_mmd.items()
        }
        xm_values = {
            name: all_reduce_mean(float(value.detach()))
            for name, value in xm_metrics.items()
        }
        ot_values = {
            name: all_reduce_mean(float(value.detach()))
            for name, value in ot_metrics.items()
        }
        samples_per_device = (
            args.local_rollout_size / step_time if step_time > 0 else 0.0
        )
        memory_gb = (
            torch.cuda.max_memory_reserved() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        queue_stats = objective.repulsion_queue_fill_stats
        if getattr(objective, "representation_swap", False) and get_global_rank() == 0:
            # MetricLogger stores rolling medians; preserve exact per-update
            # numerical diagnostics separately for this representation audit.
            record = {
                "updates": args.current_step,
                "seconds": step_time,
                "peak_reserved_gib": memory_gb,
                "loss": loss_value,
                "gradient_norm": float(grad_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "parameter_update_norm": update_norm,
                "discrepancy_kind": getattr(args, "discrepancy_kind", "joint_ot"),
                **ot_values,
            }
            with open(os.path.join(args.log_dir, "ot_diagnostics.jsonl"), "a") as handle:
                handle.write(json.dumps(record) + "\n")
        canonical_losses = {"loss_total": loss_value}
        if "loss_attr" in mmd_values:
            canonical_losses["loss_attr"] = mmd_values["loss_attr"]
        if "loss_repel" in mmd_values:
            canonical_losses["loss_repel"] = mmd_values["loss_repel"]
        if "loss_ot" in ot_values:
            canonical_losses.update(
                {
                    "loss_ot": ot_values["loss_ot"],
                    "lambda_ot": ot_values["lambda_ot"],
                }
            )
        metric_logger.update(
            loss=loss_value,
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{
                "samples/s/device": samples_per_device,
                "samples/s": samples_per_device * args.world_size,
                "samples_seen(M)": args.samples_seen / 1e6,
                "device_mem(GB)": memory_gb,
                "repulsion_queue_rows": queue_stats["total_rows"],
                "repulsion_queue_classes": queue_stats["populated_classes"],
                "repulsion_queue_min": queue_stats[
                    "min_rows_per_populated_class"
                ],
                "repulsion_queue_max": queue_stats["max_rows_per_class"],
                **canonical_losses,
                **{f"mmd_{name}": value for name, value in mmd_values.items()},
                **{f"xm_{name}": value for name, value in xm_values.items()},
                **{f"ot_{name}": value for name, value in ot_values.items()},
            },
        )

        if step % args.print_freq == 0:
            if wandb_logger:
                wandb_logger.update(
                    {
                        "train/loss": loss_value,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        **{
                            f"train/{name}": value
                            for name, value in canonical_losses.items()
                        },
                        **{
                            f"train/repulsion_queue_{name}": value
                            for name, value in queue_stats.items()
                        },
                        **{
                            f"train/mmd_{name}": value
                            for name, value in mmd_values.items()
                        },
                        **{
                            f"train/xm_{name}": value
                            for name, value in xm_values.items()
                        },
                        **{
                            f"train/ot_{name}": value
                            for name, value in ot_values.items()
                        },
                    },
                    step=args.current_step,
                )
            if ot_values:
                marginal_error = max(
                    ot_values["row_marginal_max_error"],
                    ot_values["column_marginal_max_error"],
                )
                warning_threshold = max(
                    1e-5, 10.0 * float(args.ot_tolerance)
                )
                if marginal_error > warning_threshold:
                    logger.warning(
                        "Sinkhorn marginal error %.3e exceeds %.3e at step %s",
                        marginal_error,
                        warning_threshold,
                        step,
                    )

        regular_save = (
            args.save_every > 0
            and args.current_step % args.save_every == 0
        )
        milestone_save = (
            args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0
        )
        if regular_save or milestone_save or args.current_step == args.total_steps:
            _save(
                args,
                step,
                model,
                optimizer,
                ema_model,
                objective,
                started,
                rng,
                xm_stream,
                ot_stream,
                feature_intervention,
            )
            last_saved_step = args.current_step

        if (
            args.online_eval
            and regular_save
            and args.current_step < args.total_steps
        ):
            rng_state = rng.snapshot()
            try:
                run_fdr6_evaluation(
                    args,
                    model,
                    ema_model,
                    tokenizer,
                    wandb_logger=wandb_logger,
                    feature_intervention=feature_intervention,
                )
            finally:
                rng.load(rng_state)
                gc.collect()
                step_start = time.perf_counter()

        stop_requested = preempt_requested() or os.path.isfile(os.path.join(args.log_dir, "STOP_REQUESTED"))
        if getattr(objective, "representation_swap", False) and is_enabled():
            stop_flag = torch.tensor(int(stop_requested), device="cuda")
            torch_dist.all_reduce(stop_flag, op=torch_dist.ReduceOp.MAX)
            stop_requested = bool(stop_flag.item())
        if stop_requested:
            logger.info("Preemption/operator stop requested at step %s", args.current_step)
            _save(
                args,
                step,
                model,
                optimizer,
                ema_model,
                objective,
                started,
                rng,
                xm_stream,
                ot_stream,
                feature_intervention,
            )
            return False

        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(
                args,
                model,
                ema_model,
                args.current_step,
                rng=rng,
                tokenizer=tokenizer,
            )

    elapsed = time.time() - started + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info("averaged stats: %s", metric_logger)
    logger.info(
        "Training complete. Total time: %s on %s devices",
        datetime.timedelta(seconds=int(elapsed)),
        args.world_size,
    )
    return True


def _append_intervened_eval_csv(
    args,
    ema_label,
    cfg,
    result,
    fdr,
    fdr_sim3,
    *,
    feature_intervention: FrozenJointSIMIntervention,
    mean_perturbation_energy: float,
    mean_expected_energy: float,
) -> None:
    path = Path(args.log_dir) / "intervened_eval_summary.csv"
    fields = (
        "distribution",
        "model",
        "fd",
        "fdr",
        "fdr_sim3",
        "n",
        "ema_label",
        "cfg",
        "step",
        "rank",
        "strength",
        "artifact_id",
        "mean_perturbation_energy",
        "mean_expected_energy",
        "interval_min",
        "interval_max",
        "num_sampling_steps",
        "timestamp",
    )
    write_header = not path.exists()
    if not write_header:
        with path.open(newline="") as handle:
            existing = tuple(csv.DictReader(handle).fieldnames or ())
        if existing != fields:
            raise RuntimeError(
                f"intervened evaluation CSV schema differs: {existing} != {fields}"
            )
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for name in SIM3_EVAL_NAMES:
            writer.writerow(
                {
                    "distribution": "joint_sim_intervened",
                    "model": name,
                    "fd": result.fd[name],
                    "fdr": fdr[name],
                    "fdr_sim3": fdr_sim3,
                    "n": result.num_samples,
                    "ema_label": ema_label,
                    "cfg": cfg,
                    "step": args.current_step,
                    "rank": feature_intervention.rank,
                    "strength": feature_intervention.strength,
                    "artifact_id": feature_intervention.artifact_id,
                    "mean_perturbation_energy": mean_perturbation_energy,
                    "mean_expected_energy": mean_expected_energy,
                    "interval_min": args.interval_min,
                    "interval_max": args.interval_max,
                    "num_sampling_steps": args.num_sampling_steps,
                    "timestamp": datetime.datetime.now().isoformat(timespec="minutes"),
                }
            )


@torch.inference_mode()
def run_fdr6_evaluation(
    args,
    model,
    ema_model,
    tokenizer,
    *,
    wandb_logger,
    feature_intervention: FrozenJointSIMIntervention | None = None,
):
    device = next(model.parameters()).device
    features = _features(FDR6_SPECS)
    stats = _cached_stats(FDR6_SPECS, device=device)
    evaluator = FrechetEvaluator(
        features=features,
        stats=stats,
        stats_dtype=torch.float64,
        distributed="auto",
        device=device,
    )
    inception_model, inception_reference = _prepare_inception_evaluation(
        args, device=device
    )

    world_size = get_world_size()
    rank = get_global_rank()
    cfg_values = sorted(args.cfg_list or [args.cfg])
    ema_labels = args.eval_ema_labels or ["online"]
    rows = []

    for ema_label in ema_labels:
        for cfg in cfg_values:
            start, end = get_start_end_indices(args.num_images, world_size, rank)
            classes = _prepare_eval_classes(args, args.num_images, start, end)
            evaluator.reset()
            intervention_enabled = (
                feature_intervention is not None
                and bool(getattr(args, "feature_intervention_evaluate", False))
            )
            intervened_moments = (
                evaluator.new_moments() if intervention_enabled else None
            )
            perturbation_energy_sum = torch.zeros((), dtype=torch.float64, device=device)
            expected_energy_sum = torch.zeros((), dtype=torch.float64, device=device)
            inception_feature_batches: list[torch.Tensor] = []
            inception_logit_batches: list[torch.Tensor] = []

            generated = 0
            model.eval()
            with ema_model.swap(model, label=ema_label):
                while generated < end - start:
                    batch_size = min(args.eval_bsz, end - start - generated)
                    labels = (
                        torch.from_numpy(classes[generated : generated + batch_size])
                        .long()
                        .to(device)
                    )
                    latents = (
                        sample_generation_latents(
                            args, model, batch_size, device=device
                        )
                        if intervention_enabled
                        else None
                    )
                    images = generate_images(
                        args,
                        model,
                        labels=labels,
                        cfg=float(cfg),
                        tokenizer=tokenizer,
                        z_t=latents,
                    )
                    _accumulate_inception_batch(
                        inception_model,
                        images,
                        inception_feature_batches,
                        inception_logit_batches,
                    )
                    # Preserve the BF16 DistMatch feature-forward path for FDr-6.
                    # The independent native Inception path above stays in FP32.
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        if intervention_enabled:
                            outputs = evaluator.extract_features(images)
                        else:
                            evaluator.update(images)
                    if intervention_enabled:
                        assert feature_intervention is not None
                        assert intervened_moments is not None
                        assert latents is not None
                        evaluator.update_features(outputs)
                        joint_perturbation = feature_intervention.perturbation(
                            latents, labels
                        )
                        transformed = feature_intervention.add_to_features(
                            outputs,
                            joint_perturbation,
                            feature_keys=INTERVENED_EVAL_FEATURE_KEYS,
                        )
                        evaluator.update_features(
                            transformed, moments=intervened_moments
                        )
                        perturbation_energy_sum += (
                            joint_perturbation.double().square().sum(-1).sum()
                        )
                        expected_energy_sum += (
                            feature_intervention.strength**2
                            * feature_intervention.scales.index_select(
                                0, labels
                            ).double().square().sum()
                        )
                    generated += batch_size

            result = evaluator.compute()
            intervened_result = (
                evaluator.compute_moments(intervened_moments)
                if intervened_moments is not None
                else None
            )
            inception_metrics = run_inception_evaluation(
                args,
                inception_feature_batches,
                inception_logit_batches,
                inception_reference,
                device=device,
            )
            if result.num_samples != args.num_images:
                raise RuntimeError(
                    f"FDr-6 expected {args.num_images} images, got {result.num_samples}"
                )
            if (
                intervened_result is not None
                and intervened_result.num_samples != result.num_samples
            ):
                raise RuntimeError(
                    "raw and intervened FDr sample counts differ: "
                    f"{result.num_samples} != {intervened_result.num_samples}"
                )
            if inception_metrics["num_samples"] != result.num_samples:
                raise RuntimeError(
                    "FDr-6 and Inception evaluation sample counts differ: "
                    f"{result.num_samples} != {inception_metrics['num_samples']}"
                )
            fdr = {
                name: result.fd[name] / baseline
                for name, baseline in FDR6_VALIDATION_FD.items()
            }
            fdr6 = sum(fdr.values()) / len(fdr)
            intervened_fdr = None
            intervened_fdr_sim3 = None
            mean_perturbation_energy = None
            mean_expected_energy = None
            if intervened_result is not None:
                intervened_fdr = {
                    name: intervened_result.fd[name] / FDR6_VALIDATION_FD[name]
                    for name in SIM3_EVAL_NAMES
                }
                intervened_fdr_sim3 = sum(intervened_fdr.values()) / len(
                    intervened_fdr
                )
                energy_totals = torch.stack(
                    [perturbation_energy_sum, expected_energy_sum]
                )
                if is_enabled():
                    torch_dist.all_reduce(energy_totals, op=torch_dist.ReduceOp.SUM)
                mean_perturbation_energy = float(
                    energy_totals[0].item() / result.num_samples
                )
                mean_expected_energy = float(
                    energy_totals[1].item() / result.num_samples
                )
            row = {
                "ema_label": ema_label,
                "cfg": float(cfg),
                "fd": result.fd,
                "fdr": fdr,
                "fdr6": fdr6,
                "is": inception_metrics["is"],
                "precision": inception_metrics["precision"],
                "recall": inception_metrics["recall"],
                "num_samples": result.num_samples,
                "intervened_fd": (
                    None if intervened_result is None else intervened_result.fd
                ),
                "intervened_fdr_sim3": intervened_fdr_sim3,
                "mean_perturbation_energy": mean_perturbation_energy,
                "mean_expected_energy": mean_expected_energy,
            }
            rows.append(row)

            logger.info("ema=%s cfg=%.2f FDr-6=%.6f", ema_label, cfg, fdr6)
            if intervened_result is not None:
                logger.info(
                    "ema=%s cfg=%.2f intervened_FID=%.6f intervened_FDr-SIM3=%.6f "
                    "mean_b_energy=%.6f expected=%.6f",
                    ema_label,
                    cfg,
                    intervened_result.fd["FID(ADM)"],
                    intervened_fdr_sim3,
                    mean_perturbation_energy,
                    mean_expected_energy,
                )
            if rank == 0:
                _append_eval_csv(
                    args,
                    ema_label,
                    cfg,
                    result,
                    fdr,
                    fdr6,
                    inception_metrics,
                )
                if intervened_result is not None:
                    assert feature_intervention is not None
                    assert intervened_fdr is not None
                    assert intervened_fdr_sim3 is not None
                    assert mean_perturbation_energy is not None
                    assert mean_expected_energy is not None
                    _append_intervened_eval_csv(
                        args,
                        ema_label,
                        cfg,
                        intervened_result,
                        intervened_fdr,
                        intervened_fdr_sim3,
                        feature_intervention=feature_intervention,
                        mean_perturbation_energy=mean_perturbation_energy,
                        mean_expected_energy=mean_expected_energy,
                    )
                if wandb_logger:
                    wandb_logger.update(
                        {
                            "final_eval/fdr6": fdr6,
                            "final_eval/cfg": cfg,
                            "final_eval/inception_score": inception_metrics["is"],
                            "final_eval/precision": inception_metrics["precision"],
                            "final_eval/recall": inception_metrics["recall"],
                            **{
                                f"final_eval/fd_{name}": value
                                for name, value in result.fd.items()
                            },
                            **(
                                {}
                                if intervened_result is None
                                else {
                                    "final_eval/intervened_fid": intervened_result.fd[
                                        "FID(ADM)"
                                    ],
                                    "final_eval/intervened_fdr_sim3": intervened_fdr_sim3,
                                }
                            ),
                        },
                        step=args.current_step,
                    )
    return rows


def _shutdown(wandb_logger) -> None:
    if wandb_logger:
        wandb_logger.finish()
    if is_enabled():
        torch_dist.barrier()
        torch_dist.destroy_process_group()


def _validate_training_contract(args: argparse.Namespace) -> None:
    if int(args.num_classes) != 1000:
        raise ValueError(
            "the canonical MMD setup requires all 1,000 ImageNet classes"
        )
    if bool(args.force_class_of_interest):
        raise ValueError(
            "force_class_of_interest must be false for full-ImageNet training/eval"
        )
    if int(args.mmd_repulsion_queue_size) < 1:
        raise ValueError(
            "mmd_repulsion_queue_size must be positive; current-batch-only "
            "repulsion is not supported"
        )
    if bool(getattr(args, "feature_intervention_enabled", False)):
        if not bool(getattr(args, "ot_enabled", False)):
            raise ValueError("the joint-SIM intervention requires Sinkhorn OT")
        if str(getattr(args, "ot_objective", "")) != "sinkhorn_divergence":
            raise ValueError(
                "the joint-SIM intervention requires the detached-self OT objective"
            )
        if float(getattr(args, "ot_baseline_weight", -1)) != 0:
            raise ValueError("the intervention pilot requires pure OT baseline_weight=0")
        if int(getattr(args, "feature_intervention_rank", 0) or 0) < 1:
            raise ValueError("feature_intervention_rank must be positive")
        strength = float(getattr(args, "feature_intervention_strength", -1))
        if not torch.isfinite(torch.tensor(strength)) or strength <= 0:
            raise ValueError("feature_intervention_strength must be finite and positive")
        if getattr(args, "feature_intervention_artifact", None) is None:
            raise ValueError("feature_intervention_artifact is required")
        if not bool(getattr(args, "feature_intervention_evaluate", False)):
            raise ValueError("the pilot must evaluate the intervened distribution")
        if float(args.noise_scale) != 1.0:
            raise ValueError("the frozen intervention artifact requires unit Gaussian latents")
        if bool(args.same_noise):
            raise ValueError("the intervention pilot requires independent latents")
    if bool(getattr(args, "xm_enabled", False)):
        for name in (
            "xm_num_classes_per_step",
            "xm_targets_per_class",
            "xm_candidates_per_class",
        ):
            if int(getattr(args, name)) < 1:
                raise ValueError(f"{name} must be positive")
        weight = float(args.xm_weight)
        if not torch.isfinite(torch.tensor(weight)) or weight <= 0:
            raise ValueError("xm_weight must be finite and positive when XM is enabled")
        if int(args.xm_num_classes_per_step) > int(args.num_classes):
            raise ValueError("xm_num_classes_per_step cannot exceed num_classes")
        assignment = str(getattr(args, "xm_assignment", "row_wise_min"))
        if assignment not in {"row_wise_min", "one_to_one"}:
            raise ValueError(
                "xm_assignment must be 'row_wise_min' or 'one_to_one'"
            )
        if assignment == "one_to_one" and (
            int(args.xm_targets_per_class)
            != int(args.xm_candidates_per_class)
        ):
            raise ValueError(
                "one_to_one xm_assignment requires equal target and candidate counts"
            )
    if bool(getattr(args, "ot_enabled", False)):
        if bool(getattr(args, "xm_enabled", False)):
            raise ValueError("Sinkhorn OT completely replaces the Forward-XM pass")
        for name in (
            "ot_num_classes",
            "ot_samples_per_class",
            "ot_num_real_refs",
            "ot_num_iters",
        ):
            if int(getattr(args, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if int(args.ot_num_classes) > int(args.num_classes):
            raise ValueError("ot_num_classes cannot exceed num_classes")
        for name in (
            "ot_baseline_weight",
            "lambda_ot",
            "ot_epsilon",
            "ot_tolerance",
        ):
            value = float(getattr(args, name))
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"{name} must be finite")
        if float(args.lambda_ot) <= 0:
            raise ValueError("lambda_ot must be positive when Sinkhorn OT is enabled")
        if float(args.ot_baseline_weight) < 0:
            raise ValueError("ot_baseline_weight must be nonnegative")
        if float(args.ot_baseline_weight) == 0 and bool(args.ot_log_grad_norms):
            raise ValueError(
                "OT/base gradient ratios are undefined when ot_baseline_weight=0"
            )
        if float(args.ot_epsilon) <= 0:
            raise ValueError("ot_epsilon must be positive")
        if float(args.ot_tolerance) < 0:
            raise ValueError("ot_tolerance must be nonnegative")
        expected_rollout = int(args.ot_num_classes) * int(
            args.ot_samples_per_class
        )
        if int(args.rollout_size) != expected_rollout:
            raise ValueError(
                "Sinkhorn OT requires rollout_size == "
                "ot_num_classes * ot_samples_per_class: "
                f"{args.rollout_size} != {expected_rollout}"
            )
        if not bool(args.mmd_local_generated_scoring):
            raise ValueError("Sinkhorn OT requires local generated MMD scoring")
        if str(args.mmd_normalize) != "none":
            raise ValueError("Sinkhorn OT baseline decomposition requires no normalization")
        if bool(args.mmd_include_real_constant):
            raise ValueError("Sinkhorn OT requires the existing no-real-constant path")
        if str(args.mmd_kernel) != "rbf":
            raise ValueError("the Sinkhorn OT experiment requires RBF attraction")
        if int(args.mmd_repulsion_queue_size) != 4:
            raise ValueError("the Sinkhorn OT experiment requires the historical Q4 queue")
        if str(args.mmd_repulsion_queue_kernel) != "laplacian_l2":
            raise ValueError(
                "the Sinkhorn OT experiment requires L2-Laplacian queue repulsion"
            )


def _validate_ot_distributed_contract(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "ot_enabled", False)):
        return
    if int(args.ot_num_classes) % int(args.world_size):
        raise ValueError("ot_num_classes must be divisible by world size")
    local_classes = int(args.ot_num_classes) // int(args.world_size)
    expected_local = local_classes * int(args.ot_samples_per_class)
    if int(args.local_rollout_size) != expected_local:
        raise ValueError(
            "rank-local rollout must contain complete OT class groups: "
            f"local_rollout={args.local_rollout_size}, expected={expected_local}"
        )
    if int(args.grad_accum) != 1 or int(args.batch_size) != expected_local:
        raise ValueError(
            "Sinkhorn OT requires one rank-local forward containing all class groups"
        )


def train_and_evaluate(args: argparse.Namespace) -> int:
    _validate_training_contract(args)
    wandb_logger = setup(args)
    _validate_ot_distributed_contract(args)
    register_preempt_handler()
    if args.grad_accum > 1:
        logger.info(
            "Using DistMatch feature replay: microbatch=%s chunks=%s "
            "local_rollout=%s global_rollout=%s",
            args.batch_size,
            args.grad_accum,
            args.local_rollout_size,
            args.rollout_size,
        )
    if args.online_eval:
        logger.info(
            "Online FDr-6/IS/P&R evaluation enabled at regular checkpoint frequency"
        )

    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    if args.train_last_n_blocks is not None:
        if not hasattr(model, "freeze_except_last_blocks"):
            raise ValueError("train_last_n_blocks is supported only for OST models")
        model.freeze_except_last_blocks(int(args.train_last_n_blocks))
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    checkpoint_extra = ckpt_resume(
        args,
        model,
        optimizer,
        ema_model,
        extra_keys=[
            MMD_QUEUE_STATE_KEY,
            MMD_RNG_STATE_KEY,
            XM_STREAM_STATE_KEY,
            OT_STREAM_STATE_KEY,
            FEATURE_INTERVENTION_STATE_KEY,
        ],
    )

    rng = RNGStateManager()
    rng.save()
    if not args.disable_vis or args.vis_only:
        visualize(
            args,
            model,
            ema_model,
            args.current_step,
            rng=rng,
            tokenizer=tokenizer,
        )
        if args.vis_only:
            _shutdown(wandb_logger)
            return 0

    device = next(model.parameters()).device
    representation_swap = getattr(args, "ot_representation_set", "legacy") == "pixel_drifting"
    ot_feature_weights = None
    feature_intervention = None
    if representation_swap:
        if bool(getattr(args, "feature_intervention_enabled", False)):
            raise ValueError("The frozen SIM intervention cannot be used with pixel Drifting features")
        if not args.ot_enabled or args.ot_objective != "sinkhorn_divergence" or args.ot_baseline_weight != 0 or args.xm_enabled:
            raise ValueError("Representation swap requires the pure debiased OT recipe")
        from src.ot_execution import create_provider
        objective = create_provider(args, device)
        sigmas = objective.sigmas
        ot_feature_weights = objective.weights
        full_class_attraction = objective.reference
        feature_banks = None
        if checkpoint_extra and MMD_QUEUE_STATE_KEY in checkpoint_extra:
            objective.load_running_state_dict(checkpoint_extra[MMD_QUEUE_STATE_KEY])
        elif args.current_step > 0:
            raise RuntimeError("Resumed representation swap has no saved manifest identity")
        if args.rank == 0:
            logger.info(
                "Representation swap: stack=%s; families=%s; outputs=%d; "
                "reference encoding online; manifest=%s",
                objective.stack,
                ",".join(objective.encoders.family_names),
                len(objective.names),
                objective.manifest_digest,
            )
    else:
        training_features = _features(TRAINING_SPECS)
        sigmas = _mmd_sigmas(args, training_features)
        feature_intervention = _build_feature_intervention(
            args, model, sigmas, checkpoint_extra
        )
        queue_bandwidths = _mmd_queue_bandwidths(args)
        objective = MMD(
            features=training_features,
            bandwidths=sigmas,
            weights=_training_weights(args),
            normalize=args.mmd_normalize,
            normalize_eps=args.mmd_normalize_eps,
            kernel_chunk_size=args.mmd_kernel_chunk_size,
            kernel=args.mmd_kernel,
            local_generated_scoring=args.mmd_local_generated_scoring,
            include_real_constant=args.mmd_include_real_constant,
            attraction_mode="within_class",
            attraction_balance="class",
            repulsion_mode="within_class",
            repulsion_queue_size=args.mmd_repulsion_queue_size,
            repulsion_queue_num_classes=args.num_classes,
            repulsion_queue_weight=args.mmd_repulsion_queue_weight,
            repulsion_queue_kernel=args.mmd_repulsion_queue_kernel,
            repulsion_queue_bandwidths=queue_bandwidths,
            distributed="auto",
            device=device,
        )
        if checkpoint_extra and MMD_QUEUE_STATE_KEY in checkpoint_extra:
            objective.load_running_state_dict(
                checkpoint_extra[MMD_QUEUE_STATE_KEY], strict=True
            )
            logger.info(
                "Restored classwise MMD repulsion queue: %s",
                objective.repulsion_queue_fill_stats,
            )
        elif objective.repulsion_queue_size > 0 and args.current_step > 0:
            logger.warning(
                "Resumed at step %s without a saved MMD repulsion queue; "
                "the queue will warm up from empty",
                args.current_step,
            )
        feature_banks = _load_feature_banks(args, objective)
        full_class_attraction = _build_full_class_attraction(
            args,
            feature_banks,
        )
    xm_objective = None
    xm_stream = None
    if bool(getattr(args, "xm_enabled", False)):
        xm_objective = SharedForwardXM(
            bandwidths=sigmas,
            weights=_training_weights(args),
            candidates_per_class=args.xm_candidates_per_class,
            targets_per_class=args.xm_targets_per_class,
            assignment=getattr(args, "xm_assignment", "row_wise_min"),
            distributed="auto",
        )
        xm_stream = ClasswiseXMStream(
            class_counts=full_class_attraction.class_counts,
            global_classes_per_step=args.xm_num_classes_per_step,
            targets_per_class=args.xm_targets_per_class,
            candidates_per_class=args.xm_candidates_per_class,
            rank=args.rank,
            world_size=args.world_size,
            seed=args.xm_seed,
            device=device,
        )
    ot_objective = None
    ot_self_objective = None
    ot_stream = None
    if bool(getattr(args, "ot_enabled", False)):
        ot_objective_name = str(getattr(args, "ot_objective", "cross"))
        if ot_objective_name not in {"cross", "sinkhorn_divergence"}:
            raise ValueError(
                "ot_objective must be cross or sinkhorn_divergence"
            )
        ot_objective = LossAblationOT(
            discrepancy_kind=getattr(args, "discrepancy_kind", "joint_ot"),
            energy_delta=getattr(args, "energy_delta", 0.05),
            bandwidths=sigmas,
            # The OT ablation is deliberately fixed to the Forward-XM joint
            # distance: an equal 1/3 contribution from each feature space.
            # MMD representation weights continue to affect only the unchanged
            # attraction and Q4-repulsion baseline.
            weights=ot_feature_weights,
            global_classes_per_step=args.ot_num_classes,
            samples_per_class=args.ot_samples_per_class,
            real_references_per_class=args.ot_num_real_refs,
            epsilon=args.ot_epsilon,
            epsilon_mixture=getattr(args, "ot_epsilon_mixture", None),
            log_feature_grad_norms=getattr(args, "ot_log_feature_grad_norms", False),
            num_iterations=args.ot_num_iters,
            tolerance=args.ot_tolerance,
            distributed="auto",
        )
        if ot_objective_name == "sinkhorn_divergence":
            ot_self_objective = LossAblationOT(
                discrepancy_kind=getattr(args, "discrepancy_kind", "joint_ot"),
                energy_delta=getattr(args, "energy_delta", 0.05),
                bandwidths=sigmas,
                weights=ot_feature_weights,
                global_classes_per_step=args.ot_num_classes,
                samples_per_class=args.ot_samples_per_class,
                real_references_per_class=args.ot_samples_per_class,
                epsilon=args.ot_epsilon,
                epsilon_mixture=getattr(args, "ot_epsilon_mixture", None),
                log_feature_grad_norms=getattr(args, "ot_log_feature_grad_norms", False),
                num_iterations=args.ot_num_iters,
                tolerance=args.ot_tolerance,
                distributed="auto",
            )
        ot_stream = ClasswiseOTStream(
            class_counts=full_class_attraction.class_counts,
            global_classes_per_step=args.ot_num_classes,
            samples_per_class=args.ot_samples_per_class,
            real_references_per_class=args.ot_num_real_refs,
            rank=args.rank,
            world_size=args.world_size,
            seed=args.ot_seed,
            device=device,
        )
    # The grouped full-reference copies now own everything needed by the loss.
    del feature_banks
    gc.collect()
    torch.cuda.empty_cache()
    if get_global_rank() == 0:
        logger.info(
            "Feature provider: kernel=%s "
            "attraction_bandwidths=%s "
            "baseline_active_only_when_ot_baseline_weight_is_positive "
            "queue_size_per_class=%s queue_kernel=%s "
            "queue_bandwidths=%s queue_weight=%s "
            "local_generated_scoring=%s include_real_constant=%s",
            objective.kernel,
            sigmas,
            objective.repulsion_queue_size,
            objective.repulsion_queue_kernel,
            objective.repulsion_queue_bandwidths,
            objective.repulsion_queue_weight,
            objective.local_generated_scoring,
            objective.include_real_constant,
        )
        if xm_objective is not None and xm_stream is not None:
            logger.info(
                "Using auxiliary shared-pool Forward-XM: weight=%s "
                "global_classes_per_step=%s targets_per_class=%s "
                "candidates_per_class=%s global_targets=%s global_candidates=%s "
                "distance=equal_weight_mean_feature_squared_l2_over_2sigma2 "
                "assignment=%s queue_updates=false seed=%s "
                "feature_extractors=eager branch_grad_diagnostics=%s",
                args.xm_weight,
                args.xm_num_classes_per_step,
                args.xm_targets_per_class,
                args.xm_candidates_per_class,
                xm_stream.global_target_count,
                xm_stream.global_candidate_count,
                getattr(args, "xm_assignment", "row_wise_min"),
                args.xm_seed,
                args.xm_log_branch_grad_norms,
            )
        if ot_objective is not None and ot_stream is not None:
            logger.info(
                "Using shared-main-rollout balanced Sinkhorn OT: "
                "objective=%s weight=%s baseline_weight=%s "
                "global_classes_per_step=%s "
                "local_classes_per_rank=%s "
                "samples_per_class=%s real_references_per_class=%s "
                "global_generated=%s local_generated=%s "
                "cost=one_joint_weighted_feature_squared_l2_over_2sigma2 "
                "epsilon=%s epsilon_mixture=%s iterations=%s tolerance=%s "
                "plan_gradient=false generated_feature_all_gather=false "
                "seed=%s q_prime_noise=%s branch_grad_diagnostics=%s",
                getattr(args, "ot_objective", "cross"),
                args.lambda_ot,
                args.ot_baseline_weight,
                args.ot_num_classes,
                ot_stream.local_classes_per_step,
                args.ot_samples_per_class,
                args.ot_num_real_refs,
                ot_stream.global_generated_count,
                ot_stream.local_generated_count,
                args.ot_epsilon,
                ot_objective.epsilon_mixture,
                args.ot_num_iters,
                args.ot_tolerance,
                args.ot_seed,
                (
                    "private_checkpointed"
                    if ot_self_objective is not None
                    else "disabled"
                ),
                args.ot_log_grad_norms,
            )
    if bool(getattr(args, "measure_flops", False)):
        measure_training_flops(
            args,
            model,
            tokenizer,
            objective,
            full_class_attraction,
            ot_objective,
            ot_self_objective,
            ot_stream,
            feature_intervention,
        )
    compile_training_modules(args, model, objective)

    if _restore_checkpoint_rng_state(args, rng, checkpoint_extra):
        logger.info("Restored rank-local Python/NumPy/Torch/CUDA RNG state")
    elif args.current_step > 0:
        logger.warning(
            "Checkpoint has no saved MMD RNG state; continuation uses the "
            "fresh deterministic system seed from this process start"
        )
    if _restore_checkpoint_xm_stream_state(args, xm_stream, checkpoint_extra):
        logger.info("Restored rank-local Forward-XM target/noise stream state")
    elif xm_stream is not None and args.current_step > 0:
        raise RuntimeError(
            "resumed XM training checkpoint has no Forward-XM stream state"
        )
    if _restore_checkpoint_ot_stream_state(
        args,
        ot_stream,
        checkpoint_extra,
        require_negative_noise_state=ot_self_objective is not None,
    ):
        logger.info(
            "Restored rank-local Sinkhorn OT class/reference/q-prime streams"
        )
    elif ot_stream is not None and args.current_step > 0:
        raise RuntimeError(
            "resumed Sinkhorn OT checkpoint has no class/reference stream state"
        )
    del checkpoint_extra

    if (
        feature_intervention is not None
        and bool(getattr(args, "feature_intervention_evaluate_initial", False))
        and args.current_step == 0
    ):
        logger.info("Running rank-spread step-zero FDr evaluation")
        rng_state = rng.snapshot()
        try:
            run_fdr6_evaluation(
                args,
                model,
                ema_model,
                tokenizer,
                wandb_logger=wandb_logger,
                feature_intervention=feature_intervention,
            )
        finally:
            rng.load(rng_state)
            gc.collect()
            torch.cuda.empty_cache()

    completed = run_training(
        args,
        model,
        ema_model,
        optimizer,
        tokenizer,
        objective,
        full_class_attraction,
        xm_objective,
        xm_stream,
        ot_objective,
        ot_self_objective,
        ot_stream,
        feature_intervention,
        wandb_logger=wandb_logger,
        rng=rng,
    )
    if completed:
        del objective, full_class_attraction
        gc.collect()
        torch.cuda.empty_cache()
        if bool(getattr(args, "skip_final_eval", False)):
            logger.info("Skipping final FDr-6/IS/P&R evaluation by configuration")
        else:
            run_fdr6_evaluation(
                args,
                model,
                ema_model,
                tokenizer,
                wandb_logger=wandb_logger,
                feature_intervention=feature_intervention,
            )

        from src.eval.run_completion import publish_completed_run
        publish_completed_run(args, model, ema_model)

    _shutdown(wandb_logger)
    return 0


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "OST featurewise OT training and evaluation",
        add_help=True,
    )
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def _make_args(cfg, config_path: str, overrides: Sequence[str]) -> argparse.Namespace:
    values: dict[str, Any] = {}
    for section in CONFIG_ARG_SECTIONS:
        section_values = OmegaConf.to_container(cfg.get(section, OmegaConf.create({})), resolve=True)
        overlap = set(values).intersection(section_values)
        if overlap:
            raise ValueError(
                "duplicate config keys across sections: " + ", ".join(sorted(overlap))
            )
        values.update(section_values)
    values["config_path"] = config_path
    values["config"] = OmegaConf.to_container(cfg, resolve=True)
    values["overrides"] = list(overrides)
    return argparse.Namespace(**values)


def parse_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    cli = get_args_parser().parse_args(argv)
    cfg = OmegaConf.load(cli.config)
    if cli.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli.overrides))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if cli.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        raise SystemExit(0)
    return _make_args(cfg, cli.config, cli.overrides)


if __name__ == "__main__":
    raise SystemExit(train_and_evaluate(parse_config()))
