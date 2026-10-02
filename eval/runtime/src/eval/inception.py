"""Native Inception Score and precision/recall evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as torch_dist

from src.frechet_distance.evaluator import extract_ref_features, gather_features
from src.frechet_distance.metrics import compute_isc, compute_precision_recall
from src.frechet_distance.repr_models import load_repr_model
from src.utils.distributed_util import (
    broadcast_scalar,
    get_global_rank,
    get_world_size,
    is_enabled,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRC_REF_DIR = REPO_ROOT / "data" / "prc" / "VIRTUAL_imagenet256_labeled"

logger = logging.getLogger("FD_loss")


@torch.inference_mode()
def _prepare_inception_evaluation(args, *, device: torch.device):
    """Load the native Inception model and real-image P&R features.

    This deliberately uses the implementation under ``src.frechet_distance``
    rather than DistMatch. Inception forwards remain in FP32, matching
    ``scripts/eval_all_fds.py``.
    """
    inception_model, feature_dim, has_logits, _ = load_repr_model(
        "inception", device=device
    )
    if not has_logits:
        raise RuntimeError("The Inception representation model must return logits")
    inception_model.eval().requires_grad_(False)

    configured_ref_dir = getattr(args, "prc_ref_dir", None)
    reference_dir = Path(configured_ref_dir or DEFAULT_PRC_REF_DIR).expanduser()
    if not reference_dir.is_absolute():
        reference_dir = REPO_ROOT / reference_dir
    cache_path = Path(f"{reference_dir}_inception.pt")
    rank = get_global_rank()

    if cache_path.is_file():
        reference_features = (
            torch.load(cache_path, map_location="cpu", weights_only=True)
            if rank == 0
            else None
        )
        if rank == 0:
            logger.info(
                "Loaded cached Inception P&R reference features from %s (%s)",
                cache_path,
                tuple(reference_features.shape),
            )
    else:
        if not reference_dir.is_dir():
            raise FileNotFoundError(
                f"Inception P&R reference directory does not exist: {reference_dir}"
            )

        def extract_inception_features(images):
            features, _ = inception_model(images)
            return features

        reference_features = extract_ref_features(
            extract_inception_features,
            str(reference_dir),
            cache_path=None,
            batch_size=64,
            device=device,
            img_size=256,
        )
        if rank == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(reference_features, cache_path)
            logger.info(
                "Cached Inception P&R reference features at %s (%s)",
                cache_path,
                tuple(reference_features.shape),
            )

    if rank == 0 and (
        reference_features is None
        or reference_features.ndim != 2
        or reference_features.shape[1] != feature_dim
    ):
        shape = None if reference_features is None else tuple(reference_features.shape)
        raise RuntimeError(
            f"Expected Inception P&R reference features [N, {feature_dim}], got {shape}"
        )
    return inception_model, reference_features


@torch.inference_mode()
def _accumulate_inception_batch(
    inception_model,
    images: torch.Tensor,
    feature_batches: list[torch.Tensor],
    logit_batches: list[torch.Tensor],
) -> None:
    """Extract native FP32 Inception features and logits from one image batch."""
    features, logits = inception_model(images)
    if logits is None:
        raise RuntimeError("Inception did not return classification logits")
    feature_batches.append(features.float().cpu())
    logit_batches.append(logits.cpu())


@torch.inference_mode()
def run_inception_evaluation(
    args,
    feature_batches: list[torch.Tensor],
    logit_batches: list[torch.Tensor],
    reference_features: torch.Tensor | None,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    """Compute Inception Score and k-NN precision/recall across all ranks.

    Generated features and logits come from the same image batches used for
    FDr-6. The aggregation and metric implementations match
    ``scripts/eval_all_fds.py``.
    """
    if not feature_batches or not logit_batches:
        raise RuntimeError("Inception evaluation received no generated samples")

    local_feature_count = sum(batch.shape[0] for batch in feature_batches)
    local_logit_count = sum(batch.shape[0] for batch in logit_batches)
    if local_feature_count != local_logit_count:
        raise RuntimeError(
            "Inception feature/logit counts differ: "
            f"{local_feature_count} != {local_logit_count}"
        )

    count = torch.tensor([local_feature_count], dtype=torch.long, device=device)
    if is_enabled():
        torch_dist.all_reduce(count, op=torch_dist.ReduceOp.SUM)
    total_count = int(count.item())
    if total_count != args.num_images:
        raise RuntimeError(
            f"Inception evaluation expected {args.num_images} images, got {total_count}"
        )

    local_features = torch.cat(feature_batches, dim=0).to(device)
    local_logits = torch.cat(logit_batches, dim=0).to(device)
    feature_batches.clear()
    logit_batches.clear()

    world_size = get_world_size()
    rank = get_global_rank()
    if world_size > 1:
        generated_features = gather_features(
            local_features, world_size, rank, device
        )
        generated_logits = gather_features(local_logits, world_size, rank, device)
    else:
        generated_features = local_features
        generated_logits = local_logits
    del local_features, local_logits

    if rank == 0:
        if generated_features is None or generated_logits is None:
            raise RuntimeError("Rank 0 did not receive gathered Inception outputs")
        if reference_features is None:
            raise RuntimeError("Rank 0 has no Inception P&R reference features")

        logits_cpu = generated_logits.cpu()
        del generated_logits
        inception_score, _ = compute_isc(logits_cpu)
        del logits_cpu

        generated_features = generated_features.float()
        reference_gpu = reference_features.to(device=device, dtype=torch.float32)
        prc_k = int(getattr(args, "prc_k", 3))
        prc_batch_size = int(getattr(args, "prc_batch_size", 10000))
        logger.info(
            "Inception P&R: ref=%s gen=%s k=%s",
            reference_gpu.shape[0],
            generated_features.shape[0],
            prc_k,
        )
        precision, recall = compute_precision_recall(
            reference_gpu,
            generated_features,
            k=prc_k,
            batch_size=prc_batch_size,
        )
        del reference_gpu, generated_features
    else:
        inception_score = precision = recall = 0.0
        del generated_features, generated_logits

    inception_score = broadcast_scalar(inception_score, device=device)
    precision = broadcast_scalar(precision, device=device)
    recall = broadcast_scalar(recall, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Inception Score=%.4f Precision=%.4f Recall=%.4f",
        inception_score,
        precision,
        recall,
    )
    return {
        "is": inception_score,
        "precision": precision,
        "recall": recall,
        "num_samples": total_count,
    }
