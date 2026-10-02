"""Exact, calibration-oriented helpers for image memorization audits.

The training code intentionally does not import this module.  It contains the
small tensor operations shared by the offline DINOv2 bank builder and the
checkpoint nearest-neighbor audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CosineSearchResult:
    """Top-k cosine similarities and bank-global row indices."""

    scores: torch.Tensor
    indices: torch.Tensor


def l2_normalize_rows(rows: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return finite float32 unit rows."""
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("rows must be a rank-two tensor")
    if rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError("rows must be nonempty")
    rows = rows.float()
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("rows contain NaN or Inf")
    return F.normalize(rows, p=2.0, dim=1, eps=float(eps))


@torch.no_grad()
def exact_cosine_topk(
    queries: torch.Tensor,
    references: torch.Tensor,
    *,
    k: int = 5,
    query_chunk_size: int = 256,
    reference_chunk_size: int = 65_536,
    exclude_indices: torch.Tensor | None = None,
) -> CosineSearchResult:
    """Perform an exact blocked top-k search over already-normalized rows.

    ``exclude_indices`` optionally supplies one bank-global row to mask for
    each query.  It is used for train leave-one-out radii.  Results stay on the
    input device and indices always address ``references`` globally.
    """
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("queries and references must be rank-two tensors")
    if queries.shape[0] < 1 or references.shape[0] < 1:
        raise ValueError("queries and references must be nonempty")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference dimensions differ")
    if queries.device != references.device:
        raise ValueError("queries and references must be on the same device")
    k = int(k)
    query_chunk_size = int(query_chunk_size)
    reference_chunk_size = int(reference_chunk_size)
    if k < 1 or query_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("k and chunk sizes must be positive")
    maximum_k = references.shape[0] - (1 if exclude_indices is not None else 0)
    if k > maximum_k:
        raise ValueError(f"k={k} exceeds the number of searchable rows {maximum_k}")
    if exclude_indices is not None:
        if exclude_indices.shape != (queries.shape[0],):
            raise ValueError("exclude_indices must contain one index per query")
        if exclude_indices.device != queries.device:
            exclude_indices = exclude_indices.to(queries.device)
        exclude_indices = exclude_indices.long()
        if bool(((exclude_indices < 0) | (exclude_indices >= references.shape[0])).any()):
            raise ValueError("exclude_indices contains an out-of-range row")

    output_scores: list[torch.Tensor] = []
    output_indices: list[torch.Tensor] = []
    for query_start in range(0, queries.shape[0], query_chunk_size):
        query_stop = min(query_start + query_chunk_size, queries.shape[0])
        query_block = queries[query_start:query_stop].float()
        masked = None if exclude_indices is None else exclude_indices[query_start:query_stop]
        best_scores: torch.Tensor | None = None
        best_indices: torch.Tensor | None = None

        for reference_start in range(0, references.shape[0], reference_chunk_size):
            reference_stop = min(
                reference_start + reference_chunk_size, references.shape[0]
            )
            similarity = query_block @ references[reference_start:reference_stop].float().T
            if masked is not None:
                reference_ids = torch.arange(
                    reference_start,
                    reference_stop,
                    dtype=torch.int64,
                    device=queries.device,
                )
                similarity.masked_fill_(masked[:, None] == reference_ids[None, :], -torch.inf)

            block_k = min(k, reference_stop - reference_start)
            block_scores, block_offsets = torch.topk(
                similarity, k=block_k, dim=1, largest=True, sorted=True
            )
            block_indices = block_offsets + reference_start
            if best_scores is None:
                best_scores, best_indices = block_scores, block_indices
                continue
            candidate_scores = torch.cat((best_scores, block_scores), dim=1)
            candidate_indices = torch.cat((best_indices, block_indices), dim=1)
            merged_k = min(k, candidate_scores.shape[1])
            best_scores, positions = torch.topk(
                candidate_scores, k=merged_k, dim=1, largest=True, sorted=True
            )
            best_indices = candidate_indices.gather(1, positions)

        assert best_scores is not None and best_indices is not None
        if best_scores.shape[1] != k or not bool(torch.isfinite(best_scores).all()):
            raise RuntimeError("exact cosine search did not produce k finite neighbors")
        output_scores.append(best_scores)
        output_indices.append(best_indices.long())

    return CosineSearchResult(
        scores=torch.cat(output_scores, dim=0),
        indices=torch.cat(output_indices, dim=0),
    )


@torch.no_grad()
def exact_same_class_cosine_topk(
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    references: torch.Tensor,
    reference_labels: torch.Tensor,
    *,
    k: int = 5,
    query_chunk_size: int = 256,
    reference_chunk_size: int = 65_536,
) -> CosineSearchResult:
    """Exact top-k search restricted to rows matching each query label."""
    if query_labels.shape != (queries.shape[0],):
        raise ValueError("query_labels must contain one label per query")
    if reference_labels.shape != (references.shape[0],):
        raise ValueError("reference_labels must contain one label per reference")
    query_labels = query_labels.to(device=queries.device, dtype=torch.int64)
    reference_labels = reference_labels.to(device=queries.device, dtype=torch.int64)
    output_scores = torch.empty(
        (queries.shape[0], int(k)), dtype=torch.float32, device=queries.device
    )
    output_indices = torch.empty(
        (queries.shape[0], int(k)), dtype=torch.int64, device=queries.device
    )

    for class_id in torch.unique(query_labels, sorted=True).tolist():
        query_indices = torch.nonzero(
            query_labels == int(class_id), as_tuple=False
        ).flatten()
        reference_indices = torch.nonzero(
            reference_labels == int(class_id), as_tuple=False
        ).flatten()
        if reference_indices.numel() < int(k):
            raise ValueError(
                f"class {class_id} has {reference_indices.numel()} rows, fewer than k={k}"
            )
        selected = exact_cosine_topk(
            queries.index_select(0, query_indices),
            references.index_select(0, reference_indices),
            k=k,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
        )
        output_scores.index_copy_(0, query_indices, selected.scores)
        output_indices.index_copy_(
            0, query_indices, reference_indices.index_select(0, selected.indices.flatten()).view_as(selected.indices)
        )

    return CosineSearchResult(scores=output_scores, indices=output_indices)


def balanced_indices(
    labels: torch.Tensor,
    *,
    total: int,
    seed: int = 0,
) -> torch.Tensor:
    """Select an equal deterministic random count from every represented class."""
    labels = labels.detach().cpu().long()
    classes = torch.unique(labels, sorted=True)
    total = int(total)
    if total < 1 or total % classes.numel() != 0:
        raise ValueError("total must be a positive multiple of the class count")
    per_class = total // classes.numel()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for class_id in classes.tolist():
        candidates = torch.nonzero(labels == int(class_id), as_tuple=False).flatten()
        if candidates.numel() < per_class:
            raise ValueError(
                f"class {class_id} has {candidates.numel()} rows, fewer than {per_class}"
            )
        order = torch.randperm(candidates.numel(), generator=generator)[:per_class]
        selected.append(candidates.index_select(0, order))
    return torch.cat(selected).long()


def balanced_subset_indices(
    labels: torch.Tensor,
    *,
    classes: torch.Tensor | list[int] | tuple[int, ...],
    maximum_total: int,
    seed: int = 0,
) -> torch.Tensor:
    """Select an equal deterministic sample from specified classes.

    The largest feasible per-class count is used without exceeding
    ``maximum_total``.  This is useful for a class-restricted audit: ImageNet
    validation has only 50 rows per class, while the corresponding training
    classes have roughly 1,300 rows.  Silently replacing those controls with
    samples from unrelated classes would make the calibration misleading.
    """
    labels = labels.detach().cpu().long()
    requested = torch.as_tensor(classes, dtype=torch.int64).flatten()
    maximum_total = int(maximum_total)
    if requested.numel() < 1:
        raise ValueError("classes must contain at least one class ID")
    if torch.unique(requested).numel() != requested.numel():
        raise ValueError("classes must not contain duplicates")
    if maximum_total < requested.numel():
        raise ValueError("maximum_total must allow at least one row per class")

    candidates: list[torch.Tensor] = []
    for class_id in requested.tolist():
        rows = torch.nonzero(labels == int(class_id), as_tuple=False).flatten()
        if rows.numel() < 1:
            raise ValueError(f"requested class {class_id} is absent from labels")
        candidates.append(rows)

    per_class = min(
        maximum_total // requested.numel(),
        min(rows.numel() for rows in candidates),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[torch.Tensor] = []
    for rows in candidates:
        order = torch.randperm(rows.numel(), generator=generator)[:per_class]
        selected.append(rows.index_select(0, order))
    return torch.cat(selected).long()


def first_nonduplicate_distances(
    neighbor_scores: torch.Tensor,
    *,
    duplicate_distance_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the first cosine neighbor outside an effectively identical cluster.

    Returns ``(distance, rank)``.  Rank is zero-based; if every supplied
    neighbor is a duplicate, the last available neighbor is used.
    """
    if neighbor_scores.ndim != 2 or neighbor_scores.shape[1] < 1:
        raise ValueError("neighbor_scores must have shape [N, K] with K >= 1")
    distances = (1.0 - neighbor_scores.float()).clamp_min(0.0)
    acceptable = distances > float(duplicate_distance_eps)
    ranks = acceptable.float().argmax(dim=1).long()
    no_acceptable = ~acceptable.any(dim=1)
    ranks[no_acceptable] = distances.shape[1] - 1
    chosen = distances.gather(1, ranks[:, None]).squeeze(1)
    return chosen, ranks


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    """Stable scalar summary used in JSON reports."""
    values = values.detach().float().cpu().flatten()
    if values.numel() < 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("values must be nonempty and finite")
    quantiles = torch.tensor([0.0, 0.5, 0.9, 0.99, 0.999, 1.0])
    resolved = torch.quantile(values, quantiles)
    return {
        "min": float(resolved[0]),
        "q50": float(resolved[1]),
        "q90": float(resolved[2]),
        "q99": float(resolved[3]),
        "q999": float(resolved[4]),
        "max": float(resolved[5]),
        "mean": float(values.mean()),
    }


__all__ = [
    "CosineSearchResult",
    "balanced_indices",
    "exact_cosine_topk",
    "exact_same_class_cosine_topk",
    "first_nonduplicate_distances",
    "l2_normalize_rows",
    "quantile_summary",
]
