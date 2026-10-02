from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint


SUPPORTED_KERNELS = ("rbf", "laplacian_l2")


def validate_kernel_name(kernel: str) -> str:
    """Return the canonical kernel name or reject an unsupported kernel.

    Kernel selection is intentionally explicit. ``rbf`` uses squared L2
    distance, while ``laplacian_l2`` uses unsquared L2 distance.
    """
    name = str(kernel).strip().lower()
    if name not in SUPPORTED_KERNELS:
        supported = ", ".join(SUPPORTED_KERNELS)
        raise ValueError(f"unsupported MMD kernel {kernel!r}; supported: {supported}")
    return name


def _kernel_from_squared_distance(
    squared_distances: torch.Tensor,
    gamma: float,
    *,
    kernel: str,
) -> torch.Tensor:
    kernel = validate_kernel_name(kernel)
    if kernel == "rbf":
        return torch.exp(-float(gamma) * squared_distances)
    raise ValueError(
        "laplacian_l2 must be evaluated from feature rows, not a square root "
        "of a Gram-matrix squared distance"
    )


def _kernel_matrix(
    left: torch.Tensor,
    right: torch.Tensor,
    parameter: float,
    *,
    kernel: str,
    left_squared_norms: torch.Tensor | None = None,
    right_squared_norms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate a kernel matrix with a finite L2-Laplacian zero subgradient.

    The direct ``torch.cdist`` path is deliberate. Taking ``sqrt`` of a
    Gram-matrix squared distance has an undefined derivative at exact equality
    and can turn a valid queue collision into NaN gradients. ``torch.cdist``
    returns the finite zero subgradient at equality while preserving the
    limiting ``1 / bandwidth`` force at every positive separation.
    """
    kernel = validate_kernel_name(kernel)
    if kernel == "rbf":
        if left_squared_norms is None:
            left_squared_norms = left.square().sum(dim=1)
        if right_squared_norms is None:
            right_squared_norms = right.square().sum(dim=1)
        squared_distances = (
            left_squared_norms[:, None]
            + right_squared_norms[None, :]
            - 2.0 * (left @ right.T)
        ).clamp_min(0)
        return torch.exp(-float(parameter) * squared_distances)
    if kernel == "laplacian_l2":
        distances = torch.cdist(
            left,
            right,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        return torch.exp(-float(parameter) * distances)
    raise AssertionError(f"unhandled validated kernel: {kernel}")


def kernel_parameter_from_bandwidth(bandwidth: float, *, kernel: str) -> float:
    """Map a human-scale bandwidth to the kernel exponent coefficient.

    ``rbf`` is ``exp(-||x-y||^2 / (2 sigma^2))`` and therefore returns
    ``1 / (2 sigma^2)``. ``laplacian_l2`` is ``exp(-||x-y|| / ell)`` and
    returns ``1 / ell``.
    """
    bandwidth = float(bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    kernel = validate_kernel_name(kernel)
    if kernel == "rbf":
        return 1.0 / (2.0 * bandwidth * bandwidth)
    if kernel == "laplacian_l2":
        return 1.0 / bandwidth
    raise AssertionError(f"unhandled validated kernel: {kernel}")


def gamma_from_bandwidth(bandwidth: float) -> float:
    """Backward-compatible RBF exponent coefficient from sigma."""
    return kernel_parameter_from_bandwidth(bandwidth, kernel="rbf")


def squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            "squared_distance expects [N, D] and [M, D], got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    left_squared = left.square().sum(dim=1)
    right_squared = right.square().sum(dim=1)
    return (
        left_squared[:, None]
        + right_squared[None, :]
        - 2.0 * (left @ right.T)
    ).clamp_min(0)


def gaussian_kernel(
    left: torch.Tensor, right: torch.Tensor, gamma: float
) -> torch.Tensor:
    return _kernel_from_squared_distance(
        squared_distance(left, right), gamma, kernel="rbf"
    )


def _kernel_block_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    left_squared_norms: torch.Tensor,
    right_squared_norms: torch.Tensor,
    gamma: float,
    kernel: str,
) -> torch.Tensor:
    return _kernel_matrix(
        left,
        right,
        gamma,
        kernel=kernel,
        left_squared_norms=left_squared_norms,
        right_squared_norms=right_squared_norms,
    ).sum()


def _checked_kernel_rows(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_size: int,
) -> int:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            "kernel mean expects [N, D] and [M, D], got "
            f"{tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.shape[0] < 1 or right.shape[0] < 1:
        raise ValueError("kernel mean requires at least one row in each input")
    if left.device != right.device:
        raise ValueError(
            f"kernel mean inputs must share a device, got {left.device} and {right.device}"
        )
    if left.dtype != right.dtype:
        raise ValueError(
            f"kernel mean inputs must share a dtype, got {left.dtype} and {right.dtype}"
        )
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    return int(chunk_size)


def _checkpointed_kernel_block_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    left_squared_norms: torch.Tensor,
    right_squared_norms: torch.Tensor,
    gamma: float,
    kernel: str,
) -> torch.Tensor:
    def block(
        left_rows: torch.Tensor,
        right_rows: torch.Tensor,
        left_norms: torch.Tensor,
        right_norms: torch.Tensor,
    ) -> torch.Tensor:
        return _kernel_block_sum(
            left_rows,
            right_rows,
            left_norms,
            right_norms,
            gamma,
            kernel,
        )

    if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
        return checkpoint(
            block,
            left,
            right,
            left_squared_norms,
            right_squared_norms,
            use_reentrant=False,
        )
    return block(left, right, left_squared_norms, right_squared_norms)


def _class_weighted_kernel_block_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    left_squared_norms: torch.Tensor,
    right_squared_norms: torch.Tensor,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    gamma: float,
    kernel: str,
) -> torch.Tensor:
    kernels = _kernel_matrix(
        left,
        right,
        gamma,
        kernel=kernel,
        left_squared_norms=left_squared_norms,
        right_squared_norms=right_squared_norms,
    )
    same_class = left_labels[:, None] == right_labels[None, :]
    pair_weights = left_weights[:, None] * right_weights[None, :]
    return (kernels * pair_weights * same_class).sum()


def _checkpointed_class_weighted_kernel_block_sum(
    left: torch.Tensor,
    right: torch.Tensor,
    left_squared_norms: torch.Tensor,
    right_squared_norms: torch.Tensor,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    left_weights: torch.Tensor,
    right_weights: torch.Tensor,
    gamma: float,
    kernel: str,
) -> torch.Tensor:
    def block(
        left_rows: torch.Tensor,
        right_rows: torch.Tensor,
        left_norms: torch.Tensor,
        right_norms: torch.Tensor,
        left_class_ids: torch.Tensor,
        right_class_ids: torch.Tensor,
        left_class_weights: torch.Tensor,
        right_class_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _class_weighted_kernel_block_sum(
            left_rows,
            right_rows,
            left_norms,
            right_norms,
            left_class_ids,
            right_class_ids,
            left_class_weights,
            right_class_weights,
            gamma,
            kernel,
        )

    if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
        return checkpoint(
            block,
            left,
            right,
            left_squared_norms,
            right_squared_norms,
            left_labels,
            right_labels,
            left_weights,
            right_weights,
            use_reentrant=False,
        )
    return block(
        left,
        right,
        left_squared_norms,
        right_squared_norms,
        left_labels,
        right_labels,
        left_weights,
        right_weights,
    )


def _checked_class_labels(
    labels: torch.Tensor,
    rows: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if labels.ndim != 1 or labels.shape[0] != rows.shape[0]:
        raise ValueError(
            f"{name} must have shape [{rows.shape[0]}], got {tuple(labels.shape)}"
        )
    if labels.device != rows.device:
        raise ValueError(
            f"{name} is on {labels.device}, but its features are on {rows.device}"
        )
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"{name} must contain integer class IDs, got {labels.dtype}")
    return labels


def cross_kernel_mean(
    left: torch.Tensor,
    right: torch.Tensor,
    gamma: float,
    *,
    chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Exact cross-kernel mean using checkpointed two-dimensional tiles."""
    kernel = validate_kernel_name(kernel)
    chunk_size = _checked_kernel_rows(left, right, chunk_size=chunk_size)
    left_squared_norms = left.square().sum(dim=1)
    right_squared_norms = right.square().sum(dim=1)
    total = left.new_zeros(())
    for left_start in range(0, left.shape[0], chunk_size):
        left_stop = min(left_start + chunk_size, left.shape[0])
        for right_start in range(0, right.shape[0], chunk_size):
            right_stop = min(right_start + chunk_size, right.shape[0])
            total = total + _checkpointed_kernel_block_sum(
                left[left_start:left_stop],
                right[right_start:right_stop],
                left_squared_norms[left_start:left_stop],
                right_squared_norms[right_start:right_stop],
                gamma,
                kernel,
            )
    return total / (left.shape[0] * right.shape[0])


def class_conditional_cross_kernel_mean(
    left: torch.Tensor,
    right: torch.Tensor,
    gamma: float,
    *,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    balance: str = "class",
    chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Exact same-class cross-kernel mean.

    ``balance="class"`` gives every class present in the rollout equal weight:
    ``mean_c mean_{i in left_c, j in right_c} K(i, j)``.  ``balance="sample"``
    instead gives every left/generated row equal weight and averages its targets
    within class.  A singleton class is valid and contributes its single cross
    pair; unlike a self-kernel U-statistic, there is no diagonal to remove.
    """
    kernel = validate_kernel_name(kernel)
    chunk_size = _checked_kernel_rows(left, right, chunk_size=chunk_size)
    if balance not in {"class", "sample"}:
        raise ValueError("balance must be 'class' or 'sample'")
    left_labels = _checked_class_labels(left_labels, left, name="left_labels")
    right_labels = _checked_class_labels(right_labels, right, name="right_labels")

    left_classes, _, left_counts = torch.unique(
        left_labels, sorted=True, return_inverse=True, return_counts=True
    )
    right_classes, _, right_counts = torch.unique(
        right_labels, sorted=True, return_inverse=True, return_counts=True
    )
    right_class_positions = torch.searchsorted(right_classes, left_classes)
    right_contains_left = (
        right_class_positions.numel() == 0
        or bool(
            torch.all(right_class_positions < right_classes.numel())
            and torch.equal(
                right_classes.index_select(0, right_class_positions), left_classes
            )
        )
    )
    if not right_contains_left:
        raise ValueError(
            "right_labels must contain every class present in left_labels"
        )

    # Group rows once, then evaluate only same-class blocks.  This is equivalent
    # to a dense class mask but scales with sum_c N_left_c * N_right_c instead
    # of N_left * N_right, which matters when ``right`` is a full feature bank.
    left_is_grouped = bool(
        left_labels.numel() < 2
        or torch.all(left_labels[1:] >= left_labels[:-1])
    )
    right_is_grouped = bool(
        right_labels.numel() < 2
        or torch.all(right_labels[1:] >= right_labels[:-1])
    )
    if left_is_grouped:
        grouped_left = left
    else:
        grouped_left = left.index_select(
            0, torch.argsort(left_labels, stable=True)
        )
    if right_is_grouped:
        grouped_right = right
    else:
        grouped_right = right.index_select(
            0, torch.argsort(right_labels, stable=True)
        )

    left_class_counts = [int(value) for value in left_counts.cpu().tolist()]
    right_class_counts = [int(value) for value in right_counts.cpu().tolist()]
    right_class_starts = [0]
    for count in right_class_counts[:-1]:
        right_class_starts.append(right_class_starts[-1] + count)
    selected_right_positions = [
        int(value) for value in right_class_positions.cpu().tolist()
    ]
    total = left.new_zeros(())
    left_start = 0
    for left_count, right_position in zip(
        left_class_counts, selected_right_positions
    ):
        left_stop = left_start + left_count
        right_count = right_class_counts[right_position]
        right_start = right_class_starts[right_position]
        right_stop = right_start + right_count
        class_mean = cross_kernel_mean(
            grouped_left[left_start:left_stop],
            grouped_right[right_start:right_stop],
            gamma,
            chunk_size=chunk_size,
            kernel=kernel,
        )
        class_weight = (
            1.0 / float(left_classes.numel())
            if balance == "class"
            else left_count / float(left.shape[0])
        )
        total = total + class_weight * class_mean
        left_start = left_stop
    return total


def class_conditional_cross_kernel_partial(
    left: torch.Tensor,
    right: torch.Tensor,
    gamma: float,
    *,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    global_left_labels: torch.Tensor,
    balance: str = "class",
    chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Local-row contribution to a globally normalized same-class mean.

    ``left`` and ``left_labels`` contain only this process's generated rows,
    while ``global_left_labels`` contains the labels for the complete logical
    rollout. Summing this function's value over processes is exactly
    :func:`class_conditional_cross_kernel_mean` on the concatenated generated
    rows. This lets every process score only its local generated rows without
    changing class-balanced loss or gradient semantics.
    """
    kernel = validate_kernel_name(kernel)
    chunk_size = _checked_kernel_rows(left, right, chunk_size=chunk_size)
    if balance not in {"class", "sample"}:
        raise ValueError("balance must be 'class' or 'sample'")
    left_labels = _checked_class_labels(left_labels, left, name="left_labels")
    right_labels = _checked_class_labels(right_labels, right, name="right_labels")
    if not isinstance(global_left_labels, torch.Tensor):
        raise TypeError("global_left_labels must be a torch.Tensor")
    if global_left_labels.ndim != 1 or global_left_labels.shape[0] < 1:
        raise ValueError("global_left_labels must have non-empty shape [N]")
    if global_left_labels.device != left.device:
        raise ValueError(
            "global_left_labels must share the generated-feature device, got "
            f"{global_left_labels.device} and {left.device}"
        )
    if (
        global_left_labels.dtype == torch.bool
        or global_left_labels.is_floating_point()
        or global_left_labels.is_complex()
    ):
        raise TypeError("global_left_labels must contain integer class IDs")

    left_classes, _, left_counts = torch.unique(
        left_labels, sorted=True, return_inverse=True, return_counts=True
    )
    global_classes, global_counts = torch.unique(
        global_left_labels, sorted=True, return_counts=True
    )
    global_positions = torch.searchsorted(global_classes, left_classes)
    global_contains_left = bool(
        torch.all(global_positions < global_classes.numel())
        and torch.equal(
            global_classes.index_select(0, global_positions), left_classes
        )
    )
    if not global_contains_left:
        raise ValueError(
            "global_left_labels must contain every class in left_labels"
        )

    right_classes, right_counts = torch.unique(
        right_labels, sorted=True, return_counts=True
    )
    right_positions = torch.searchsorted(right_classes, left_classes)
    right_contains_left = bool(
        torch.all(right_positions < right_classes.numel())
        and torch.equal(
            right_classes.index_select(0, right_positions), left_classes
        )
    )
    if not right_contains_left:
        raise ValueError(
            "right_labels must contain every class present in left_labels"
        )

    left_is_grouped = bool(
        left_labels.numel() < 2
        or torch.all(left_labels[1:] >= left_labels[:-1])
    )
    right_is_grouped = bool(
        right_labels.numel() < 2
        or torch.all(right_labels[1:] >= right_labels[:-1])
    )
    grouped_left = (
        left
        if left_is_grouped
        else left.index_select(0, torch.argsort(left_labels, stable=True))
    )
    grouped_right = (
        right
        if right_is_grouped
        else right.index_select(0, torch.argsort(right_labels, stable=True))
    )

    local_counts = [int(value) for value in left_counts.cpu().tolist()]
    complete_counts = [
        int(value)
        for value in global_counts.index_select(0, global_positions).cpu().tolist()
    ]
    right_class_counts = [int(value) for value in right_counts.cpu().tolist()]
    right_class_starts = [0]
    for count in right_class_counts[:-1]:
        right_class_starts.append(right_class_starts[-1] + count)
    selected_right_positions = [
        int(value) for value in right_positions.cpu().tolist()
    ]

    total = left.new_zeros(())
    left_start = 0
    for local_count, complete_count, right_position in zip(
        local_counts, complete_counts, selected_right_positions
    ):
        left_stop = left_start + local_count
        right_count = right_class_counts[right_position]
        right_start = right_class_starts[right_position]
        right_stop = right_start + right_count
        local_class_mean = cross_kernel_mean(
            grouped_left[left_start:left_stop],
            grouped_right[right_start:right_stop],
            gamma,
            chunk_size=chunk_size,
            kernel=kernel,
        )
        class_weight = (
            local_count
            / float(complete_count * global_classes.numel())
            if balance == "class"
            else local_count / float(global_left_labels.shape[0])
        )
        total = total + class_weight * local_class_mean
        left_start = left_stop
    return total


def paired_cross_kernel_mean(
    left: torch.Tensor,
    right: torch.Tensor,
    gamma: float,
    *,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Mean kernel over aligned same-class pairs, for one NN target per row."""
    _checked_kernel_rows(left, right, chunk_size=1)
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired attraction requires equally many feature rows")
    left_labels = _checked_class_labels(left_labels, left, name="left_labels")
    right_labels = _checked_class_labels(right_labels, right, name="right_labels")
    if not torch.equal(left_labels, right_labels):
        raise ValueError("paired attraction requires aligned same-class labels")
    kernel = validate_kernel_name(kernel)
    if kernel == "rbf":
        distances = (left - right).square().sum(dim=1).clamp_min(0)
    else:
        distances = torch.linalg.vector_norm(left - right, ord=2, dim=1)
    return torch.exp(-float(gamma) * distances).mean()


def class_conditional_self_kernel_mean(
    rows: torch.Tensor,
    gamma: float,
    *,
    labels: torch.Tensor,
    chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Exact class-balanced biased self-kernel mean.

    This computes ``mean_c n_c^-2 sum_{i,j in c} K(i, j)``.  The biased
    diagonal is intentionally preserved to match the existing exact MMD.
    Consequently, a singleton class contributes ``K(x, x) == 1`` with zero
    feature gradient instead of being dropped or causing division by zero.
    """
    kernel = validate_kernel_name(kernel)
    chunk_size = _checked_kernel_rows(rows, rows, chunk_size=chunk_size)
    labels = _checked_class_labels(labels, rows, name="labels")
    classes, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )

    # The product of two row weights for class c is 1 / (C * n_c^2),
    # yielding an equal-weight mean over per-class self-kernel means.
    row_counts = counts.index_select(0, inverse).to(dtype=rows.dtype)
    row_weights = row_counts.reciprocal() / math.sqrt(float(classes.numel()))
    squared_norms = rows.square().sum(dim=1)
    total = rows.new_zeros(())
    for left_start in range(0, rows.shape[0], chunk_size):
        left_stop = min(left_start + chunk_size, rows.shape[0])
        total = total + _checkpointed_class_weighted_kernel_block_sum(
            rows[left_start:left_stop],
            rows[left_start:left_stop],
            squared_norms[left_start:left_stop],
            squared_norms[left_start:left_stop],
            labels[left_start:left_stop],
            labels[left_start:left_stop],
            row_weights[left_start:left_stop],
            row_weights[left_start:left_stop],
            gamma,
            kernel,
        )
        for right_start in range(left_stop, rows.shape[0], chunk_size):
            right_stop = min(right_start + chunk_size, rows.shape[0])
            value = _checkpointed_class_weighted_kernel_block_sum(
                rows[left_start:left_stop],
                rows[right_start:right_stop],
                squared_norms[left_start:left_stop],
                squared_norms[right_start:right_stop],
                labels[left_start:left_stop],
                labels[right_start:right_stop],
                row_weights[left_start:left_stop],
                row_weights[right_start:right_stop],
                gamma,
                kernel,
            )
            total = total + 2.0 * value
    return total


def self_kernel_mean(
    rows: torch.Tensor,
    gamma: float,
    *,
    chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Exact biased self-kernel mean, checkpointed over left-row blocks."""
    if rows.ndim != 2:
        raise ValueError(f"rows must have shape [B, D], got {tuple(rows.shape)}")
    if rows.shape[0] < 1:
        raise ValueError("self-kernel mean requires at least one row")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    chunk_size = int(chunk_size)
    kernel = validate_kernel_name(kernel)

    def block(all_rows: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        return _kernel_matrix(
            all_rows[start:stop], all_rows, gamma, kernel=kernel
        ).sum()

    total = rows.new_zeros(())
    use_checkpoint = torch.is_grad_enabled() and rows.requires_grad
    for start in range(0, rows.shape[0], chunk_size):
        stop = min(start + chunk_size, rows.shape[0])
        value = (
            checkpoint(block, rows, start, stop, use_reentrant=False)
            if use_checkpoint
            else block(rows, start, stop)
        )
        total = total + value
    return total / (rows.shape[0] * rows.shape[0])


def _tiled_self_kernel_mean(
    rows: torch.Tensor,
    gamma: float,
    *,
    chunk_size: int,
    kernel: str = "rbf",
) -> torch.Tensor:
    kernel = validate_kernel_name(kernel)
    chunk_size = _checked_kernel_rows(rows, rows, chunk_size=chunk_size)
    squared_norms = rows.square().sum(dim=1)
    total = rows.new_zeros(())
    for left_start in range(0, rows.shape[0], chunk_size):
        left_stop = min(left_start + chunk_size, rows.shape[0])
        total = total + _checkpointed_kernel_block_sum(
            rows[left_start:left_stop],
            rows[left_start:left_stop],
            squared_norms[left_start:left_stop],
            squared_norms[left_start:left_stop],
            gamma,
            kernel,
        )
        for right_start in range(left_stop, rows.shape[0], chunk_size):
            right_stop = min(right_start + chunk_size, rows.shape[0])
            value = _checkpointed_kernel_block_sum(
                rows[left_start:left_stop],
                rows[right_start:right_stop],
                squared_norms[left_start:left_stop],
                squared_norms[right_start:right_stop],
                gamma,
                kernel,
            )
            total = total + 2.0 * value
    return total / (rows.shape[0] * rows.shape[0])


def mmd_loss(
    generated: torch.Tensor,
    real: torch.Tensor,
    bandwidth: float,
    *,
    generated_labels: torch.Tensor | None = None,
    real_labels: torch.Tensor | None = None,
    attraction_real: torch.Tensor | None = None,
    attraction_real_labels: torch.Tensor | None = None,
    attraction_mode: str = "global",
    attraction_balance: str = "class",
    repulsion_mode: str = "global",
    include_generated_repulsion: bool = True,
    include_real_constant: bool = True,
    kernel_chunk_size: int = 8192,
    kernel: str = "rbf",
) -> torch.Tensor:
    """Exact biased kernel MMD² with configurable gradient terms."""
    if generated.ndim != 2 or real.ndim != 2:
        raise ValueError("generated and real must have shape [N, D] and [M, D]")
    if generated.shape[1] != real.shape[1]:
        raise ValueError("generated and real feature dimensions differ")
    rows = generated.float()
    target = real.float()
    attraction_target = (
        target if attraction_real is None else attraction_real.float()
    )
    attraction_labels = (
        real_labels
        if attraction_real is None
        else attraction_real_labels
    )
    kernel = validate_kernel_name(kernel)
    gamma = kernel_parameter_from_bandwidth(bandwidth, kernel=kernel)
    if repulsion_mode not in {"global", "within_class"}:
        raise ValueError("repulsion_mode must be 'global' or 'within_class'")
    if not isinstance(include_generated_repulsion, bool):
        raise TypeError("include_generated_repulsion must be a bool")
    if not isinstance(include_real_constant, bool):
        raise TypeError("include_real_constant must be a bool")
    if not include_generated_repulsion:
        generated_mean = rows.new_zeros(())
    elif repulsion_mode == "global":
        generated_mean = _tiled_self_kernel_mean(
            rows, gamma, chunk_size=kernel_chunk_size, kernel=kernel
        )
    elif repulsion_mode == "within_class":
        if generated_labels is None:
            raise ValueError("within_class repulsion requires generated_labels")
        generated_mean = class_conditional_self_kernel_mean(
            rows,
            gamma,
            labels=generated_labels,
            chunk_size=kernel_chunk_size,
            kernel=kernel,
        )
    real_mean = (
        _tiled_self_kernel_mean(
            target, gamma, chunk_size=kernel_chunk_size, kernel=kernel
        )
        if include_real_constant
        else rows.new_zeros(())
    )
    if attraction_mode == "global":
        cross_mean = cross_kernel_mean(
            rows,
            attraction_target,
            gamma,
            chunk_size=kernel_chunk_size,
            kernel=kernel,
        )
    elif attraction_mode == "within_class":
        if generated_labels is None or attraction_labels is None:
            raise ValueError("within_class attraction requires generated/real labels")
        cross_mean = class_conditional_cross_kernel_mean(
            rows,
            attraction_target,
            gamma,
            left_labels=generated_labels,
            right_labels=attraction_labels,
            balance=attraction_balance,
            chunk_size=kernel_chunk_size,
            kernel=kernel,
        )
    elif attraction_mode == "paired":
        if generated_labels is None or attraction_labels is None:
            raise ValueError("paired attraction requires generated/real labels")
        cross_mean = paired_cross_kernel_mean(
            rows,
            attraction_target,
            gamma,
            left_labels=generated_labels,
            right_labels=attraction_labels,
            kernel=kernel,
        )
    else:
        raise ValueError(
            "attraction_mode must be 'global', 'within_class', or 'paired'"
        )
    return generated_mean + real_mean - 2.0 * cross_mean


__all__ = [
    "class_conditional_cross_kernel_mean",
    "class_conditional_cross_kernel_partial",
    "class_conditional_self_kernel_mean",
    "cross_kernel_mean",
    "gamma_from_bandwidth",
    "gaussian_kernel",
    "kernel_parameter_from_bandwidth",
    "mmd_loss",
    "paired_cross_kernel_mean",
    "self_kernel_mean",
    "squared_distance",
    "SUPPORTED_KERNELS",
    "validate_kernel_name",
]
