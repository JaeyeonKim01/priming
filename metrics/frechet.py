from __future__ import annotations

import torch


def _checked_nonnegative(eigenvalues: torch.Tensor, *, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(eigenvalues).all().item()):
        raise FloatingPointError(f"{label} eigenspectrum contains NaN or Inf")
    scale = max(1.0, float(eigenvalues.detach().abs().max().item()))
    tolerance = 1e-8 * scale * max(1, eigenvalues.numel())
    minimum = float(eigenvalues.detach().min().item())
    if minimum < -tolerance:
        raise FloatingPointError(
            f"{label} has a materially negative eigenvalue {minimum:.6g} "
            f"(tolerance {tolerance:.6g})"
        )
    return eigenvalues.clamp_min(0)


@torch.no_grad()
def covariance_sqrt(covariance: torch.Tensor) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = _checked_nonnegative(eigenvalues, label="reference covariance")
    output = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    return 0.5 * (output + output.T)


def frechet_distance(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_covariance: torch.Tensor,
    *,
    reference_sqrt: torch.Tensor | None = None,
    mean_weight: float = 1.0,
) -> torch.Tensor:
    if mean.ndim != 1 or covariance.shape != (mean.numel(), mean.numel()):
        raise ValueError("invalid generated mean/covariance shapes")
    if reference_mean.shape != mean.shape or reference_covariance.shape != covariance.shape:
        raise ValueError("generated and reference Gaussian dimensions differ")
    covariance = 0.5 * (covariance + covariance.T)
    if reference_sqrt is None:
        reference_sqrt = covariance_sqrt(reference_covariance)
    matrix = reference_sqrt @ covariance @ reference_sqrt
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = _checked_nonnegative(
        torch.linalg.eigvalsh(matrix), label="Fréchet covariance sandwich"
    )
    trace_sqrt = eigenvalues.sqrt().sum()
    difference = mean - reference_mean
    output = (
        float(mean_weight) * difference.square().sum()
        + covariance.diagonal().sum()
        + reference_covariance.diagonal().sum()
        - 2.0 * trace_sqrt
    )
    if not bool(torch.isfinite(output.detach()).item()):
        raise FloatingPointError("Fréchet distance is NaN or Inf")
    return output


__all__ = ["covariance_sqrt", "frechet_distance"]
