from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class NamedLoss:
    name: str
    value: torch.Tensor
    weight: float


class LossAggregator(Protocol):
    def __call__(
        self, terms: tuple[NamedLoss, ...]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


class RawWeightedSum:
    def __call__(
        self, terms: tuple[NamedLoss, ...]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not terms:
            raise ValueError("cannot aggregate an empty term list")
        normalized = {term.name: term.value for term in terms}
        loss = sum(
            (term.weight * term.value for term in terms),
            start=terms[0].value.new_zeros(()),
        )
        return loss, normalized


class DetachedValueWeightedSum:
    def __init__(self, eps: float) -> None:
        if eps <= 0:
            raise ValueError("detached-value normalization requires eps > 0")
        self.eps = float(eps)

    def __call__(
        self, terms: tuple[NamedLoss, ...]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not terms:
            raise ValueError("cannot aggregate an empty term list")
        normalized = {
            term.name: term.value / (term.value.detach() + self.eps) for term in terms
        }
        loss = sum(
            (term.weight * normalized[term.name] for term in terms),
            start=terms[0].value.new_zeros(()),
        )
        return loss, normalized


class DetachedAbsoluteValueWeightedSum:
    def __init__(self, eps: float) -> None:
        if eps <= 0:
            raise ValueError("detached-absolute-value normalization requires eps > 0")
        self.eps = float(eps)

    def __call__(
        self, terms: tuple[NamedLoss, ...]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not terms:
            raise ValueError("cannot aggregate an empty term list")
        normalized = {
            term.name: term.value / (term.value.detach().abs() + self.eps)
            for term in terms
        }
        loss = sum(
            (term.weight * normalized[term.name] for term in terms),
            start=terms[0].value.new_zeros(()),
        )
        return loss, normalized


__all__ = [
    "DetachedAbsoluteValueWeightedSum",
    "DetachedValueWeightedSum",
    "LossAggregator",
    "NamedLoss",
    "RawWeightedSum",
]
