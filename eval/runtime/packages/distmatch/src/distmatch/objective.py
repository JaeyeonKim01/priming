from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from .aggregation import LossAggregator, NamedLoss


class StateUpdate(Protocol):
    """Internal staged transition; apply must not fail after validate succeeds."""

    def validate(self) -> None: ...

    def apply(self) -> None: ...


@dataclass(frozen=True)
class TermResult:
    raw_loss: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    staged_update: StateUpdate | None = None


class DistributionObjective(Protocol):
    def score(self, features: torch.Tensor) -> TermResult: ...


@dataclass(frozen=True)
class LossTerm:
    name: str
    feature_name: str
    objective: DistributionObjective
    weight: float = 1.0


@dataclass(frozen=True)
class CompositeResult:
    loss: torch.Tensor
    raw: dict[str, torch.Tensor]
    normalized: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


class CompositeDistributionObjective:
    """Estimator-neutral scoring and one-pending-update lifecycle."""

    def __init__(self, terms: list[LossTerm], *, aggregator: LossAggregator) -> None:
        self.terms = tuple(terms)
        if not self.terms:
            raise ValueError("CompositeDistributionObjective requires at least one term")
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError("LossTerm.name values must be unique")
        self.aggregator = aggregator
        self._awaiting_update = False
        self._updates: tuple[StateUpdate, ...] = ()

    @property
    def has_pending_update(self) -> bool:
        return self._awaiting_update

    def score_features(self, features: dict[str, torch.Tensor]) -> CompositeResult:
        if self._awaiting_update:
            raise RuntimeError("update_state() must be called before scoring another batch")
        missing = sorted({term.feature_name for term in self.terms} - set(features))
        if missing:
            raise KeyError(f"missing feature tensors for {missing}")
        results = []
        updates = []
        metrics = {}
        for term in self.terms:
            result = term.objective.score(features[term.feature_name])
            if result.raw_loss.ndim != 0:
                raise ValueError(f"objective term {term.name!r} returned a non-scalar loss")
            results.append((term, result))
            if result.staged_update is not None:
                updates.append(result.staged_update)
            metrics.update(
                {f"{term.name}/{key}": value.detach() for key, value in result.metrics.items()}
            )
        named = tuple(
            NamedLoss(term.name, result.raw_loss, term.weight) for term, result in results
        )
        loss, normalized = self.aggregator(named)
        self._updates = tuple(updates)
        self._awaiting_update = True
        return CompositeResult(
            loss=loss,
            raw={term.name: result.raw_loss.detach() for term, result in results},
            normalized={name: value.detach() for name, value in normalized.items()},
            metrics=metrics,
        )

    def update_state(self) -> None:
        if not self._awaiting_update:
            raise RuntimeError("update_state() requires exactly one preceding objective forward")
        for update in self._updates:
            update.validate()
        for update in self._updates:
            update.apply()
        self._updates = ()
        self._awaiting_update = False


__all__ = [
    "CompositeDistributionObjective",
    "CompositeResult",
    "DistributionObjective",
    "LossTerm",
    "StateUpdate",
    "TermResult",
]
