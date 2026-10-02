from __future__ import annotations

from typing import Any, Protocol

import torch


class Metric(Protocol):
    def reset(self) -> None: ...

    def update(self, values: torch.Tensor) -> None: ...

    def compute(self) -> Any: ...


__all__ = ["Metric"]
