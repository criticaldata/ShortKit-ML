"""Base types for pluggable overall risk scoring conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConditionContext:
    """Normalized detector output passed to overall assessment conditions."""

    methods: list[str]
    results: dict[str, dict[str, Any]]


class RiskCondition(ABC):
    """Aggregate method-level outputs into a human-readable overall assessment."""

    name: str

    @abstractmethod
    def assess(self, ctx: ConditionContext) -> str:
        """Return the overall assessment string."""
        raise NotImplementedError
