"""Registry for pluggable overall risk scoring conditions."""

from __future__ import annotations

from collections.abc import Callable

from .base import RiskCondition

_CONDITION_REGISTRY: dict[str, type[RiskCondition]] = {}


def register_condition(
    name: str,
) -> Callable[[type[RiskCondition]], type[RiskCondition]]:
    """Decorator to register a condition class by name."""

    def decorator(cls: type[RiskCondition]) -> type[RiskCondition]:
        if name in _CONDITION_REGISTRY:
            raise ValueError(f"Condition already registered: {name}")
        cls.name = name
        _CONDITION_REGISTRY[name] = cls
        return cls

    return decorator


def create_condition(name: str, **kwargs) -> RiskCondition:
    """Instantiate a registered condition by name."""
    condition_cls = _CONDITION_REGISTRY.get(name)
    if condition_cls is None:
        available = ", ".join(sorted(_CONDITION_REGISTRY))
        raise ValueError(f"Unknown condition '{name}'. Available conditions: {available}")
    return condition_cls(**kwargs)


def available_conditions() -> dict[str, type[RiskCondition]]:
    """Return a copy of the registered conditions."""
    return dict(_CONDITION_REGISTRY)
