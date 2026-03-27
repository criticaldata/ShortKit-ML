"""Probe-based detection methods."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "ProbeDetectorFactory": ".probe_factory",
    "ProbeFactoryContext": ".probe_factory",
    "SKLearnProbe": ".sklearn_probe",
    "TorchProbe": ".torch_probe",
    "evaluate_probe_cv": ".pipeline",
    "train_test_pipeline": ".pipeline",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
