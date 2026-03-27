"""Shared DataLoader hook utilities for torch-based detectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from torch.utils.data import DataLoader

LoaderStage = Literal[
    "train",
    "val",
    "eval",
    "predict",
    "full",
    "dl_train",
    "dl_val",
    "du_train",
    "du_eval",
    "du_bullet",
]


@dataclass(frozen=True)
class LoaderRequest:
    """Canonical request object used to build a stage-specific DataLoader."""

    stage: LoaderStage | str
    dataset: Any
    batch_size: int
    shuffle: bool
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


LoaderFactory = Callable[[LoaderRequest], DataLoader]
StageLoaderOverrides = Mapping[str, Mapping[str, Any]] | None


def _apply_stage_overrides(
    request: LoaderRequest,
    stage_loader_overrides: StageLoaderOverrides,
) -> LoaderRequest:
    if not stage_loader_overrides:
        return request

    stage_overrides = dict(stage_loader_overrides.get(str(request.stage), {}))
    if not stage_overrides:
        return request

    known_keys = {
        "batch_size",
        "shuffle",
        "num_workers",
        "pin_memory",
        "drop_last",
    }
    known = {k: stage_overrides.pop(k) for k in list(stage_overrides.keys()) if k in known_keys}
    extra = dict(request.extra_kwargs)
    extra.update(stage_overrides)

    return LoaderRequest(
        stage=request.stage,
        dataset=request.dataset,
        batch_size=int(known.get("batch_size", request.batch_size)),
        shuffle=bool(known.get("shuffle", request.shuffle)),
        num_workers=int(known.get("num_workers", request.num_workers)),
        pin_memory=bool(known.get("pin_memory", request.pin_memory)),
        drop_last=bool(known.get("drop_last", request.drop_last)),
        extra_kwargs=extra,
    )


def build_loader(
    request: LoaderRequest,
    *,
    loader_factory: LoaderFactory | None = None,
    stage_loader_overrides: StageLoaderOverrides = None,
) -> DataLoader:
    """Build a DataLoader using optional stage overrides and an optional factory hook."""
    resolved = _apply_stage_overrides(request, stage_loader_overrides)
    if loader_factory is not None:
        return loader_factory(resolved)

    return DataLoader(
        resolved.dataset,
        batch_size=resolved.batch_size,
        shuffle=resolved.shuffle,
        num_workers=resolved.num_workers,
        pin_memory=resolved.pin_memory,
        drop_last=resolved.drop_last,
        **resolved.extra_kwargs,
    )
