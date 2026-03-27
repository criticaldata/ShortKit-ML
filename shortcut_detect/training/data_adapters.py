"""Utilities for adapting dataset/loader batches into canonical torch training tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class DataSpec:
    """Optional schema/size hints for loader-native training flows."""

    n_features: int
    n_classes: int
    n_groups: int | None = None
    group_ids: np.ndarray | None = None
    train_size: int | None = None
    val_size: int | None = None


def resolve_data_spec(data_spec: DataSpec | Mapping[str, Any] | None) -> DataSpec | None:
    """Normalize dict-or-dataclass specs into DataSpec."""
    if data_spec is None:
        return None
    if isinstance(data_spec, DataSpec):
        return data_spec
    if not isinstance(data_spec, Mapping):
        raise TypeError("data_spec must be a DataSpec or mapping.")
    group_ids = data_spec.get("group_ids")
    group_ids_arr = None if group_ids is None else np.asarray(group_ids, dtype=np.int64)
    return DataSpec(
        n_features=int(data_spec["n_features"]),
        n_classes=int(data_spec["n_classes"]),
        n_groups=None if data_spec.get("n_groups") is None else int(data_spec["n_groups"]),
        group_ids=group_ids_arr,
        train_size=None if data_spec.get("train_size") is None else int(data_spec["train_size"]),
        val_size=None if data_spec.get("val_size") is None else int(data_spec["val_size"]),
    )


def is_iterable_dataset(obj: Any) -> bool:
    """Return True when obj is a torch IterableDataset."""
    return isinstance(obj, IterableDataset)


def safe_len(obj: Any) -> int | None:
    """Best-effort length extraction."""
    try:
        return int(len(obj))
    except Exception:
        return None


def _pick_from_mapping(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _as_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x).float()


def _as_long_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long().view(-1)
    return torch.as_tensor(x).long().view(-1)


def extract_xy_batch(
    batch: Any,
    *,
    target_extractor: Callable[[Any], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (x, y) tensors from tuple/list/dict batches."""
    if target_extractor is not None:
        extracted = target_extractor(batch)
        if isinstance(extracted, tuple | list) and len(extracted) == 2:
            x, y = extracted
        else:
            x = (
                _pick_from_mapping(batch, ("embeddings", "x", "features"))
                if isinstance(batch, Mapping)
                else batch[0]
            )
            y = extracted
        return _as_float_tensor(x), _as_long_tensor(y)

    if isinstance(batch, Mapping):
        x = _pick_from_mapping(batch, ("embeddings", "x", "features"))
        y = _pick_from_mapping(batch, ("labels", "y", "target"))
        if x is None or y is None:
            raise ValueError("Batch dict must include embedding/features and labels/target fields.")
        return _as_float_tensor(x), _as_long_tensor(y)

    if isinstance(batch, tuple | list) and len(batch) >= 2:
        x, y = batch[0], batch[1]
        return _as_float_tensor(x), _as_long_tensor(y)

    raise ValueError("Batch must be mapping or tuple/list with at least 2 entries (x, y).")


def extract_xyg_batch(
    batch: Any,
    *,
    target_extractor: Callable[[Any], Any] | None = None,
    group_extractor: Callable[[Any], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (x, y, g) tensors from tuple/list/dict batches."""
    if isinstance(batch, Mapping):
        x = _pick_from_mapping(batch, ("embeddings", "x", "features"))
        y = (
            target_extractor(batch)
            if target_extractor is not None
            else _pick_from_mapping(batch, ("labels", "y", "target"))
        )
        g = (
            group_extractor(batch)
            if group_extractor is not None
            else _pick_from_mapping(batch, ("group_labels", "g", "groups"))
        )
        if x is None or y is None or g is None:
            raise ValueError(
                "Batch dict must include x/embeddings, y/labels, and g/group_labels (or provide extractors)."
            )
        return _as_float_tensor(x), _as_long_tensor(y), _as_long_tensor(g)

    if isinstance(batch, tuple | list) and len(batch) >= 3:
        x = batch[0]
        y = target_extractor(batch) if target_extractor is not None else batch[1]
        g = group_extractor(batch) if group_extractor is not None else batch[2]
        return _as_float_tensor(x), _as_long_tensor(y), _as_long_tensor(g)

    raise ValueError("Batch must be mapping or tuple/list with at least 3 entries (x, y, g).")
