"""Abstract base class for detector builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def resolve_target(
    labels: np.ndarray, group_labels: np.ndarray | None, mode: str = "attribute"
) -> tuple[np.ndarray, str]:
    """Pick the label a label-predicting detector should be fitted on.

    HBAC, Frequency and SIS fit a model to a label. A shortcut audit asks about
    the attribute A, so by default they target ``group_labels`` when it is given
    and differs from the task labels. ``mode="task"`` restores the submitted
    behaviour (fit on the task label and never see A).
    """
    if mode not in {"attribute", "task"}:
        raise ValueError("target mode must be 'attribute' or 'task'")
    if mode == "task" or group_labels is None:
        return np.asarray(labels), "task"
    g = np.asarray(group_labels)
    if g.shape[0] != np.asarray(labels).shape[0]:
        raise ValueError("group_labels must have one entry per sample")
    if np.array_equal(g, np.asarray(labels)):
        return np.asarray(labels), "task"
    if not np.issubdtype(g.dtype, np.number):
        g = np.unique(g, return_inverse=True)[1].astype(np.int64)
    return g, "attribute"


class BaseDetector(ABC):
    """Abstract base for detector builders."""

    def __init__(self, seed: int, kwargs: dict[str, Any] | None = None, method: str = ""):
        self.seed = seed
        self.kwargs = dict(kwargs or {})
        self.method = method

    @abstractmethod
    def build(self) -> Any:
        """Return the configured detector instance."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names: list[str] | None,
        protected_labels: np.ndarray | None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Execute the detector workflow and return result metadata."""

    def run_from_loader(
        self,
        loader: Any,
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Optional override: execute detection using a user-provided loader."""
        data = loader() if callable(loader) else loader

        if isinstance(data, dict):
            embeddings = data.get("embeddings")
            labels = data.get("labels")
            if embeddings is None or labels is None:
                raise ValueError(
                    f"Loader for method '{self.method}' must provide 'embeddings' and 'labels'."
                )
            group_labels = data.get("group_labels", labels)
            protected = data.get("protected_labels")
            if protected is None:
                protected = protected_labels if protected_labels is not None else group_labels
            return self.run(
                embeddings=embeddings,
                labels=labels,
                group_labels=group_labels,
                feature_names=data.get("feature_names", feature_names),
                protected_labels=protected,
                splits=data.get("splits", splits),
                extra_labels=data.get("extra_labels", extra_labels),
            )

        if not hasattr(data, "__iter__"):
            raise ValueError(
                f"Loader for method '{self.method}' must be a dict or an iterable of batches."
            )

        def to_numpy(value):
            if isinstance(value, np.ndarray):
                return value
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy()
            return np.asarray(value)

        embeddings_batches = []
        labels_batches = []
        group_batches = []
        for batch in data:
            if isinstance(batch, dict):
                x = batch.get("embeddings")
                if x is None:
                    x = batch.get("x")
                y = batch.get("labels")
                if y is None:
                    y = batch.get("y")
                g = batch.get("group_labels")
                if g is None:
                    g = batch.get("g")
            else:
                if len(batch) == 2:
                    x, y = batch
                    g = None
                elif len(batch) == 3:
                    x, y, g = batch
                else:
                    raise ValueError(
                        f"Loader batches for '{self.method}' must be (x, y) or (x, y, g)."
                    )
            embeddings_batches.append(to_numpy(x))
            labels_batches.append(to_numpy(y))
            if g is not None:
                group_batches.append(to_numpy(g))

        embeddings = np.concatenate(embeddings_batches, axis=0)
        labels = np.concatenate(labels_batches, axis=0)
        group_labels = np.concatenate(group_batches, axis=0) if group_batches else labels

        return self.run(
            embeddings=embeddings,
            labels=labels,
            group_labels=group_labels,
            feature_names=feature_names,
            protected_labels=protected_labels,
            splits=splits,
            extra_labels=extra_labels,
        )
