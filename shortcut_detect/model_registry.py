"""
Registry for embedding models. Enables modular integration of new models
for model comparison and benchmarking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .embedding_sources import EmbeddingSource, HuggingFaceEmbeddingSource


def _make_hf_factory(model_name: str) -> Callable[..., EmbeddingSource]:
    """Create a factory that returns HuggingFaceEmbeddingSource for a given model."""

    def factory(**kwargs: Any) -> EmbeddingSource:
        return HuggingFaceEmbeddingSource(model_name=model_name, **kwargs)

    return factory


# Pre-populated models for benchmarking (id -> (factory, display_name, description))
_DEFAULT_MODELS: dict[str, tuple[Callable[..., EmbeddingSource], str, str]] = {
    "sentence-transformers/all-MiniLM-L6-v2": (
        _make_hf_factory("sentence-transformers/all-MiniLM-L6-v2"),
        "MiniLM-L6-v2",
        "Lightweight sentence transformer, 384-dim",
    ),
    "sentence-transformers/all-mpnet-base-v2": (
        _make_hf_factory("sentence-transformers/all-mpnet-base-v2"),
        "MPNet-base-v2",
        "Higher quality sentence transformer, 768-dim",
    ),
    "bert-base-uncased": (
        _make_hf_factory("bert-base-uncased"),
        "BERT-base",
        "BERT base uncased, 768-dim",
    ),
    "roberta-base": (
        _make_hf_factory("roberta-base"),
        "RoBERTa-base",
        "RoBERTa base, 768-dim",
    ),
    "distilbert-base-uncased": (
        _make_hf_factory("distilbert-base-uncased"),
        "DistilBERT-base",
        "Distilled BERT, 768-dim",
    ),
}


class EmbeddingModelRegistry:
    """
    Registry for embedding models. Supports registration of embedding sources
    and instantiation by ID. New models integrate by calling register().
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Callable[..., EmbeddingSource], str, str]] = dict(
            _DEFAULT_MODELS
        )

    def register(
        self,
        model_id: str,
        factory_fn: Callable[..., EmbeddingSource],
        display_name: str | None = None,
        description: str | None = "",
    ) -> None:
        """
        Register an embedding model.

        Args:
            model_id: Unique identifier for the model.
            factory_fn: Callable that returns an EmbeddingSource. May accept **kwargs.
            display_name: Human-readable name (defaults to model_id).
            description: Optional description.
        """
        self._entries[model_id] = (
            factory_fn,
            display_name or model_id,
            description or "",
        )

    def list_registered(self) -> list[tuple[str, str, str]]:
        """
        Return list of registered models as (id, display_name, description).
        """
        return [
            (model_id, display_name, desc)
            for model_id, (_, display_name, desc) in self._entries.items()
        ]

    def create(self, model_id: str, **kwargs: Any) -> EmbeddingSource:
        """
        Instantiate an EmbeddingSource by model ID.

        Args:
            model_id: Registered model ID or a HuggingFace model name for pass-through.
            **kwargs: Passed to the factory if it accepts them.

        Returns:
            EmbeddingSource instance.

        Raises:
            KeyError: If model_id is not registered and not a valid pass-through.
        """
        if model_id in self._entries:
            factory, _, _ = self._entries[model_id]
            return factory(**kwargs)
        # Pass-through: treat as custom HuggingFace model
        return HuggingFaceEmbeddingSource(model_name=model_id, **kwargs)

    def is_registered(self, model_id: str) -> bool:
        """Check if model_id is in the registry (explicit registration)."""
        return model_id in self._entries


# Global registry instance
_default_registry = EmbeddingModelRegistry()


def list_embedding_models() -> list[tuple[str, str, str]]:
    """Convenience: list models from the default registry."""
    return _default_registry.list_registered()


def get_embedding_registry() -> EmbeddingModelRegistry:
    """Return the default embedding model registry."""
    return _default_registry
