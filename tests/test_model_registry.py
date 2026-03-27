"""Tests for embedding model registry."""

from shortcut_detect.embedding_sources import EmbeddingSource, HuggingFaceEmbeddingSource
from shortcut_detect.model_registry import (
    EmbeddingModelRegistry,
    list_embedding_models,
)


def test_list_registered():
    """Registry lists pre-populated models."""
    models = list_embedding_models()
    assert len(models) >= 4
    ids = [m[0] for m in models]
    assert "sentence-transformers/all-MiniLM-L6-v2" in ids
    assert "bert-base-uncased" in ids


def test_registry_register_and_create():
    """Can register and create custom models."""
    registry = EmbeddingModelRegistry()
    # Clear defaults for isolated test
    registry._entries.clear()

    def _make_dummy():
        return HuggingFaceEmbeddingSource(model_name="bert-base-uncased")

    registry.register("dummy", _make_dummy, display_name="Dummy", description="Test")
    assert registry.is_registered("dummy")
    source = registry.create("dummy")
    assert isinstance(source, EmbeddingSource)


def test_registry_create_pass_through():
    """Unregistered HuggingFace ID creates HuggingFaceEmbeddingSource."""
    registry = EmbeddingModelRegistry()
    source = registry.create("custom/org/my-model")
    assert isinstance(source, HuggingFaceEmbeddingSource)
    assert source.model_name == "custom/org/my-model"
