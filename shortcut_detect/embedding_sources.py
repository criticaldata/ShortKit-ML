"""
Helpers for generating embeddings when only inference access is available.

Provides abstraction layers for Hugging Face models or arbitrary embedding
functions so the rest of the library can stay agnostic to how embeddings are
produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyTorch is required for HuggingFaceEmbeddingSource. "
            "Install a compatible torch build for your platform."
        ) from exc
    return torch


class EmbeddingSource(ABC):
    """Abstract base class for embedding generators."""

    def __init__(self, name: str = "embedding_source"):
        self.name = name

    @abstractmethod
    def generate(self, inputs: Sequence[Any]) -> np.ndarray:
        """
        Generate embeddings for a sequence of inputs.

        Args:
            inputs: Sequence of raw inputs (text, images, etc.)

        Returns:
            np.ndarray of shape (n_samples, embedding_dim)
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name})"


def _batch_iterator(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    """Yield items in fixed-size batches."""
    current: list[Any] = []
    for item in items:
        current.append(item)
        if len(current) == batch_size:
            yield current
            current = []
    if current:
        yield current


class CallableEmbeddingSource(EmbeddingSource):
    """
    Wrap an arbitrary callable so it can be used as an embedding source.

    The callable should accept a sequence of inputs and return a 2D numpy array.
    This is useful for production or closed-source APIs where only inference
    access is available.
    """

    def __init__(self, fn: Callable[[Sequence[Any]], np.ndarray], name: str = "callable_source"):
        super().__init__(name=name)
        self.fn = fn

    def generate(self, inputs: Sequence[Any]) -> np.ndarray:
        outputs = self.fn(inputs)
        arr = np.asarray(outputs)
        if arr.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D, got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)


class HuggingFaceEmbeddingSource(EmbeddingSource):
    """
    Generate embeddings from any Hugging Face transformer model without
    requiring gradient access.
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        pooling: str = "cls",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: Name or path of the Hugging Face model to load.
            tokenizer_name: Optional tokenizer name (defaults to model_name).
            device: Device string ("cpu", "cuda", etc.). Auto-detect if None.
            batch_size: Batch size for inference.
            pooling: "cls" or "mean" pooling strategy.
            normalize: Whether to L2-normalize embeddings.
        """
        super().__init__(name=f"huggingface:{model_name}")
        torch = _require_torch()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.batch_size = batch_size
        self.pooling = pooling
        self.normalize = normalize
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "transformers is required for HuggingFaceEmbeddingSource. "
                    "Install with `uv pip install transformers` or "
                    "`pip install shortcut-detect[hf]`."
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def _pool(self, hidden_states, attention_mask):
        torch = _require_torch()
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling strategy '{self.pooling}'")
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled

    def generate(self, inputs: Sequence[str]) -> np.ndarray:
        if not isinstance(inputs, Sequence):
            raise TypeError("inputs must be a sequence of strings")
        self._ensure_model()
        assert self._model is not None and self._tokenizer is not None

        all_embeddings: list[np.ndarray] = []
        torch = _require_torch()
        with torch.no_grad():
            for batch in _batch_iterator(inputs, self.batch_size):
                encoded = self._tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"])
                all_embeddings.append(pooled.cpu().numpy())
        if not all_embeddings:
            raise ValueError("No inputs provided to HuggingFaceEmbeddingSource")
        return np.vstack(all_embeddings)


__all__ = ["EmbeddingSource", "CallableEmbeddingSource", "HuggingFaceEmbeddingSource"]
