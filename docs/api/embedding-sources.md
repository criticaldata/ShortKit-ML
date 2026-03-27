# Embedding Sources API

Embedding sources allow shortcut detection without direct model access.

## Class Reference

### EmbeddingSource

::: shortcut_detect.embedding_sources.EmbeddingSource
    options:
      show_root_heading: true

### HuggingFaceEmbeddingSource

::: shortcut_detect.embedding_sources.HuggingFaceEmbeddingSource
    options:
      show_root_heading: true

### CallableEmbeddingSource

::: shortcut_detect.embedding_sources.CallableEmbeddingSource
    options:
      show_root_heading: true

## EmbeddingSource (Base)

Abstract base class for embedding generators.

```python
from shortcut_detect.embedding_sources import EmbeddingSource

class MyEmbeddingSource(EmbeddingSource):
    def embed(self, inputs: list) -> np.ndarray:
        # Return embeddings for inputs
        return embeddings

    @property
    def name(self) -> str:
        return "my_source"
```

## HuggingFaceEmbeddingSource

Generate embeddings using HuggingFace transformers.

### Constructor

```python
HuggingFaceEmbeddingSource(
    model_name: str,
    pooling: str = 'mean',
    batch_size: int = 32,
    device: str = None,
    max_length: int = 512
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | HuggingFace model name |
| `pooling` | str | 'mean' | Pooling strategy |
| `batch_size` | int | 32 | Batch size for encoding |
| `device` | str | None | Device (auto-detected) |
| `max_length` | int | 512 | Maximum sequence length |

### Pooling Options

| Value | Description |
|-------|-------------|
| `'mean'` | Mean of all token embeddings |
| `'cls'` | [CLS] token embedding |
| `'max'` | Max pooling over tokens |
| `'last'` | Last token embedding |

### Usage

```python
from shortcut_detect import HuggingFaceEmbeddingSource

source = HuggingFaceEmbeddingSource(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pooling="mean",
    batch_size=64
)

texts = ["Sample text 1", "Sample text 2", ...]
embeddings = source.embed(texts)
print(embeddings.shape)  # (n_samples, 384)
```

### With ShortcutDetector

```python
from shortcut_detect import ShortcutDetector, HuggingFaceEmbeddingSource

source = HuggingFaceEmbeddingSource("bert-base-uncased")

detector = ShortcutDetector(methods=['probe', 'statistical'])
detector.fit(
    embeddings=None,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=source,
    embedding_cache_path="cached_embeddings.npy"
)
```

---

## CallableEmbeddingSource

Wrap any function as an embedding source.

### Constructor

```python
CallableEmbeddingSource(
    embed_fn: Callable[[list], np.ndarray],
    name: str = "callable"
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embed_fn` | callable | Function taking list, returning ndarray |
| `name` | str | Name for logging |

### Usage

```python
from shortcut_detect import CallableEmbeddingSource
import numpy as np

# Wrap external API
def my_embedding_api(texts):
    # Call your API
    response = external_client.embed(texts)
    return np.array(response["embeddings"])

source = CallableEmbeddingSource(
    embed_fn=my_embedding_api,
    name="my_api"
)

embeddings = source.embed(["text1", "text2"])
```

### With Batching

```python
def batched_api(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = external_api.embed(batch)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

source = CallableEmbeddingSource(
    embed_fn=lambda x: batched_api(x, batch_size=64),
    name="batched_api"
)
```

## Caching

All embedding sources support caching:

```python
detector.fit(
    embeddings=None,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=source,
    embedding_cache_path="embeddings.npy"  # Cache here
)

# Second run loads from cache
detector2.fit(
    embeddings=None,
    group_labels=groups,
    raw_inputs=texts,
    embedding_source=source,
    embedding_cache_path="embeddings.npy"  # Loaded from cache
)
```

## See Also

- [Quick Start - Embedding Mode](../getting-started/quickstart.md#embedding-only-mode)
- [ShortcutDetector API](shortcut-detector.md)
