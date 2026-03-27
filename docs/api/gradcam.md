# GradCAM API

The GradCAM module provides visual shortcut detection for image models.

## Class Reference

::: shortcut_detect.gradcam.GradCAMHeatmapGenerator
    options:
      show_root_heading: true
      show_source: true

## GradCAMHeatmapGenerator

### Constructor

```python
GradCAMHeatmapGenerator(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    device: str = 'cuda',
    use_guided: bool = False
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | nn.Module | required | PyTorch model |
| `target_layer` | nn.Module | required | Layer for GradCAM |
| `device` | str | 'cuda' | Computation device |
| `use_guided` | bool | False | Use Guided GradCAM |

## Methods

### generate()

```python
def generate(
    input_tensor: torch.Tensor,
    target_class: int = None
) -> np.ndarray
```

Generate GradCAM heatmap for a single input.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tensor` | Tensor | Shape (C, H, W) or (1, C, H, W) |
| `target_class` | int | Class to explain (None = predicted) |

**Returns:** Heatmap array (H, W)

### generate_batch()

```python
def generate_batch(
    inputs: torch.Tensor,
    target_classes: list[int] = None
) -> list[np.ndarray]
```

Generate heatmaps for a batch of inputs.

### visualize()

```python
def visualize(
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet',
    save_path: str = None
) -> np.ndarray
```

Overlay heatmap on input image.

### compare_groups()

```python
def compare_groups(
    heatmaps: np.ndarray,
    group_labels: np.ndarray
) -> AttentionOverlapResult
```

Compare attention patterns between groups.

**Returns:** AttentionOverlapResult dataclass

## AttentionOverlapResult

```python
@dataclass
class AttentionOverlapResult:
    overlap_score: float        # Attention overlap (0-1)
    group_heatmaps: dict        # Average heatmap per group
    divergence_regions: ndarray # Regions with different attention
    summary: str                # Human-readable summary
```

## Usage Examples

### Basic Usage

```python
from shortcut_detect import GradCAMHeatmapGenerator
import torch

model = torch.load("model.pth")
target_layer = model.layer4[-1]

gradcam = GradCAMHeatmapGenerator(model, target_layer)

heatmap = gradcam.generate(image_tensor)
gradcam.visualize(image_tensor, heatmap, save_path="attention.png")
```

### Group Comparison

```python
# Generate heatmaps for all images
heatmaps = []
for img in images:
    heatmaps.append(gradcam.generate(img))
heatmaps = np.stack(heatmaps)

# Compare groups
result = gradcam.compare_groups(heatmaps, group_labels)
print(f"Overlap: {result.overlap_score:.2f}")
print(result.summary)
```

### Batch Processing

```python
from torch.utils.data import DataLoader

all_heatmaps = []
for batch in DataLoader(dataset, batch_size=32):
    images, labels = batch
    heatmaps = gradcam.generate_batch(images.cuda(), labels)
    all_heatmaps.extend(heatmaps)
```

## Layer Selection Tips

```python
# ResNet
target_layer = model.layer4[-1]

# VGG
target_layer = model.features[-1]

# DenseNet
target_layer = model.features.denseblock4

# EfficientNet
target_layer = model.features[-1]
```

## See Also

- [GradCAM Guide](../methods/gradcam.md)
- [ShortcutDetector API](shortcut-detector.md)
