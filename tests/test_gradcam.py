import numpy as np
import torch
import torch.nn as nn

from shortcut_detect.gradcam import AttentionOverlapResult, GradCAMHeatmapGenerator


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        feats = self.conv(x)
        pooled = self.pool(feats).view(x.shape[0], -1)
        logits = self.head(pooled)
        return {
            "disease": logits[:, :2],
            "attribute": logits[:, 2:],
            "feats": feats,
        }


class TupleOutputModel(nn.Module):
    """Model that outputs tuple instead of dict."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(4, 2)
        self.head2 = nn.Linear(4, 2)

    def forward(self, x):
        feats = self.conv(x)
        pooled = feats.mean(dim=(2, 3))
        return (self.head1(pooled), self.head2(pooled))


def test_gradcam_heatmap_shape_and_range():
    torch.manual_seed(0)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    inputs = torch.randn(2, 1, 16, 16)
    heatmap = generator.generate_heatmap(inputs, head="disease", target_index=1)

    assert heatmap.shape == (2, 16, 16)
    assert np.all(heatmap >= 0)
    assert np.all(heatmap <= 1.0 + 1e-5)
    generator.close()


def test_attention_overlap_metrics():
    torch.manual_seed(1)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    inputs = torch.randn(1, 1, 16, 16)
    result = generator.generate_attention_overlap(inputs, disease_target=0, attribute_target=1)

    assert isinstance(result, AttentionOverlapResult)
    assert result.disease_heatmap.shape == (1, 16, 16)
    assert result.attribute_heatmap.shape == (1, 16, 16)
    assert 0.0 <= result.metrics["dice"] <= 1.0
    assert 0.0 <= result.metrics["iou"] <= 1.0
    assert -1.0 <= result.metrics["cosine"] <= 1.0
    generator.close()


def test_single_image_input():
    """Test that single image (3D tensor) is handled correctly."""
    torch.manual_seed(2)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    # Single image without batch dimension
    inputs = torch.randn(1, 16, 16)
    # Should be automatically batched
    heatmap = generator.generate_heatmap(inputs, head="disease")
    assert heatmap.shape == (1, 16, 16)
    generator.close()


def test_numpy_array_input():
    """Test that numpy arrays are accepted."""
    torch.manual_seed(3)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    inputs_np = np.random.randn(1, 1, 16, 16).astype(np.float32)
    heatmap = generator.generate_heatmap(inputs_np, head="disease")
    assert heatmap.shape == (1, 16, 16)
    generator.close()


def test_tuple_output_model():
    """Test with model that outputs tuple."""
    torch.manual_seed(4)
    model = TupleOutputModel()
    generator = GradCAMHeatmapGenerator(
        model, target_layer="conv", head_mappings={"disease": 0, "attribute": 1}
    )

    inputs = torch.randn(1, 1, 16, 16)
    heatmap = generator.generate_heatmap(inputs, head="disease")
    assert heatmap.shape == (1, 16, 16)

    result = generator.generate_attention_overlap(inputs)
    assert result.disease_heatmap.shape == (1, 16, 16)
    assert result.attribute_heatmap.shape == (1, 16, 16)
    generator.close()


def test_overlap_metrics_calculation():
    """Test overlap metrics with known inputs."""
    # Identical heatmaps should have high overlap
    heatmap1 = np.ones((1, 10, 10), dtype=np.float32) * 0.8
    heatmap2 = np.ones((1, 10, 10), dtype=np.float32) * 0.8

    metrics = GradCAMHeatmapGenerator.calculate_overlap(heatmap1, heatmap2, threshold=0.5)
    assert metrics["dice"] > 0.9
    assert metrics["iou"] > 0.9
    assert metrics["cosine"] > 0.9

    # Non-overlapping heatmaps should have low overlap
    heatmap1 = np.zeros((1, 10, 10), dtype=np.float32)
    heatmap1[0, :5, :] = 1.0
    heatmap2 = np.zeros((1, 10, 10), dtype=np.float32)
    heatmap2[0, 5:, :] = 1.0

    metrics = GradCAMHeatmapGenerator.calculate_overlap(heatmap1, heatmap2, threshold=0.5)
    assert metrics["dice"] < 0.1
    assert metrics["iou"] < 0.1


def test_layer_path_resolution():
    """Test that layer paths are resolved correctly."""
    model = TinyBackbone()

    # Test with string path
    generator1 = GradCAMHeatmapGenerator(model, target_layer="conv")
    assert generator1._target_layer == model.conv

    # Test with module directly
    generator2 = GradCAMHeatmapGenerator(model, target_layer=model.conv)
    assert generator2._target_layer == model.conv

    generator1.close()
    generator2.close()


def test_argmax_target_selection():
    """Test that argmax is used when target_index is None."""
    torch.manual_seed(5)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    inputs = torch.randn(1, 1, 16, 16)
    # Should use argmax of disease head
    heatmap = generator.generate_heatmap(inputs, head="disease", target_index=None)
    assert heatmap.shape == (1, 16, 16)
    generator.close()


def test_different_image_sizes():
    """Test with different input image sizes."""
    torch.manual_seed(6)
    model = TinyBackbone()
    generator = GradCAMHeatmapGenerator(
        model,
        target_layer="conv",
        head_mappings={
            "disease": lambda outputs: outputs["disease"],
            "attribute": lambda outputs: outputs["attribute"],
        },
    )

    for size in [16, 32, 64]:
        inputs = torch.randn(1, 1, size, size)
        heatmap = generator.generate_heatmap(inputs, head="disease")
        assert heatmap.shape == (1, size, size)

    generator.close()
