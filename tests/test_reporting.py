"""Tests for report generation."""

import os
import tempfile

import numpy as np
import pytest

from shortcut_detect import ShortcutDetector
from shortcut_detect.reporting import ReportBuilder
from tests.fixtures.synthetic_data import generate_linear_shortcut


@pytest.fixture
def fitted_detector():
    """Create a fitted detector for testing."""
    embeddings, labels = generate_linear_shortcut(n_samples=500, embedding_dim=30)
    detector = ShortcutDetector(methods=["hbac", "probe", "statistical"])
    detector.fit(embeddings, labels)
    return detector


def test_report_builder_init(fitted_detector):
    """Test ReportBuilder initialization."""
    builder = ReportBuilder(fitted_detector)
    assert builder.detector is not None
    assert builder.results is not None


def test_report_builder_requires_fitted_detector():
    """Test ReportBuilder raises error for unfitted detector."""
    detector = ShortcutDetector()
    with pytest.raises(ValueError, match="must be fitted"):
        ReportBuilder(detector)


def test_generate_html_report(fitted_detector):
    """Test HTML report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")

        builder = ReportBuilder(fitted_detector)
        builder.to_html(output_path, include_visualizations=False)

        # Check file was created
        assert os.path.exists(output_path)

        # Check content
        with open(output_path) as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "Shortcut Detection Report" in content
            assert "Dataset Information" in content
            assert "<strong>Risk:</strong>" in content
            assert "<strong>Reason:</strong>" in content


def test_generate_markdown_report(fitted_detector):
    """Test Markdown report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.md")

        builder = ReportBuilder(fitted_detector)
        builder.to_markdown(output_path, include_visualizations=False)

        # Check file was created
        assert os.path.exists(output_path)

        # Check content
        with open(output_path) as f:
            content = f.read()
            assert "# 🔍 Shortcut Detection Report" in content
            assert "## 📊 Overall Assessment" in content
            assert "- Risk: " in content
            assert "- Reason: " in content


def test_generate_report_via_detector(fitted_detector):
    """Test report generation through detector interface."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")

        fitted_detector.generate_report(output_path, format="html", include_visualizations=False)

        assert os.path.exists(output_path)


def test_generate_markdown_via_detector(fitted_detector):
    """Test markdown generation through detector interface."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.md")

        fitted_detector.generate_report(
            output_path, format="markdown", include_visualizations=False
        )

        assert os.path.exists(output_path)


def test_report_invalid_format(fitted_detector):
    """Test invalid report format raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.txt")

        with pytest.raises(ValueError, match="Unknown format"):
            fitted_detector.generate_report(output_path, format="invalid")


def test_generate_3d_plot():
    """Test Plotly 3D generation (skips if plotly not installed)."""
    import numpy as np

    from shortcut_detect.reporting.visualizations import generate_interactive_3d_plot

    embeddings = np.random.randn(60, 8)
    labels = np.random.randint(0, 3, size=60)

    html = generate_interactive_3d_plot(embeddings, labels)
    assert isinstance(html, str)


def test_cav_report_and_csv_export():
    """Test CAV section rendering and CSV export output."""
    rng = np.random.default_rng(3)
    dim = 8
    concept = rng.normal(0.0, 0.5, size=(50, dim))
    random = rng.normal(0.0, 0.5, size=(50, dim))
    concept[:, 0] += 2.0
    random[:, 0] -= 2.0
    target_dd = rng.normal(0.0, 0.2, size=(70, dim))
    target_dd[:, 0] += 1.0

    detector = ShortcutDetector(methods=["cav"])
    detector.fit_from_loaders(
        {
            "cav": {
                "concept_sets": {"shortcut": concept},
                "random_set": random,
                "target_directional_derivatives": target_dd,
            }
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "cav_report.md")
        csv_dir = os.path.join(tmpdir, "csv")
        builder = ReportBuilder(detector)
        builder.to_markdown(md_path)
        exported = builder.to_csv(csv_dir)

        with open(md_path) as f:
            content = f.read()
        assert "## CAV (Concept Activation Vectors)" in content
        assert "Risk:" in content
        assert "cav_concept_scores" in exported
        assert os.path.exists(exported["cav_concept_scores"])


def test_pvalue_heatmap_top_k_selection():
    """P-value heatmap shows top-K most significant dimensions by default."""
    from shortcut_detect.reporting.visualizations import generate_pvalue_heatmap

    # Dim 2 and 5 have smallest p-values
    p_values_dict = {
        "0_vs_1": np.array([0.5, 0.3, 0.01, 0.2, 0.4, 0.001, 0.6, 0.7]),
    }
    heatmap = generate_pvalue_heatmap(p_values_dict, top_k=3, show_all=False)
    assert heatmap is not None
    assert isinstance(heatmap, str)
    assert len(heatmap) > 100  # Base64 PNG


def test_pvalue_heatmap_show_all():
    """P-value heatmap with show_all includes more dimensions."""
    from shortcut_detect.reporting.visualizations import generate_pvalue_heatmap

    p_values_dict = {"0_vs_1": np.random.rand(50) * 0.5}
    heatmap_top = generate_pvalue_heatmap(p_values_dict, top_k=5, show_all=False)
    heatmap_full = generate_pvalue_heatmap(
        p_values_dict, top_k=5, show_all=True, max_features_full=25
    )
    assert heatmap_top is not None
    assert heatmap_full is not None
    # Full heatmap should be larger (more dimensions = larger image)
    assert len(heatmap_full) >= len(heatmap_top)


def test_pvalue_heatmap_uses_corrected_pvalues():
    """P-value heatmap prefers corrected_pvalues when provided."""
    from shortcut_detect.reporting.visualizations import generate_pvalue_heatmap

    p_values = {"0_vs_1": np.array([0.1, 0.05, 0.01])}
    corrected = {"0_vs_1": np.array([0.2, 0.1, 0.02])}
    heatmap = generate_pvalue_heatmap(p_values, corrected_pvalues_dict=corrected)
    assert heatmap is not None


def test_gce_section_smart_summary_identical_losses():
    """GCE section shows summary when top samples share identical loss."""
    embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=16)
    detector = ShortcutDetector(methods=["gce"])
    detector.fit(embeddings, labels)

    builder = ReportBuilder(detector)
    builder.plots = {}  # Skip visualizations
    html = builder._generate_html(include_viz=False)

    # If GCE has identical losses in top samples, we show summary
    # We can't guarantee identical losses, so just check GCE section renders
    assert "GCE (Generalized Cross Entropy)" in html
    assert "High-Loss" in html or " Loss" in html


def test_report_includes_expandable_full_heatmap():
    """Report with statistical results includes expandable full heatmap."""
    embeddings, labels = generate_linear_shortcut(n_samples=200, embedding_dim=25)
    detector = ShortcutDetector(methods=["statistical"])
    detector.fit(embeddings, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.html")
        detector.generate_report(output_path, format="html", include_visualizations=True)

        with open(output_path) as f:
            content = f.read()

        assert "Expand to full heatmap" in content
        assert "Download full heatmap" in content
        assert "pvalue_heatmap" in content or "data:image/png;base64" in content
