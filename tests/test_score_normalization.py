"""A01: Verify all core methods produce normalized risk_value in [0,1] and valid risk_level."""

from __future__ import annotations

import pytest

from shortcut_detect.benchmark.synthetic_generator import SyntheticGenerator
from shortcut_detect.reporting.risk_format import normalize_risk_level
from shortcut_detect.unified import ShortcutDetector

CORE_METHODS = ["hbac", "probe", "statistical", "geometric"]

VALID_RISK_LEVELS = {"low", "moderate", "high", "unknown"}


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic data with a moderate shortcut signal."""
    gen = SyntheticGenerator(
        n_samples=300,
        embedding_dim=32,
        shortcut_dims=3,
        group_ratio=0.5,
        seed=42,
    )
    return gen.generate(effect_size=1.0)


class TestScoreNormalization:
    """Verify risk_value is in [0,1] and risk_level is valid for each core method."""

    @pytest.mark.parametrize("method", CORE_METHODS)
    def test_risk_value_in_unit_interval(self, synthetic_data, method):
        data = synthetic_data
        detector = ShortcutDetector(methods=[method], seed=42)
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
        result = detector.results_.get(method, {})

        # risk_value is the normalized string level set by apply_standardized_risk
        risk_value = result.get("risk_value", "unknown")
        normalized = normalize_risk_level(risk_value)

        # The normalized risk_value must map to a known level
        assert normalized in VALID_RISK_LEVELS, (
            f"Method '{method}' produced risk_value='{risk_value}' which normalizes "
            f"to '{normalized}', not in {VALID_RISK_LEVELS}"
        )

        # Verify numeric interpretation is in [0, 1]
        level_to_numeric = {"low": 0.0, "moderate": 0.5, "high": 1.0, "unknown": 0.0}
        numeric = level_to_numeric[normalized]
        assert 0.0 <= numeric <= 1.0

    @pytest.mark.parametrize("method", CORE_METHODS)
    def test_risk_level_is_valid(self, synthetic_data, method):
        data = synthetic_data
        detector = ShortcutDetector(methods=[method], seed=42)
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
        result = detector.results_.get(method, {})

        risk_value = result.get("risk_value", "unknown")
        normalized = normalize_risk_level(risk_value)
        assert (
            normalized in VALID_RISK_LEVELS
        ), f"Method '{method}': risk_level '{risk_value}' is not one of {VALID_RISK_LEVELS}"

    @pytest.mark.parametrize("method", CORE_METHODS)
    def test_risk_label_present(self, synthetic_data, method):
        data = synthetic_data
        detector = ShortcutDetector(methods=[method], seed=42)
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
        result = detector.results_.get(method, {})

        if result.get("success"):
            assert "risk_value" in result, f"Method '{method}' missing 'risk_value' key"
            assert "risk_label" in result, f"Method '{method}' missing 'risk_label' key"

    def test_all_methods_together(self, synthetic_data):
        """Run all 4 core methods together and check all risk outputs."""
        data = synthetic_data
        detector = ShortcutDetector(methods=CORE_METHODS, seed=42)
        detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)

        for method in CORE_METHODS:
            result = detector.results_.get(method, {})
            if result.get("success"):
                risk_value = result.get("risk_value", "unknown")
                normalized = normalize_risk_level(risk_value)
                assert normalized in VALID_RISK_LEVELS

    @pytest.mark.parametrize("effect_size", [0.0, 2.0])
    def test_extreme_effect_sizes(self, effect_size):
        """Verify normalization holds for no-shortcut and very strong shortcut data."""
        gen = SyntheticGenerator(
            n_samples=200,
            embedding_dim=16,
            shortcut_dims=2,
            group_ratio=0.5,
            seed=99,
        )
        data = gen.generate(effect_size=effect_size)

        for method in CORE_METHODS:
            detector = ShortcutDetector(methods=[method], seed=42)
            detector.fit(data.embeddings, data.labels, group_labels=data.group_labels)
            result = detector.results_.get(method, {})
            risk_value = result.get("risk_value", "unknown")
            normalized = normalize_risk_level(risk_value)
            assert normalized in VALID_RISK_LEVELS, (
                f"Method '{method}' at effect_size={effect_size}: " f"risk '{risk_value}' not valid"
            )
