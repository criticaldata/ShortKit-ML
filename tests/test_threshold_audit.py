"""Threshold documentation audit – verify code defaults match paper claims.

Compares the actual default threshold values used in each detector module
against the values stated in the paper's Appendix C (Table tab:thresholds).

A05 – Threshold documentation audit (#68)
"""

from __future__ import annotations

# ── Paper-claimed thresholds (Appendix C, Table tab:thresholds) ──────────

PAPER_THRESHOLDS = {
    "probe": {
        "high_above": 0.85,
        "moderate_above": 0.70,
    },
    "statistical": {
        "high_above_frac": 0.30,  # >30% dims significant → HIGH
        "moderate_above_frac": 0.10,  # >10% dims significant → MODERATE
    },
    "hbac": {
        "high_purity": 0.80,
        "high_linear_acc": 0.85,
        "moderate_purity": 0.65,  # claimed in paper but see code note
    },
    "geometric": {
        "high_effect_threshold": 0.80,  # Cohen's d
        "moderate_effect_threshold": 0.50,  # Cohen's d (paper claim)
    },
    "equalized_odds": {
        "high_gap": 0.15,  # paper: HIGH > 0.15
        "moderate_gap": 0.05,  # paper: MODERATE > 0.05
    },
    "demographic_parity": {
        "high_gap": 0.15,  # paper: HIGH > 0.15
        "moderate_gap": 0.05,  # paper: MODERATE > 0.05
    },
}


# ── Probe thresholds ────────────────────────────────────────────────────


class TestProbeThresholds:
    """Probe: HIGH when metric >= 85%, MODERATE when metric >= threshold (default 70%)."""

    def test_default_threshold(self):
        from shortcut_detect.probes.sklearn_probe import MLProbeConfig

        cfg = MLProbeConfig()
        assert cfg.threshold == PAPER_THRESHOLDS["probe"]["moderate_above"], (
            f"Probe default threshold: expected {PAPER_THRESHOLDS['probe']['moderate_above']}, "
            f"got {cfg.threshold}"
        )

    def test_high_boundary(self):
        """Verify the sklearn probe maps metric >= 0.85 to 'high'."""
        from shortcut_detect.probes.sklearn_probe import MLProbeConfig

        cfg = MLProbeConfig()
        # The code uses: max(threshold, 0.85) for the HIGH boundary.
        # With default threshold=0.70, the HIGH boundary is 0.85.
        high_boundary = max(cfg.threshold, 0.85)
        assert high_boundary == PAPER_THRESHOLDS["probe"]["high_above"], (
            f"Probe HIGH boundary: expected {PAPER_THRESHOLDS['probe']['high_above']}, "
            f"got {high_boundary}"
        )

    def test_torch_probe_default_threshold(self):
        from shortcut_detect.probes.torch_probe import TorchProbeConfig

        cfg = TorchProbeConfig()
        assert cfg.threshold == PAPER_THRESHOLDS["probe"]["moderate_above"], (
            f"Torch probe default threshold: expected {PAPER_THRESHOLDS['probe']['moderate_above']}, "
            f"got {cfg.threshold}"
        )


# ── Statistical thresholds ──────────────────────────────────────────────


class TestStatisticalThresholds:
    """Statistical: risk derived from fraction of significant dims.

    The code sets risk_level = 'moderate' when *any* comparison has
    significant features, and 'low' otherwise.  There is no hard-coded
    30%/10% fraction threshold in the statistical detector – the paper's
    Appendix C describes higher-level interpretation guidelines rather
    than code defaults.  We document this discrepancy.
    """

    def test_alpha_default(self):
        """Default significance level should be 0.05."""
        # apply_threshold default alpha
        import inspect

        from shortcut_detect.statistical.group_diff_test import FeatureGroupDiffTest

        sig = inspect.signature(FeatureGroupDiffTest.apply_threshold)
        alpha_default = sig.parameters["alpha"].default
        assert alpha_default == 0.05, f"Expected alpha=0.05, got {alpha_default}"

    def test_risk_level_logic_documented(self):
        """The code uses binary moderate/low – the paper's 30%/10% fraction
        thresholds are interpretation guidelines, not code defaults.

        This test documents the discrepancy: the code triggers MODERATE for
        *any* significant features, while the paper suggests a tiered
        fraction-based scheme.
        """
        # Verify the code's actual logic by reading the risk assignment
        import inspect

        from shortcut_detect.statistical import group_diff_test

        source = inspect.getsource(
            group_diff_test.FeatureGroupDiffTest._update_results_from_significance
        )
        # The code does: risk_level = "moderate" if has_significant else "low"
        assert '"moderate"' in source, "Expected 'moderate' risk when significant features exist"
        assert '"low"' in source, "Expected 'low' risk when no significant features"
        # Document: no 30%/10% fraction thresholds in code
        assert "0.30" not in source and "0.10" not in source, (
            "DISCREPANCY RESOLVED: code now contains fraction thresholds "
            "(was expected to be absent)"
        )


# ── HBAC thresholds ─────────────────────────────────────────────────────


class TestHBACThresholds:
    """HBAC: HIGH when purity > 80% + linear acc > 85%; confidence-based levels."""

    def test_purity_threshold(self):
        """Code uses purity > 0.8 for high-purity cluster counting."""
        import inspect

        from shortcut_detect.clustering import hbac_detector

        source = inspect.getsource(
            hbac_detector.EmbeddingShortcutDetector._determine_shortcut_existence
        )
        assert 'c["purity"] > 0.8' in source, "Expected purity threshold of 0.8 in HBAC detector"

    def test_linear_accuracy_threshold(self):
        """Code uses test_accuracy > 0.85 for linearity check."""
        import inspect

        from shortcut_detect.clustering import hbac_detector

        source = inspect.getsource(
            hbac_detector.EmbeddingShortcutDetector._determine_shortcut_existence
        )
        assert (
            '["test_accuracy"] > 0.85' in source
        ), "Expected linear accuracy threshold of 0.85 in HBAC detector"

    def test_confidence_mapping(self):
        """Confidence 'high'/'medium'/'low' maps to risk levels correctly."""
        import inspect

        from shortcut_detect.clustering import hbac_detector

        source = inspect.getsource(hbac_detector.EmbeddingShortcutDetector._finalize_results)
        assert '"high": "high"' in source
        assert '"medium": "moderate"' in source
        assert '"low": "low"' in source

    def test_moderate_purity_discrepancy(self):
        """Paper claims MODERATE > 65% purity.

        The code does NOT use a 65% purity threshold directly.  Instead,
        confidence is derived from the number of shortcut types detected
        (high if >=2, medium if >=1, low otherwise).  This is a
        discrepancy with the paper's Appendix C claim.
        """
        import inspect

        from shortcut_detect.clustering import hbac_detector

        source = inspect.getsource(
            hbac_detector.EmbeddingShortcutDetector._determine_shortcut_existence
        )
        # The code does NOT contain a 0.65 threshold
        assert "0.65" not in source, (
            "DISCREPANCY RESOLVED: code now uses 0.65 purity threshold "
            "(was expected to be absent)"
        )


# ── Geometric thresholds ────────────────────────────────────────────────


class TestGeometricThresholds:
    """Geometric: HIGH when effect_size >= 0.8; MODERATE when effect or subspace triggers."""

    def test_effect_threshold_default(self):
        from shortcut_detect.geometric.geometric.src.detector import GeometricShortcutAnalyzer

        analyzer = GeometricShortcutAnalyzer()
        assert (
            analyzer.effect_threshold == PAPER_THRESHOLDS["geometric"]["high_effect_threshold"]
        ), (
            f"Geometric effect_threshold: "
            f"expected {PAPER_THRESHOLDS['geometric']['high_effect_threshold']}, "
            f"got {analyzer.effect_threshold}"
        )

    def test_moderate_threshold_discrepancy(self):
        """Paper claims MODERATE > 0.5 Cohen's d.

        The code does NOT use a separate 0.5 threshold for MODERATE.
        Instead, MODERATE is triggered when *either* the effect threshold
        (0.8) OR the subspace cosine threshold (0.85) triggers, but not
        both.  There is no 0.5 cutoff for moderate in code.
        """
        import inspect

        from shortcut_detect.geometric.geometric.src.detector import GeometricShortcutAnalyzer

        source = inspect.getsource(GeometricShortcutAnalyzer._assess_risk)
        # HIGH requires both high_effect AND overlapping_subspaces
        assert "high_effect and overlapping_subspaces" in source
        # MODERATE requires either one
        assert "high_effect or overlapping_subspaces" in source
        # No 0.5 threshold in the risk assessment
        assert "0.5" not in source, (
            "DISCREPANCY RESOLVED: code now uses 0.5 moderate threshold "
            "(was expected to be absent)"
        )

    def test_subspace_cosine_threshold(self):
        from shortcut_detect.geometric.geometric.src.detector import GeometricShortcutAnalyzer

        analyzer = GeometricShortcutAnalyzer()
        assert analyzer.subspace_cosine_threshold == 0.85


# ── Equalized Odds thresholds ──────────────────────────────────────────


class TestEqualizedOddsThresholds:
    """Equalized Odds: paper claims HIGH > 0.15, MODERATE > 0.05.

    Code default thresholds are 0.1 for both TPR/FPR gaps.
    HIGH = gap >= 2 * threshold (i.e., >= 0.2 by default)
    MODERATE = gap >= threshold (i.e., >= 0.1 by default)

    This means the code's effective HIGH boundary is 0.20 (not 0.15)
    and the MODERATE boundary is 0.10 (not 0.05).
    """

    def test_default_gap_thresholds(self):
        from shortcut_detect.fairness.equalized_odds.src.detector import EqualizedOddsDetector

        det = EqualizedOddsDetector()
        assert (
            det.tpr_gap_threshold == 0.1
        ), f"Expected tpr_gap_threshold=0.1, got {det.tpr_gap_threshold}"
        assert (
            det.fpr_gap_threshold == 0.1
        ), f"Expected fpr_gap_threshold=0.1, got {det.fpr_gap_threshold}"

    def test_high_boundary_discrepancy(self):
        """Code: HIGH at gap >= 2*threshold = 0.20; paper claims 0.15."""
        from shortcut_detect.fairness.equalized_odds.src.detector import EqualizedOddsDetector

        det = EqualizedOddsDetector()
        effective_high = 2 * max(det.tpr_gap_threshold, det.fpr_gap_threshold)
        paper_high = PAPER_THRESHOLDS["equalized_odds"]["high_gap"]
        # Document the discrepancy: 0.20 != 0.15
        assert (
            effective_high != paper_high
        ), f"Expected discrepancy: code HIGH={effective_high} vs paper HIGH={paper_high}"
        assert effective_high == 0.2

    def test_moderate_boundary_discrepancy(self):
        """Code: MODERATE at gap >= threshold = 0.10; paper claims 0.05."""
        from shortcut_detect.fairness.equalized_odds.src.detector import EqualizedOddsDetector

        det = EqualizedOddsDetector()
        effective_moderate = max(det.tpr_gap_threshold, det.fpr_gap_threshold)
        paper_moderate = PAPER_THRESHOLDS["equalized_odds"]["moderate_gap"]
        # Document the discrepancy: 0.10 != 0.05
        assert (
            effective_moderate != paper_moderate
        ), f"Expected discrepancy: code MODERATE={effective_moderate} vs paper={paper_moderate}"
        assert effective_moderate == 0.1


# ── Demographic Parity thresholds ──────────────────────────────────────


class TestDemographicParityThresholds:
    """Demographic Parity: paper claims HIGH > 0.15, MODERATE > 0.05.

    Code default dp_gap_threshold = 0.1.
    HIGH = gap >= 2 * threshold (i.e., >= 0.2 by default)
    MODERATE = gap >= threshold (i.e., >= 0.1 by default)

    Same discrepancy pattern as equalized odds.
    """

    def test_default_gap_threshold(self):
        from shortcut_detect.fairness.demographic_parity.src.detector import (
            DemographicParityDetector,
        )

        det = DemographicParityDetector()
        assert (
            det.dp_gap_threshold == 0.1
        ), f"Expected dp_gap_threshold=0.1, got {det.dp_gap_threshold}"

    def test_high_boundary_discrepancy(self):
        """Code: HIGH at gap >= 2*threshold = 0.20; paper claims 0.15."""
        from shortcut_detect.fairness.demographic_parity.src.detector import (
            DemographicParityDetector,
        )

        det = DemographicParityDetector()
        effective_high = 2 * det.dp_gap_threshold
        paper_high = PAPER_THRESHOLDS["demographic_parity"]["high_gap"]
        assert (
            effective_high != paper_high
        ), f"Expected discrepancy: code HIGH={effective_high} vs paper HIGH={paper_high}"
        assert effective_high == 0.2

    def test_moderate_boundary_discrepancy(self):
        """Code: MODERATE at gap >= threshold = 0.10; paper claims 0.05."""
        from shortcut_detect.fairness.demographic_parity.src.detector import (
            DemographicParityDetector,
        )

        det = DemographicParityDetector()
        effective_moderate = det.dp_gap_threshold
        paper_moderate = PAPER_THRESHOLDS["demographic_parity"]["moderate_gap"]
        assert (
            effective_moderate != paper_moderate
        ), f"Expected discrepancy: code MODERATE={effective_moderate} vs paper={paper_moderate}"
        assert effective_moderate == 0.1


# ── Summary of discrepancies ───────────────────────────────────────────


class TestDiscrepancySummary:
    """Aggregate summary of all discrepancies between paper and code."""

    def test_print_discrepancy_report(self):
        """This test always passes but prints a summary of findings."""
        discrepancies = [
            {
                "method": "Statistical",
                "paper": "HIGH > 30% dims, MODERATE > 10% dims",
                "code": "MODERATE if any significant features, else LOW (binary)",
                "severity": "MEDIUM – paper describes tiered fractions, code uses binary",
            },
            {
                "method": "HBAC",
                "paper": "MODERATE > 65% purity",
                "code": "Confidence from shortcut_type count (>=2 high, >=1 medium, else low); "
                "no 65% purity threshold",
                "severity": "LOW – different mechanism, similar intent",
            },
            {
                "method": "Geometric",
                "paper": "MODERATE > 0.5 Cohen's d",
                "code": "MODERATE when either effect (>=0.8) or subspace cosine (>=0.85) "
                "triggers alone; no 0.5 threshold",
                "severity": "MEDIUM – no separate moderate cutoff in code",
            },
            {
                "method": "Equalized Odds",
                "paper": "HIGH > 0.15, MODERATE > 0.05",
                "code": "HIGH >= 0.20 (2*threshold), MODERATE >= 0.10 (threshold=0.1)",
                "severity": "MEDIUM – code thresholds are more conservative",
            },
            {
                "method": "Demographic Parity",
                "paper": "HIGH > 0.15, MODERATE > 0.05",
                "code": "HIGH >= 0.20 (2*threshold), MODERATE >= 0.10 (threshold=0.1)",
                "severity": "MEDIUM – code thresholds are more conservative",
            },
        ]

        matches = [
            {"method": "Probe", "detail": "HIGH >= 85%, MODERATE >= 70% – matches paper"},
            {"method": "HBAC purity", "detail": "Purity > 80% for high-purity clusters – matches"},
            {"method": "HBAC linear acc", "detail": "Linear accuracy > 85% – matches paper"},
            {"method": "Geometric effect", "detail": "effect_threshold=0.8 – matches paper HIGH"},
        ]

        print("\n" + "=" * 70)
        print("THRESHOLD AUDIT: Paper vs Code Comparison")
        print("=" * 70)
        print(f"\n  MATCHES ({len(matches)}):")
        for m in matches:
            print(f"    [OK] {m['method']}: {m['detail']}")
        print(f"\n  DISCREPANCIES ({len(discrepancies)}):")
        for d in discrepancies:
            print(f"    [!!] {d['method']}")
            print(f"         Paper:    {d['paper']}")
            print(f"         Code:     {d['code']}")
            print(f"         Severity: {d['severity']}")
        print("=" * 70)

        # Test passes – discrepancies are documented, not failures.
        assert len(discrepancies) == 5, "Update this count if discrepancies are fixed"
