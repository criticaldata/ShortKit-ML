from shortcut_detect.detector_base import DetectorBase, RiskLevel


class _DummyDetector(DetectorBase):
    def __init__(self) -> None:
        super().__init__(method="dummy")

    def fit(self, *args, **kwargs):
        self._set_results(shortcut_detected=False, risk_level=RiskLevel.LOW)
        self._is_fitted = True
        return self


def test_risk_level_from_string_normalization():
    assert RiskLevel.from_string(None) == RiskLevel.UNKNOWN
    assert RiskLevel.from_string("High") == RiskLevel.HIGH
    assert RiskLevel.from_string(" medium ") == RiskLevel.MODERATE
    assert RiskLevel.from_string("invalid-value") == RiskLevel.UNKNOWN


def test_set_results_normalizes_risk_level_values():
    detector = _DummyDetector()

    detector._set_results(shortcut_detected=True, risk_level="Medium")
    assert detector.results_["risk_level"] == "moderate"

    detector._set_results(shortcut_detected=True, risk_level=RiskLevel.HIGH)
    assert detector.results_["risk_level"] == "high"

    detector._set_results(shortcut_detected=None, risk_level="not-a-risk")
    assert detector.results_["risk_level"] == "unknown"


def test_summary_uses_normalized_canonical_risk():
    detector = _DummyDetector()
    detector._set_results(shortcut_detected=True, risk_level="Medium", metrics={"x": 1})
    detector._is_fitted = True

    summary = detector.summary()
    assert "risk=MODERATE" in summary
    assert "x=1" in summary
