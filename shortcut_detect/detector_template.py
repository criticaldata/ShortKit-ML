"""Copy-paste template for implementing a new shortcut detection method.

This file contains two template classes:

1. ``MyMethodDetector`` -- a standalone detector (subclasses
   :class:`DetectorBase`).  Use this when you want a self-contained
   detector that can be used independently or plugged into the unified
   ``ShortcutDetector``.

2. ``MyMethodBuilder`` -- a builder/runner (subclasses
   :class:`BaseDetector`).  Use this to integrate your detector with
   the ``ShortcutDetector`` orchestrator, which calls ``build()`` and
   ``run()`` on your builder.

How to use this template:
    1. Copy this file into your method's package, e.g.
       ``shortcut_detect/my_method/my_method_detector.py``.
    2. Rename the classes (search-replace ``MyMethod``).
    3. Fill in the ``# TODO`` sections with your detection logic.
    4. Delete the template comments once the implementation is complete.
    5. Register the builder in the unified detector
       (``shortcut_detect/unified.py``).
    6. Add tests in ``tests/test_my_method.py``.

See ``docs/contributing.md`` for the full step-by-step guide.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from shortcut_detect.base_builder import BaseDetector
from shortcut_detect.detector_base import DetectorBase, RiskLevel

# ---------------------------------------------------------------------------
# Optional: configuration dataclass
# ---------------------------------------------------------------------------
# Use a frozen dataclass when your detector has many constructor
# parameters.  This keeps the __init__ signature clean and makes the
# config serializable for metadata/report output.


@dataclass(frozen=True)
class MyMethodConfig:
    """Configuration for MyMethodDetector.

    Attributes:
        threshold: Decision boundary for shortcut detection.
        min_samples: Minimum number of samples required per group.
    """

    threshold: float = 0.5
    min_samples: int = 10


# ---------------------------------------------------------------------------
# Optional: structured report dataclass
# ---------------------------------------------------------------------------
# A dataclass makes it easy to convert the detailed report to a dict
# via ``dataclasses.asdict()`` when passing to ``_set_results(report=...)``.


@dataclass
class MyMethodReport:
    """Structured report for MyMethodDetector.

    Attributes:
        score: The primary detection score.
        group_scores: Per-group breakdown of scores.
        reference: Citation or method reference.
        risk_level: Assessed risk level string.
        notes: Human-readable explanation.
    """

    score: float
    group_scores: dict[str, float]
    reference: str
    risk_level: str
    notes: str


# ===========================================================================
# Part 1: Standalone detector (DetectorBase subclass)
# ===========================================================================


class MyMethodDetector(DetectorBase):
    """Detect shortcuts using the MyMethod algorithm.

    This detector analyzes embeddings and group labels to determine
    whether the model relies on spurious correlations (shortcuts).

    Args:
        config: Optional configuration object.  If ``None``, default
            values are used.
        threshold: Shortcut detection threshold (convenience parameter;
            ignored when *config* is provided).

    Attributes:
        results_: Standardized results dict (populated after ``fit``).
        shortcut_detected_: Boolean detection outcome (populated after ``fit``).
        report_: Detailed :class:`MyMethodReport` (populated after ``fit``).

    Example:
        >>> detector = MyMethodDetector(threshold=0.6)
        >>> detector.fit(embeddings, labels, group_labels)
        >>> print(detector.summary())
        my_method: shortcut=NO, risk=LOW, score=0.42
        >>> report = detector.get_report()
    """

    def __init__(
        self,
        config: MyMethodConfig | None = None,
        threshold: float = 0.5,
    ) -> None:
        # Step 1: Call super().__init__ with a unique method name.
        # Convention: use snake_case, e.g. "demographic_parity".
        super().__init__(method="my_method")

        # Step 2: Store configuration.
        self.config = config or MyMethodConfig(threshold=threshold)

        # Step 3: Initialize method-specific fitted attributes to
        # sentinel values.  These will be populated in fit().
        self.score_: float = float("nan")
        self.report_: MyMethodReport | None = None

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> MyMethodDetector:
        """Run detection and populate results.

        Args:
            embeddings: Array of shape ``(n_samples, n_features)``
                containing the embedding vectors.
            labels: Array of shape ``(n_samples,)`` with target labels.
            group_labels: Array of shape ``(n_samples,)`` with
                protected/group attribute labels.

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        # ------------------------------------------------------------------
        # Step 1: Validate inputs
        # ------------------------------------------------------------------
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (n_samples, n_features).")
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels length must match embeddings rows.")
        if group_labels.shape[0] != embeddings.shape[0]:
            raise ValueError("group_labels length must match embeddings rows.")

        # ------------------------------------------------------------------
        # Step 2: Run your detection algorithm
        # ------------------------------------------------------------------
        # TODO: Replace this placeholder with your actual logic.
        self.score_ = float(np.random.default_rng(42).random())
        group_scores: dict[str, float] = {}
        for g in np.unique(group_labels):
            mask = group_labels == g
            group_scores[str(g)] = float(np.mean(embeddings[mask, 0]))

        # ------------------------------------------------------------------
        # Step 3: Determine detection outcome
        # ------------------------------------------------------------------
        shortcut = self.score_ >= self.config.threshold
        self.shortcut_detected_ = shortcut

        # ------------------------------------------------------------------
        # Step 4: Assess risk level
        # ------------------------------------------------------------------
        # Map your score to one of the canonical risk levels.
        if self.score_ >= 2 * self.config.threshold:
            risk_level = RiskLevel.HIGH
        elif self.score_ >= self.config.threshold:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW

        notes = (
            "Score exceeds threshold -- potential shortcut detected."
            if shortcut
            else "Score below threshold -- no shortcut evidence."
        )

        # ------------------------------------------------------------------
        # Step 5: Build the structured report (optional but recommended)
        # ------------------------------------------------------------------
        self.report_ = MyMethodReport(
            score=self.score_,
            group_scores=group_scores,
            reference="Author et al. 2025",
            risk_level=risk_level.value,
            notes=notes,
        )

        # ------------------------------------------------------------------
        # Step 6: Call _set_results to populate self.results_
        # ------------------------------------------------------------------
        # This is REQUIRED.  _set_results normalizes the risk level and
        # builds the standardized dict consumed by get_report(),
        # summary(), and the meta-classifier.
        self._set_results(
            shortcut_detected=shortcut,
            risk_level=risk_level,
            # metrics: small scalar values for dashboards/summaries.
            metrics={
                "score": self.score_,
            },
            notes=notes,
            # metadata: config values, dataset sizes, etc.
            metadata={
                "threshold": self.config.threshold,
                "n_samples": int(embeddings.shape[0]),
                "n_groups": len(group_scores),
            },
            # report: detailed structured output (optional).
            report=asdict(self.report_),
            # details: large auxiliary outputs (optional).
            # details={"per_sample_scores": scores_array.tolist()},
        )

        # ------------------------------------------------------------------
        # Step 7: Mark the detector as fitted
        # ------------------------------------------------------------------
        # IMPORTANT: set _is_fitted = True AFTER _set_results so that
        # get_report() / summary() never see an incomplete results_ dict.
        self._is_fitted = True

        # ------------------------------------------------------------------
        # Step 8: Return self for method chaining
        # ------------------------------------------------------------------
        return self


# ===========================================================================
# Part 2: Builder/runner for the unified ShortcutDetector (BaseDetector)
# ===========================================================================
# The builder pattern is used by ShortcutDetector to construct and run
# detectors in a uniform way.  The builder receives raw data and returns
# a result dict that the orchestrator consumes for reporting.
#
# Required methods:
#   build() -> detector instance
#   run(embeddings, labels, group_labels, ...) -> dict
#
# The returned dict from run() has this structure:
# {
#     "detector": <fitted DetectorBase instance>,
#     "results": <the detector's results_ dict>,
#     "summary_title": "My Method Detection",
#     "summary_lines": ["Score: 0.42", "Risk: low"],
#     "risk_indicators": ["score=0.85 (threshold exceeded)"],
#     "success": True,
# }


class MyMethodBuilder(BaseDetector):
    """Builder that constructs and runs :class:`MyMethodDetector`.

    This class is registered in ``shortcut_detect/unified.py`` so the
    orchestrator can invoke the method by name.

    Args:
        seed: Random seed for reproducibility.
        kwargs: Additional keyword arguments forwarded to the detector.
        method: Method identifier string.
    """

    def __init__(
        self,
        seed: int = 42,
        kwargs: dict[str, Any] | None = None,
        method: str = "my_method",
    ) -> None:
        super().__init__(seed=seed, kwargs=kwargs, method=method)

    def build(self) -> MyMethodDetector:
        """Construct and return a configured detector instance.

        Returns:
            An unfitted :class:`MyMethodDetector`.
        """
        threshold = self.kwargs.get("threshold", 0.5)
        config = MyMethodConfig(threshold=threshold)
        return MyMethodDetector(config=config)

    def run(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        group_labels: np.ndarray,
        feature_names: list[str] | None,
        protected_labels: np.ndarray | None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Execute the full detection workflow.

        Args:
            embeddings: Embedding matrix of shape ``(n_samples, dim)``.
            labels: Target labels of shape ``(n_samples,)``.
            group_labels: Group/protected attribute labels.
            feature_names: Optional feature name list.
            protected_labels: Optional protected attribute labels (may
                differ from *group_labels* in some setups).
            splits: Optional dict mapping split names to index arrays.
            extra_labels: Optional dict of additional label arrays.

        Returns:
            A dict with keys ``detector``, ``results``,
            ``summary_title``, ``summary_lines``,
            ``risk_indicators``, and ``success``.
        """
        print("Running my_method detection...")

        detector = self.build()

        try:
            # Fit the detector.  Pass whichever arguments your
            # detector's fit() method expects.
            detector.fit(embeddings, labels, group_labels)
            results = detector.get_report()

            # Extract key metrics for the summary.
            score = results["metrics"].get("score")
            shortcut = results.get("shortcut_detected")

            # Build human-readable summary lines.
            if shortcut is True:
                risk_line = "Shortcut signal detected"
            elif shortcut is False:
                risk_line = "No shortcut signal"
            else:
                risk_line = "Shortcut signal inconclusive"

            # Build risk indicators (consumed by the unified report).
            risk_indicators: list[str] = []
            if shortcut is True and isinstance(score, int | float):
                risk_indicators.append(f"score={score:.3f} (threshold exceeded)")

            score_text = (
                f"Score: {score:.4f}" if isinstance(score, int | float) else f"Score: {score}"
            )

            return {
                "detector": detector,
                "results": results,
                "summary_title": "My Method Detection",
                "summary_lines": [score_text, risk_line],
                "risk_indicators": risk_indicators,
                "success": True,
            }

        except Exception as exc:
            warnings.warn(f"my_method detection failed: {exc}", stacklevel=2)
            return {"success": False, "error": str(exc)}
