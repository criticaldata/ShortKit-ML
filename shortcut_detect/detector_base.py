"""Detector base class contract for shortcut detection methods.

Every standalone detection method in ``shortcut_detect`` inherits from
:class:`DetectorBase`.  This module also defines the :class:`RiskLevel`
enum used to communicate risk across the library.

See Also:
    ``shortcut_detect/detector_template.py`` for a copy-paste-ready
    template that implements both :class:`DetectorBase` (standalone
    detector) and :class:`BaseDetector` (builder/runner pattern).
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """Canonical risk level values used across detectors.

    Members:
        LOW: The detection method found little or no evidence of a shortcut.
        MODERATE: Some evidence of a shortcut; warrants investigation.
        HIGH: Strong evidence of a shortcut.
        UNKNOWN: The method could not determine a risk level (e.g.
            insufficient data, or not yet implemented).

    Example:
        >>> RiskLevel.from_string("medium")  # legacy alias
        <RiskLevel.MODERATE: 'moderate'>
        >>> RiskLevel.HIGH.to_display()
        'High'
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | RiskLevel | None) -> RiskLevel:
        """Normalize legacy/user values into canonical risk levels.

        Args:
            value: A risk string such as ``"low"``, ``"medium"``
                (mapped to MODERATE), ``"high"``, ``"unknown"``, a
                :class:`RiskLevel` member, or ``None``.

        Returns:
            The corresponding :class:`RiskLevel` member.  Returns
            ``RiskLevel.UNKNOWN`` when *value* is ``None`` or
            unrecognized.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN

        normalized = str(value).strip().lower()
        legacy_map = {"medium": cls.MODERATE.value}
        normalized = legacy_map.get(normalized, normalized)

        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN

    def to_display(self) -> str:
        """Human-readable risk label."""
        return self.value.capitalize()


class DetectorBase(ABC):
    """Abstract base class that every standalone detector must subclass.

    The detector contract requires that subclasses:

    1. Call ``super().__init__(method="<method_name>")`` in their
       ``__init__``.
    2. Implement :meth:`fit`, which runs the detection algorithm.
    3. Inside ``fit``, call :meth:`_set_results` to populate the
       standardized :attr:`results_` dictionary.
    4. Set ``self.shortcut_detected_`` to ``True``, ``False``, or
       ``None``.
    5. Set ``self._is_fitted = True`` **after** results are finalized.
    6. Return ``self`` from ``fit`` to allow method chaining.

    Standard ``results_`` schema
    ----------------------------

    After ``fit`` completes, :attr:`results_` must conform to this
    schema (populated automatically by :meth:`_set_results`)::

        {
            "method": str,
            # Unique method identifier (set in __init__).

            "shortcut_detected": bool | None,
            # True  -> shortcut evidence found.
            # False -> no shortcut evidence.
            # None  -> method cannot determine (e.g. not implemented).

            "risk_level": "low" | "moderate" | "high" | "unknown",
            # Normalized via RiskLevel.from_string().

            "metrics": {
                # Small, scalar summaries suitable for tables/dashboards.
                # Example: {"accuracy": 0.95, "dp_gap": 0.12}
            },

            "notes": str,
            # Human-readable explanation of the finding.

            "metadata": {
                # Configuration values, dataset counts, or other
                # non-metric context.
                # Example: {"n_samples": 5000, "threshold": 0.1}
            },

            # --- optional keys (omitted when None) ---

            "report": {
                # Detailed structured report: per-group breakdowns,
                # confusion matrices, etc.
            },

            "details": {
                # Large or auxiliary outputs: arrays, plots, model
                # weights.  Consumers should not assume these are small.
            },
        }

    Formatting guidelines:
        - Keep ``metrics`` small and scalar; reserve arrays/tables for
          ``report`` or ``details``.
        - Use ``risk_level="unknown"`` when the method does not provide
          a clear risk signal.
        - ``shortcut_detected`` should be ``None`` if the algorithm has
          not yet implemented detection logic.

    Example:
        >>> class MyDetector(DetectorBase):
        ...     def __init__(self, threshold: float = 0.5):
        ...         super().__init__(method="my_detector")
        ...         self.threshold = threshold
        ...
        ...     def fit(self, embeddings, labels):
        ...         score = compute_score(embeddings, labels)
        ...         shortcut = score >= self.threshold
        ...         self.shortcut_detected_ = shortcut
        ...         self._set_results(
        ...             shortcut_detected=shortcut,
        ...             risk_level="high" if shortcut else "low",
        ...             metrics={"score": score},
        ...             notes="Detection complete.",
        ...             metadata={"threshold": self.threshold},
        ...         )
        ...         self._is_fitted = True
        ...         return self

    See Also:
        ``shortcut_detect/detector_template.py`` for a full copy-paste
        template including both DetectorBase and BaseDetector patterns.
    """

    def __init__(self, method: str) -> None:
        """Initialize the detector.

        Args:
            method: A unique string identifier for this detection
                method.  This value is stored in ``results_["method"]``
                and used in summary output.  By convention, use
                snake_case (e.g. ``"demographic_parity"``).
        """
        self.method: str = method
        self.results_: dict[str, Any] = {}
        self.shortcut_detected_: bool | None = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> DetectorBase:
        """Run the detection algorithm and populate results.

        Subclasses **must** override this method.  The implementation
        should:

        1. Accept whatever data the method needs (typically
           ``embeddings``, ``labels``, and/or ``group_labels``).
        2. Compute detection metrics.
        3. Call ``self._set_results(...)`` with the outcome.
        4. Set ``self.shortcut_detected_`` to the boolean outcome.
        5. Set ``self._is_fitted = True``.
        6. Return ``self``.

        Returns:
            The fitted detector instance (``self``), enabling method
            chaining like ``detector.fit(X, y).summary()``.

        Raises:
            NotImplementedError: If the subclass does not override this
                method.
        """
        raise NotImplementedError

    def _set_results(
        self,
        *,
        shortcut_detected: bool | None,
        risk_level: str | RiskLevel = RiskLevel.UNKNOWN,
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
        report: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Populate :attr:`results_` with the standardized schema.

        This is the **only** recommended way to build ``results_``.
        Calling this method guarantees that the dict conforms to the
        schema expected by :meth:`get_report`, :meth:`summary`, and the
        meta-classifier.

        Args:
            shortcut_detected: Whether the method found shortcut
                evidence.  Use ``True`` for detected, ``False`` for not
                detected, ``None`` if the method cannot determine.
            risk_level: The assessed risk level.  Accepts a string
                (``"low"``, ``"moderate"``, ``"high"``, ``"unknown"``)
                or a :class:`RiskLevel` member.  Legacy value
                ``"medium"`` is mapped to ``"moderate"`` automatically.
            metrics: Small, scalar metrics suitable for display in
                summaries and dashboards.  Example:
                ``{"accuracy": 0.95, "gap": 0.12}``.  Avoid placing
                arrays or large objects here; use *report* or *details*
                instead.
            notes: A human-readable string explaining the finding
                (e.g. ``"Large parity gap detected across groups."``).
            metadata: Non-metric context such as configuration values,
                dataset sizes, or feature counts.  Example:
                ``{"n_samples": 5000, "threshold": 0.1}``.
            report: An optional detailed structured report (per-group
                breakdowns, confusion matrices, etc.).  Only included
                in ``results_`` when not ``None``.
            details: An optional dict for large or auxiliary outputs
                (arrays, model weights, plots).  Only included in
                ``results_`` when not ``None``.
        """
        canonical_risk = RiskLevel.from_string(risk_level).value
        self.results_ = {
            "method": self.method,
            "shortcut_detected": shortcut_detected,
            "risk_level": canonical_risk,
            "metrics": metrics or {},
            "notes": notes,
            "metadata": metadata or {},
        }
        if report is not None:
            self.results_["report"] = report
        if details is not None:
            self.results_["details"] = details

    def _ensure_fitted(self) -> None:
        """Raise :class:`ValueError` if the detector has not been fitted."""
        if not self._is_fitted:
            raise ValueError("Must call fit() before requesting results.")

    def get_report(self) -> dict[str, Any]:
        """Return the standardized report dict (:attr:`results_`).

        Returns:
            A copy-safe reference to :attr:`results_`.

        Raises:
            ValueError: If :meth:`fit` has not been called yet.
        """
        self._ensure_fitted()
        return self.results_

    def summary(self) -> str:
        """Return a human-readable one-line summary.

        The format is::

            method_name: shortcut=YES|NO|UNKNOWN, risk=LOW|..., key=val, ...

        Returns:
            A single-line string summarizing the detection outcome and
            scalar metrics.
        """
        if not self._is_fitted or not self.results_:
            return f"{self.method}: not fitted"

        shortcut = self.results_.get("shortcut_detected")
        if shortcut is True:
            shortcut_text = "YES"
        elif shortcut is False:
            shortcut_text = "NO"
        else:
            shortcut_text = "UNKNOWN"

        risk = RiskLevel.from_string(self.results_.get("risk_level")).value.upper()
        metrics = self.results_.get("metrics", {})
        metric_parts = []
        for key, value in metrics.items():
            if isinstance(value, str | bool | numbers.Number):
                metric_parts.append(f"{key}={value}")
        metrics_text = ", ".join(metric_parts)
        if metrics_text:
            return f"{self.method}: shortcut={shortcut_text}, risk={risk}, {metrics_text}"
        return f"{self.method}: shortcut={shortcut_text}, risk={risk}"
