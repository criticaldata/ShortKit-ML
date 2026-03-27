"""
Causal Effect Regularization detector for shortcut detection.

Estimates causal effect of each attribute on the task label.
Attributes with near-zero estimated causal effect are flagged as likely spurious shortcuts.

Reference: Kumar et al. 2023, "Causal Effect Regularization: Automated Detection and
Removal of Spurious Attributes", arXiv:2306.11072.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ....detector_base import DetectorBase


@dataclass
class AttributeEffectResult:
    """Per-attribute causal effect summary."""

    attribute_name: str
    causal_effect: float
    is_spurious: bool
    n_samples_a0: int
    n_samples_a1: int


class CausalEffectDetector(DetectorBase):
    """
    Detect shortcut attributes via causal effect estimation.

    Estimates the causal effect of each candidate attribute on the task label.
    Attributes with near-zero estimated effect are flagged as spurious (shortcuts),
    since changing them should not change the true label.
    """

    def __init__(
        self,
        *,
        effect_estimator: str = "direct",
        spurious_threshold: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            effect_estimator: Estimator for causal effect ("direct" supported).
            spurious_threshold: Attributes with |TE_a| < threshold are flagged
                as spurious. Default 0.1.
            random_state: Random seed for reproducibility.
        """
        super().__init__(method="causal_effect")
        if effect_estimator != "direct":
            raise ValueError(f"effect_estimator must be 'direct'; got '{effect_estimator}'.")
        if not 0.0 <= spurious_threshold <= 1.0:
            raise ValueError("spurious_threshold must be in [0, 1].")
        self.effect_estimator = effect_estimator
        self.spurious_threshold = spurious_threshold
        self.random_state = int(random_state)

        self.attribute_results_: list[AttributeEffectResult] = []

    def fit(
        self,
        *,
        embeddings: np.ndarray,
        labels: np.ndarray,
        attributes: dict[str, np.ndarray],
        counterfactual_pairs: np.ndarray | list | None = None,
    ) -> CausalEffectDetector:
        """
        Fit causal effect estimator and detect spurious attributes.

        Args:
            embeddings: (n_samples, n_features) representation space.
            labels: (n_samples,) task labels (binary or multi-class).
            attributes: Dict of attribute_name -> (n_samples,) values per sample.
                Binary (0/1) or categorical; multi-valued attributes are binarized.
            counterfactual_pairs: Optional. For interventional data (Phase 2).
                Not used in current Direct estimator.

        Returns:
            self
        """
        X = np.asarray(embeddings, dtype=float)
        y = np.asarray(labels)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("labels must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("embeddings and labels must have same length")

        if not isinstance(attributes, dict) or not attributes:
            raise ValueError("attributes must be a non-empty dict of name -> (n,) array")

        n_samples = X.shape[0]
        for name, arr in attributes.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Attribute names must be non-empty strings")
            arr = np.asarray(arr)
            if arr.ndim != 1 or arr.shape[0] != n_samples:
                raise ValueError(f"attributes['{name}'] must be 1D of length {n_samples}")

        if counterfactual_pairs is not None:
            # Placeholder for Phase 2; Direct estimator ignores for now
            pass

        # Map labels to 0, 1, ... for binary/multi-class
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("At least 2 distinct labels are required")
        y_int = np.searchsorted(classes, y)

        attribute_results: list[AttributeEffectResult] = []

        for attr_name, attr_values in attributes.items():
            effect, n_a0, n_a1 = self._estimate_causal_effect_direct(X, y_int, attr_values)
            is_spurious = abs(effect) < self.spurious_threshold
            attribute_results.append(
                AttributeEffectResult(
                    attribute_name=attr_name,
                    causal_effect=effect,
                    is_spurious=is_spurious,
                    n_samples_a0=n_a0,
                    n_samples_a1=n_a1,
                )
            )

        self.attribute_results_ = attribute_results

        n_spurious = sum(1 for r in attribute_results if r.is_spurious)
        if n_spurious > 1:
            shortcut_detected = True
            risk_level = "high"
            notes = f"Multiple attributes ({n_spurious}) have low causal effect (spurious)."
        elif n_spurious == 1:
            shortcut_detected = True
            risk_level = "moderate"
            spurious_names = [r.attribute_name for r in attribute_results if r.is_spurious]
            notes = f"Attribute '{spurious_names[0]}' has low causal effect (spurious)."
        else:
            shortcut_detected = False
            risk_level = "low"
            notes = "No attribute has estimated causal effect below threshold."

        effects_dict = {r.attribute_name: r.causal_effect for r in attribute_results}
        ranking = sorted(
            attribute_results,
            key=lambda r: abs(r.causal_effect),
            reverse=True,
        )

        metrics = {
            "n_attributes": len(attribute_results),
            "n_spurious": n_spurious,
            "spurious_threshold": self.spurious_threshold,
            "per_attribute_effects": effects_dict,
            "attribute_ranking": [r.attribute_name for r in ranking],
        }

        metadata = {
            "effect_estimator": self.effect_estimator,
            "random_state": self.random_state,
            "attribute_names": list(attributes.keys()),
        }

        report = {
            "per_attribute": [
                {
                    "attribute_name": r.attribute_name,
                    "causal_effect": r.causal_effect,
                    "is_spurious": r.is_spurious,
                    "n_samples_a0": r.n_samples_a0,
                    "n_samples_a1": r.n_samples_a1,
                }
                for r in attribute_results
            ],
        }

        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=notes,
            metadata=metadata,
            report=report,
        )
        self._is_fitted = True
        return self

    def _estimate_causal_effect_direct(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attr: np.ndarray,
    ) -> tuple[float, int, int]:
        """
        Direct estimator: TE_a = E_x[E(Y|X,a=1) - E(Y|X,a=0)].

        Fit E(Y|X,a) via logistic regression on [X, a], then compute
        mean over samples of pred(X,1) - pred(X,0) for binary labels.
        For multi-class labels, use the mean total variation distance
        between predicted class distributions under a=1 vs a=0.
        """
        attr = np.asarray(attr)
        # Binarize if needed: map to 0/1
        unique, counts = np.unique(attr, return_counts=True)
        if len(unique) == 0:
            return 0.0, 0, 0

        is_numeric = np.issubdtype(attr.dtype, np.number)
        if len(unique) > 2:
            if is_numeric:
                attr_num = np.asarray(attr, dtype=float)
                median_val = np.median(attr_num)
                attr_bin = (attr_num > median_val).astype(np.float64)
            else:
                # For categorical attributes, split majority class vs rest.
                majority_class = unique[int(np.argmax(counts))]
                attr_bin = (attr != majority_class).astype(np.float64)
        elif is_numeric:
            attr_num = np.asarray(attr, dtype=float)
            if set(np.unique(attr_num)) - {0.0, 1.0}:
                # Map numeric binary-like labels deterministically to {0,1}.
                attr_bin = (attr_num == np.max(attr_num)).astype(np.float64)
            else:
                attr_bin = attr_num
        else:
            if len(unique) == 1:
                attr_bin = np.zeros(attr.shape[0], dtype=np.float64)
            else:
                # Deterministic binary mapping for non-numeric categorical inputs.
                attr_bin = (attr == unique[1]).astype(np.float64)

        n_a0 = int(np.sum(attr_bin == 0))
        n_a1 = int(np.sum(attr_bin == 1))
        if n_a0 < 2 or n_a1 < 2:
            # Insufficient variation
            return 0.0, n_a0, n_a1

        # Features: [X, a]
        X_with_a = np.hstack([X, attr_bin.reshape(-1, 1)])

        model = LogisticRegression(
            max_iter=2000,
            random_state=self.random_state,
        )
        model.fit(X_with_a, y)

        # TE_a = E_x[E(Y|X,a=1) - E(Y|X,a=0)]
        # For each x, compute pred(x,a=1) - pred(x,a=0), then average
        X_a0 = np.hstack([X, np.zeros((X.shape[0], 1))])
        X_a1 = np.hstack([X, np.ones((X.shape[0], 1))])
        prob_a0 = model.predict_proba(X_a0)
        prob_a1 = model.predict_proba(X_a1)

        if prob_a0.shape[1] == 2:
            te_per_sample = prob_a1[:, 1] - prob_a0[:, 1]
            te = float(np.mean(te_per_sample))
        else:
            te_per_sample = 0.5 * np.sum(np.abs(prob_a1 - prob_a0), axis=1)
            te = float(np.mean(te_per_sample))
        return te, n_a0, n_a1
