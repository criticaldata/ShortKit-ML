"""Concept Activation Vectors (CAV) detector for shortcut concept testing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ....detector_base import DetectorBase

ArrayLike = np.ndarray | Sequence[np.ndarray]


@dataclass
class ConceptResult:
    """Per-concept CAV summary."""

    concept_name: str
    n_concept: int
    n_random: int
    quality_auc: float
    tcav_score: float | None
    activation_mean: float | None
    activation_p95: float | None
    flagged: bool


@dataclass(frozen=True)
class CAVConfig:
    classifier: str = "logreg"
    random_state: int = 42
    test_size: float = 0.2
    min_examples_per_set: int = 20
    shortcut_threshold: float = 0.6
    quality_threshold: float = 0.7


class CAVDetector(DetectorBase):
    """Test shortcut concepts using Concept Activation Vectors (Kim et al., 2018)."""

    def __init__(
        self,
        *,
        config: CAVConfig | None = None,
        classifier: str = "logreg",
        random_state: int = 42,
        test_size: float = 0.2,
        min_examples_per_set: int = 20,
        shortcut_threshold: float = 0.6,
        quality_threshold: float = 0.7,
    ) -> None:
        super().__init__(method="cav")
        cfg = config or CAVConfig(
            classifier=classifier,
            random_state=int(random_state),
            test_size=float(test_size),
            min_examples_per_set=int(min_examples_per_set),
            shortcut_threshold=float(shortcut_threshold),
            quality_threshold=float(quality_threshold),
        )
        self.config = cfg
        if cfg.classifier != "logreg":
            raise ValueError("Only classifier='logreg' is currently supported for CAVDetector.")
        if not 0.0 < cfg.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1).")
        if cfg.min_examples_per_set < 2:
            raise ValueError("min_examples_per_set must be >= 2.")
        if not 0.0 <= cfg.shortcut_threshold <= 1.0:
            raise ValueError("shortcut_threshold must be in [0, 1].")
        if not 0.0 <= cfg.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be in [0, 1].")

        self.classifier = cfg.classifier
        self.random_state = int(cfg.random_state)
        self.test_size = float(cfg.test_size)
        self.min_examples_per_set = int(cfg.min_examples_per_set)
        self.shortcut_threshold = float(cfg.shortcut_threshold)
        self.quality_threshold = float(cfg.quality_threshold)

        self.cav_vectors_: dict[str, np.ndarray] = {}
        self.concept_results_: list[ConceptResult] = []

    def fit(
        self,
        *,
        concept_sets: dict[str, np.ndarray],
        random_set: ArrayLike,
        target_activations: np.ndarray | None = None,
        target_directional_derivatives: np.ndarray | None = None,
    ) -> CAVDetector:
        """Fit CAVs for concept-vs-random discrimination and compute TCAV metrics."""
        concept_sets_arr = self._validate_concept_sets(concept_sets)
        random_map = self._normalize_random_set(random_set, concept_sets_arr.keys())

        dim = next(iter(concept_sets_arr.values())).shape[1]
        target_acts_arr = self._validate_optional_matrix(
            target_activations, dim, "target_activations"
        )
        target_dd_arr = self._validate_optional_matrix(
            target_directional_derivatives,
            dim,
            "target_directional_derivatives",
        )

        concept_results: list[ConceptResult] = []
        cav_vectors: dict[str, np.ndarray] = {}

        for concept_name, concept_examples in concept_sets_arr.items():
            random_examples = random_map[concept_name]
            self._validate_min_examples(concept_examples, random_examples, concept_name)

            X = np.vstack([concept_examples, random_examples])
            y = np.concatenate(
                [
                    np.ones(concept_examples.shape[0], dtype=int),
                    np.zeros(random_examples.shape[0], dtype=int),
                ]
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )

            model = LogisticRegression(max_iter=2000, random_state=self.random_state)
            model.fit(X_train, y_train)

            coef = np.asarray(model.coef_[0], dtype=float)
            norm = float(np.linalg.norm(coef))
            if norm <= 0.0:
                raise ValueError(f"Concept '{concept_name}' produced a zero-norm CAV vector.")
            cav_vector = coef / norm
            cav_vectors[concept_name] = cav_vector

            y_prob = model.predict_proba(X_test)[:, 1]
            quality_auc = float(roc_auc_score(y_test, y_prob))

            tcav_score: float | None = None
            if target_dd_arr is not None:
                directional = target_dd_arr @ cav_vector
                tcav_score = float(np.mean(directional > 0.0))

            activation_mean: float | None = None
            activation_p95: float | None = None
            if target_acts_arr is not None:
                projections = target_acts_arr @ cav_vector
                activation_mean = float(np.mean(projections))
                activation_p95 = float(np.percentile(projections, 95))

            flagged = (
                tcav_score is not None
                and quality_auc >= self.quality_threshold
                and tcav_score >= self.shortcut_threshold
            )

            concept_results.append(
                ConceptResult(
                    concept_name=concept_name,
                    n_concept=int(concept_examples.shape[0]),
                    n_random=int(random_examples.shape[0]),
                    quality_auc=quality_auc,
                    tcav_score=tcav_score,
                    activation_mean=activation_mean,
                    activation_p95=activation_p95,
                    flagged=bool(flagged),
                )
            )

        self.cav_vectors_ = cav_vectors
        self.concept_results_ = concept_results

        tcav_values = [r.tcav_score for r in concept_results if r.tcav_score is not None]
        quality_values = [r.quality_auc for r in concept_results]
        n_flagged = int(sum(1 for r in concept_results if r.flagged))

        if not tcav_values:
            shortcut_detected = None
            risk_level = "unknown"
            notes = "Directional derivatives were not provided; TCAV scores unavailable."
        else:
            max_tcav = float(max(tcav_values))
            if n_flagged > 0:
                shortcut_detected = True
                risk_level = "high" if max_tcav >= 0.75 else "moderate"
                notes = "At least one concept exceeded quality and TCAV thresholds."
            else:
                shortcut_detected = False
                risk_level = "low"
                notes = "No concept exceeded both quality and TCAV thresholds."

        metrics = {
            "n_concepts": int(len(concept_results)),
            "n_tested": int(len(tcav_values)),
            "max_tcav_score": float(max(tcav_values)) if tcav_values else None,
            "mean_tcav_score": float(np.mean(tcav_values)) if tcav_values else None,
            "max_concept_quality": float(max(quality_values)) if quality_values else None,
            "n_flagged": n_flagged,
            "shortcut_threshold": self.shortcut_threshold,
            "quality_threshold": self.quality_threshold,
        }

        metadata = {
            "classifier": self.classifier,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "min_examples_per_set": self.min_examples_per_set,
            "concept_names": [r.concept_name for r in concept_results],
            "has_target_activations": target_acts_arr is not None,
            "has_target_directional_derivatives": target_dd_arr is not None,
        }

        report = {
            "per_concept": [self._concept_result_to_dict(result) for result in concept_results],
        }

        details = {
            "cav_vectors": {name: vec.tolist() for name, vec in cav_vectors.items()},
        }

        self._set_results(
            shortcut_detected=shortcut_detected,
            risk_level=risk_level,
            metrics=metrics,
            notes=notes,
            metadata=metadata,
            report=report,
            details=details,
        )
        self._is_fitted = True
        return self

    def _validate_concept_sets(self, concept_sets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not isinstance(concept_sets, dict) or not concept_sets:
            raise ValueError(
                "concept_sets must be a non-empty dictionary of concept_name -> 2D arrays."
            )

        out: dict[str, np.ndarray] = {}
        expected_dim: int | None = None
        for name, arr in concept_sets.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("All concept names must be non-empty strings.")
            concept_arr = np.asarray(arr, dtype=float)
            if concept_arr.ndim != 2:
                raise ValueError(
                    f"concept_sets['{name}'] must be 2D, got shape {concept_arr.shape}."
                )
            if expected_dim is None:
                expected_dim = concept_arr.shape[1]
            elif concept_arr.shape[1] != expected_dim:
                raise ValueError(
                    f"All concept sets must share feature dimension {expected_dim}; "
                    f"concept '{name}' has {concept_arr.shape[1]}."
                )
            out[name] = concept_arr

        return out

    def _normalize_random_set(
        self,
        random_set: ArrayLike,
        concept_names: Iterable[str],
    ) -> dict[str, np.ndarray]:
        names = list(concept_names)

        if isinstance(random_set, np.ndarray):
            arr = np.asarray(random_set, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"random_set must be 2D, got shape {arr.shape}.")
            return dict.fromkeys(names, arr)

        if not isinstance(random_set, Sequence) or len(random_set) == 0:
            raise ValueError("random_set must be a 2D array or a non-empty sequence of 2D arrays.")

        random_arrays = [np.asarray(item, dtype=float) for item in random_set]
        if len(random_arrays) == 1:
            arr = random_arrays[0]
            if arr.ndim != 2:
                raise ValueError(f"random_set[0] must be 2D, got shape {arr.shape}.")
            return dict.fromkeys(names, arr)

        if len(random_arrays) != len(names):
            raise ValueError(
                "When random_set is a sequence with multiple arrays, it must have "
                "the same length as concept_sets."
            )

        out: dict[str, np.ndarray] = {}
        for name, arr in zip(names, random_arrays, strict=False):
            if arr.ndim != 2:
                raise ValueError(
                    f"random_set for concept '{name}' must be 2D, got shape {arr.shape}."
                )
            out[name] = arr
        return out

    def _validate_optional_matrix(
        self,
        value: np.ndarray | None,
        expected_dim: int,
        field_name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{field_name} must be 2D, got shape {arr.shape}.")
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"{field_name} feature dimension mismatch: expected {expected_dim}, got {arr.shape[1]}."
            )
        return arr

    def _validate_min_examples(
        self,
        concept_examples: np.ndarray,
        random_examples: np.ndarray,
        concept_name: str,
    ) -> None:
        if concept_examples.shape[1] != random_examples.shape[1]:
            raise ValueError(
                f"Dimension mismatch for concept '{concept_name}': "
                f"concept dim={concept_examples.shape[1]}, random dim={random_examples.shape[1]}."
            )
        if concept_examples.shape[0] < self.min_examples_per_set:
            raise ValueError(
                f"Concept '{concept_name}' has {concept_examples.shape[0]} examples; "
                f"requires at least {self.min_examples_per_set}."
            )
        if random_examples.shape[0] < self.min_examples_per_set:
            raise ValueError(
                f"Random set for concept '{concept_name}' has {random_examples.shape[0]} examples; "
                f"requires at least {self.min_examples_per_set}."
            )

    @staticmethod
    def _concept_result_to_dict(result: ConceptResult) -> dict[str, float | None]:
        return {
            "concept_name": result.concept_name,
            "n_concept": result.n_concept,
            "n_random": result.n_random,
            "quality_auc": result.quality_auc,
            "tcav_score": result.tcav_score,
            "activation_mean": result.activation_mean,
            "activation_p95": result.activation_p95,
            "flagged": result.flagged,
        }
