"""
Unified API for shortcut detection combining all methods.
"""

import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_builder import BaseDetector
from .conditions import ConditionContext, create_condition
from .reporting.risk_format import apply_standardized_risk
from .utils import set_seed, validate_embeddings_labels

# Reserved keys in extra_labels that are not demographic attributes
_RESERVED_EXTRA_LABELS = {"spurious", "early_epoch_reps"}

# Methods that support multi-attribute (run once per attribute when 2+ attributes)
# GCE omitted: GCEDetector.fit() does not use group_labels; per-attribute runs would be identical
_MULTI_ATTRIBUTE_METHODS = frozenset(
    {
        "equalized_odds",
        "demographic_parity",
        "geometric",
        "statistical",
        "groupdro",
        "bias_direction_pca",
    }
)


def _get_attribute_sources(
    group_labels: np.ndarray | None,
    extra_labels: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Build unified attribute sources for multi-attribute analysis.

    Returns dict of {attr_name: array} from group_labels ("group") and
    extra_labels (excluding reserved keys). Used when len(sources) > 1
    to run single-attribute methods per attribute.
    """
    sources: dict[str, np.ndarray] = {}
    if group_labels is not None:
        sources["group"] = np.asarray(group_labels)
    if extra_labels is not None:
        for k, v in extra_labels.items():
            if k not in _RESERVED_EXTRA_LABELS:
                sources[k] = np.asarray(v)
    return sources


class DetectorFactory:
    """Factory for building detector instances with extensible registrations."""

    _registry: dict[str, type[BaseDetector]] = {}

    def __init__(self, seed: int, kwargs: dict[str, Any] | None = None):
        self.seed = seed
        self.kwargs = dict(kwargs or {})

    @classmethod
    def register(cls, method: str, builder_cls: type[BaseDetector]) -> None:
        """Register a builder for a detector method."""
        cls._registry[method] = builder_cls

    def supported_methods(self) -> list[str]:
        from .discovery import available_method_plugins

        return sorted(set(self._registry) | available_method_plugins())

    def create(self, method: str):
        if method not in self._registry:
            from .discovery import load_method_plugin

            load_method_plugin(method)
        builder_cls = self._registry.get(method)
        if builder_cls is None:
            raise ValueError(f"Detector method '{method}' is not supported.")
        return builder_cls(self.seed, self.kwargs, method=method)


# Builders are registered by plugin modules (discovery.load_all_method_plugins)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .embedding_sources import EmbeddingSource


class ShortcutDetector:
    """
    Unified interface for detecting shortcuts using multiple methods.

    Combines five detection approaches:
    1. HBAC (Hierarchical Bias-Aware Clustering)
    2. Probe-based detection (train classifiers on embeddings)
    3. Statistical testing (feature-wise group differences)
    4. Geometric shortcut analysis (subspace monitoring)
    5. Equalized odds gap analysis (fairness-based)

    Example:
        >>> detector = ShortcutDetector(
        ...     methods=['hbac', 'probe', 'statistical', 'geometric', 'equalized_odds']
        ... )
        >>> results = detector.fit(embeddings, labels, group_labels=groups)
        >>> print(detector.summary())
    """

    def __init__(
        self,
        methods: list[str] = None,
        seed: int = 42,
        condition_name: str = "indicator_count",
        condition_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize unified shortcut detector.

        Args:
            methods: List of methods to use. Options: 'hbac', 'probe', 'statistical'
            seed: Random seed for reproducibility
            condition_name: Registered overall assessment condition name
            condition_kwargs: Keyword arguments passed to the selected condition
            **kwargs: Additional arguments passed to individual detectors
        """
        if methods is None:
            methods = ["hbac", "probe", "statistical"]
        self.methods = methods
        self.seed = seed
        self.condition_name = condition_name
        self.condition_kwargs = dict(condition_kwargs or {})
        create_condition(self.condition_name, **self.condition_kwargs)
        self.kwargs = kwargs
        set_seed(seed)

        # Results storage
        self.results_ = {}
        self.embeddings_ = None
        self.labels_ = None
        self.group_labels_ = None
        self.protected_labels_ = None
        self.embedding_source_ = None
        self.raw_inputs_ = None
        self.embedding_metadata_ = {}
        self.splits_ = None
        self.extra_labels_ = None

        # Initialize detectors
        self._init_detectors()

    def _init_detectors(self):
        """Initialize individual detection components."""
        self.detector_builders_ = {}
        self.detectors_ = {}

        factory = DetectorFactory(self.seed, self.kwargs)
        supported = set(factory.supported_methods())

        for method in self.methods:
            if method not in supported:
                warnings.warn(f"Skipping unsupported detection method '{method}'.", stacklevel=2)
                continue

            self.detector_builders_[method] = factory.create(method)

    def _generate_embeddings_from_source(
        self,
        raw_inputs: Sequence[Any],
        embedding_source: "EmbeddingSource",
        cache_path: str | None,
        force_recompute: bool,
    ) -> np.ndarray:
        """Generate embeddings using the provided source with optional caching."""
        raw_inputs_list = list(raw_inputs)
        use_cache = cache_path is not None and os.path.exists(cache_path) and not force_recompute

        if use_cache:
            embeddings = np.load(cache_path)
            cached = True
        else:
            embeddings = embedding_source.generate(raw_inputs_list)
            cached = False
            if cache_path is not None:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:  # Only create directory if path includes a directory
                    os.makedirs(cache_dir, exist_ok=True)
                np.save(cache_path, embeddings)

        self.embedding_source_ = embedding_source
        self.raw_inputs_ = raw_inputs_list
        self.embedding_metadata_ = {
            "mode": "embedding-only",
            "cached": cached,
            "cache_path": cache_path,
            "source": repr(embedding_source),
        }
        return embeddings

    def fit(
        self,
        embeddings: np.ndarray | None,
        labels: np.ndarray,
        group_labels: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        raw_inputs: Sequence[Any] | None = None,
        embedding_source: "EmbeddingSource | None" = None,
        embedding_cache_path: str | None = None,
        force_embedding_recompute: bool = False,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> "ShortcutDetector":
        """
        Fit all detection methods on embeddings.

        Args:
            embeddings: (n_samples, embedding_dim) array. Optional when using
                embedding-only mode.
            labels: (n_samples,) target labels
            group_labels: (n_samples,) group labels (e.g., demographic attributes)
                         If None, uses `labels` for group-based tests
            feature_names: Optional list of feature names
            raw_inputs: Optional raw inputs (text, ids, etc.). Required when
                `embedding_source` is provided.
            embedding_source: Source used to generate embeddings when they are
                not pre-computed.
            embedding_cache_path: Optional path to cache generated embeddings.
            force_embedding_recompute: Whether to ignore cached embeddings.
            splits: Optional dictionary of named index sets for semi-supervised methods.
                Expected keys: 'train_l' (labeled), 'train_u' (unlabeled).
            extra_labels: Optional dictionary of named per-sample arrays for additional
                supervision signals (e.g., 'spurious' labels). Use -1 for unknown labels.

        Returns:
            self
        """
        if embeddings is None:
            if embedding_source is None or raw_inputs is None:
                raise ValueError(
                    "Provide either `embeddings` directly or both `raw_inputs` and `embedding_source`."
                )
            embeddings = self._generate_embeddings_from_source(
                raw_inputs,
                embedding_source,
                cache_path=embedding_cache_path,
                force_recompute=force_embedding_recompute,
            )
        else:
            if embedding_source is not None or raw_inputs is not None:
                warnings.warn(
                    "embeddings were provided directly; ignoring raw_inputs/embedding_source parameters.",
                    stacklevel=2,
                )
            self.embedding_source_ = None
            self.raw_inputs_ = None
            self.embedding_metadata_ = {"mode": "precomputed", "cached": False}

        # Validate shapes, finite values, and minimum requirements
        embeddings, labels = validate_embeddings_labels(
            embeddings, labels, min_samples=4, min_classes=0, check_finite=True
        )
        # Require at least 2 distinct classes in the effective group signal
        effective_groups = group_labels if group_labels is not None else labels
        if len(np.unique(effective_groups)) < 2:
            raise ValueError(
                "At least 2 distinct classes required in labels "
                "(or group_labels when provided), got 1."
            )
        n_samples = embeddings.shape[0]

        # Validate splits if provided
        if splits is not None:
            if not isinstance(splits, dict):
                raise TypeError("splits must be a dictionary")
            for split_name, indices in splits.items():
                if not isinstance(indices, np.ndarray):
                    raise TypeError(
                        f"Split '{split_name}' must be a numpy array, got {type(indices)}"
                    )
                if indices.ndim != 1:
                    raise ValueError(
                        f"Split '{split_name}' must be 1D array of indices, got shape {indices.shape}"
                    )
                if len(indices) > 0 and (indices.min() < 0 or indices.max() >= n_samples):
                    raise ValueError(
                        f"Split '{split_name}' contains invalid indices (must be in [0, {n_samples}))"
                    )

        # Validate extra_labels if provided
        if extra_labels is not None:
            if not isinstance(extra_labels, dict):
                raise TypeError("extra_labels must be a dictionary")
            for label_name, label_array in extra_labels.items():
                if not isinstance(label_array, np.ndarray):
                    raise TypeError(
                        f"Extra label '{label_name}' must be a numpy array, got {type(label_array)}"
                    )
                if label_array.ndim != 1:
                    raise ValueError(
                        f"Extra label '{label_name}' must be 1D, got shape {label_array.shape}"
                    )
                if len(label_array) != n_samples:
                    raise ValueError(
                        f"Extra label '{label_name}' must have same length as embeddings: {len(label_array)} != {n_samples}"
                    )

        self.embeddings_ = embeddings
        self.labels_ = labels
        self.protected_labels_ = group_labels
        self.group_labels_ = group_labels if group_labels is not None else labels

        self.splits_ = splits
        self.extra_labels_ = extra_labels
        attribute_sources = _get_attribute_sources(group_labels, extra_labels)
        use_multi = len(attribute_sources) > 1

        for method in self.methods:
            builder = self.detector_builders_.get(method)
            if builder is None:
                continue

            if method in _MULTI_ATTRIBUTE_METHODS and use_multi:
                by_attribute: dict[str, dict] = {}
                for attr_name, attr_array in attribute_sources.items():
                    try:
                        res = builder.run(
                            embeddings=embeddings,
                            labels=labels,
                            group_labels=attr_array,
                            feature_names=feature_names,
                            protected_labels=attr_array,
                            splits=self.splits_,
                            extra_labels=self.extra_labels_,
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"Detection for '{method}' (attribute '{attr_name}') failed: {exc}",
                            stacklevel=2,
                        )
                        res = {"success": False, "error": str(exc)}
                    apply_standardized_risk(method, res)
                    by_attribute[attr_name] = res
                result = {
                    "success": any(r.get("success") for r in by_attribute.values()),
                    "by_attribute": by_attribute,
                }
                apply_standardized_risk(method, result)
            else:
                eff_group = self.group_labels_
                eff_protected = self.protected_labels_
                if method in _MULTI_ATTRIBUTE_METHODS and len(attribute_sources) == 1:
                    eff_group = next(iter(attribute_sources.values()))
                    eff_protected = eff_group
                try:
                    result = builder.run(
                        embeddings=embeddings,
                        labels=labels,
                        group_labels=eff_group,
                        feature_names=feature_names,
                        protected_labels=eff_protected,
                        splits=self.splits_,
                        extra_labels=self.extra_labels_,
                    )
                except Exception as exc:
                    warnings.warn(f"Detection for '{method}' failed: {exc}", stacklevel=2)
                    result = {"success": False, "error": str(exc)}
                apply_standardized_risk(method, result)

            self.results_[method] = result
            detector_instance = result.get("detector")
            if detector_instance is not None:
                self.detectors_[method] = detector_instance
            elif "by_attribute" in result:
                for attr_name, sub in result["by_attribute"].items():
                    d = sub.get("detector")
                    if d is not None:
                        self.detectors_[f"{method}_{attr_name}"] = d

        print("✅ Detection complete!")
        return self

    @classmethod
    def from_arrays(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        methods: list[str] | None = None,
        seed: int = 42,
        **kwargs,
    ) -> "ShortcutDetector":
        """
        Convenience constructor for array-based inputs.

        Args:
            embeddings: Embedding matrix.
            labels: Target labels.
            methods: Optional list of methods to run.
            seed: Random seed.
            **kwargs: Passed to ShortcutDetector constructor and fit.
        """
        inst = cls(methods=methods or ["hbac", "probe", "statistical"], seed=seed, **kwargs)
        inst.fit(embeddings, labels, **kwargs)
        return inst

    @classmethod
    def from_loaders(
        cls,
        loaders: dict[str, Any],
        methods: list[str] | None = None,
        seed: int = 42,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> "ShortcutDetector":
        """
        Convenience constructor for loader-based execution.

        Args:
            loaders: Mapping of method name to loader objects.
            methods: Optional list of methods to run.
            seed: Random seed.
            **kwargs: Passed to ShortcutDetector constructor.
        """
        inst = cls(methods=methods or list(loaders.keys()), seed=seed, **kwargs)
        inst.fit_from_loaders(loaders, splits=splits, extra_labels=extra_labels)
        return inst

    def fit_from_loaders(
        self,
        loaders: dict[str, Any],
        feature_names: list[str] | None = None,
        protected_labels: np.ndarray | None = None,
        splits: dict[str, np.ndarray] | None = None,
        extra_labels: dict[str, np.ndarray] | None = None,
    ) -> "ShortcutDetector":
        """
        Execute detectors using user-provided loaders (advanced/streaming use-cases).

        Args:
            loaders: Mapping of method -> loader object (or callable) to be used by that method.
            feature_names: Optional feature names to pass through.
            protected_labels: Optional protected labels to pass through.
        """
        # Minimal placeholders so summary functions do not fail in loader mode
        self.embeddings_ = np.empty((0, 0))
        self.labels_ = np.array([])
        self.group_labels_ = None
        self.protected_labels_ = protected_labels
        self.splits_ = splits
        self.extra_labels_ = extra_labels
        self.embedding_metadata_ = {"mode": "loader", "cached": False}

        for method in self.methods:
            builder = self.detector_builders_.get(method)
            if builder is None:
                continue

            loader = loaders.get(method)
            if loader is None:
                raise ValueError(f"No loader provided for method '{method}'.")

            try:
                result = builder.run_from_loader(
                    loader=loader,
                    feature_names=feature_names,
                    protected_labels=protected_labels,
                    splits=splits,
                    extra_labels=extra_labels,
                )
            except Exception as exc:
                warnings.warn(f"Loader-based detection for '{method}' failed: {exc}", stacklevel=2)
                result = {"success": False, "error": str(exc)}

            apply_standardized_risk(method, result)
            self.results_[method] = result
            detector_instance = result.get("detector")
            if detector_instance is not None:
                self.detectors_[method] = detector_instance

        print("✅ Detection complete (loader mode)!")
        return self

    def summary(self) -> str:
        """
        Generate a text summary of detection results.

        Returns:
            Formatted summary string
        """
        if not self.results_:
            return "No results available. Call .fit() first."

        lines = []
        lines.append("=" * 70)
        lines.append("UNIFIED SHORTCUT DETECTION SUMMARY")
        lines.append("=" * 70)
        lines.append(
            f"Dataset: {len(self.embeddings_)} samples, {self.embeddings_.shape[1]} dimensions"
        )
        lines.append(f"Methods used: {', '.join(self.methods)}")
        lines.append("")

        for method in self.methods:
            result = self.results_.get(method)
            if not result:
                continue

            title = result.get("summary_title") or method.replace("_", " ").title()

            if result.get("success"):
                summary_lines = result.get("summary_lines") or ["Summary unavailable."]
            else:
                # Method failed or was skipped
                error_msg = result.get("error", "Unknown error")
                summary_lines = result.get("summary_lines") or [f"⚠️  Skipped: {error_msg}"]

            lines.append("-" * 70)
            lines.append(title)
            lines.append("-" * 70)
            lines.extend(summary_lines)
            lines.append("")

        # Overall Assessment
        lines.append("=" * 70)
        lines.append("OVERALL ASSESSMENT")
        lines.append("=" * 70)
        lines.append(self._generate_overall_assessment())

        return "\n".join(lines)

    def _generate_overall_assessment(self) -> str:
        """Generate overall shortcut risk assessment."""
        ctx = ConditionContext(methods=self.methods, results=self.results_)
        condition = create_condition(self.condition_name, **self.condition_kwargs)
        return condition.assess(ctx)

    def get_results(self) -> dict[str, Any]:
        """
        Get raw results dictionary.

        Returns:
            Dictionary with results from all methods
        """
        return self.results_

    def generate_report(
        self,
        output_path: str = None,
        format: str = "html",
        include_visualizations: bool = True,
        export_csv: bool = False,
        csv_dir: str = None,
    ):
        """
        Generate comprehensive report with visualizations.

        Args:
            output_path: Path to save report (required for HTML/PDF)
            format: Report format ('html' or 'pdf')
            include_visualizations: Whether to include plots in HTML/PDF
            export_csv: Whether to export results to CSV files
            csv_dir: Directory to save CSV files (default: same dir as output_path)

        Raises:
            ValueError: If format is not supported or required paths are missing
        """
        import os

        from .reporting import ReportBuilder

        builder = ReportBuilder(self)

        # Generate HTML or PDF report
        if format in ["html", "pdf", "markdown"]:
            if not output_path:
                raise ValueError(f"output_path is required for {format} format")

            if format == "html":
                builder.to_html(output_path, include_visualizations=include_visualizations)
            elif format == "pdf":
                builder.to_pdf(output_path, include_visualizations=include_visualizations)
            elif format == "markdown":
                builder.to_markdown(output_path, include_visualizations=include_visualizations)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'html', 'pdf', or 'markdown'")

        # Export CSV if requested
        if export_csv:
            if csv_dir is None:
                # Default to same directory as report
                if output_path:
                    csv_dir = os.path.join(os.path.dirname(output_path) or ".", "csv_results")
                else:
                    csv_dir = "csv_results"

            builder.to_csv(csv_dir)
