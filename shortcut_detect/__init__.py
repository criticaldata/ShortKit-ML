"""ShortKit-ML - Detect biases in embedding spaces."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS: dict[str, str] = {
    "AdversarialDebiasing": ".mitigation",
    "AttentionOverlapResult": ".gradcam",
    "BackgroundRandomizer": ".mitigation",
    "BenchmarkConfig": ".benchmark",
    "BenchmarkRunner": ".benchmark",
    "BiasDirectionPCADetector": ".geometric",
    "BiasDirectionPCAReport": ".geometric",
    "CAVDetector": ".xai",
    "CallableEmbeddingSource": ".embedding_sources",
    "ComparisonResult": ".comparison",
    "ConditionContext": ".conditions",
    "ContrastiveDebiasing": ".mitigation",
    "CausalEffectDetector": ".causal",
    "DemographicParityDetector": ".fairness",
    "DemographicParityReport": ".fairness",
    "DetectorBase": ".detector_base",
    "EarlyEpochClusteringDetector": ".training",
    "EarlyEpochClusteringReport": ".training",
    "EmbeddingModelRegistry": ".model_registry",
    "EmbeddingSource": ".embedding_sources",
    "EqualizedOddsDetector": ".fairness",
    "EqualizedOddsReport": ".fairness",
    "ExplanationRegularization": ".mitigation",
    "FrequencyDetector": ".frequency",
    "GeometricShortcutAnalyzer": ".geometric",
    "GradCAMHeatmapGenerator": ".gradcam",
    "GradCAMMaskOverlapDetector": ".xai",
    "GroupDiffTest": ".statistical",
    "HBACDetector": ".clustering",
    "HuggingFaceEmbeddingSource": ".embedding_sources",
    "IntersectionalDetector": ".fairness",
    "IntersectionalReport": ".fairness",
    "LastLayerRetraining": ".mitigation",
    "MetaClassifierCondition": ".conditions",
    "ModelComparisonRunner": ".comparison",
    "MAX_SHORTCUT_EFFECT_SIZE": ".benchmark",
    "MIN_SHORTCUT_EFFECT_SIZE": ".benchmark",
    "MultiAttributeCondition": ".conditions",
    "PaperBenchmarkConfig": ".benchmark",
    "PaperBenchmarkRunner": ".benchmark",
    "RiskCondition": ".conditions",
    "SISDetector": ".xai",
    "SKLearnProbe": ".probes",
    "SSADetector": ".ssa",
    "ShortcutDetector": ".unified",
    "ShortcutMasker": ".mitigation",
    "SpRAyDetector": ".xai",
    "SyntheticShortcutConfig": ".benchmark",
    "SyntheticShortcutDataset": ".benchmark",
    "TorchProbe": ".probes",
    "VAEDetector": ".vae",
    "WeightedRiskCondition": ".conditions",
    "available_conditions": ".conditions",
    "create_condition": ".conditions",
    "generate_parametric_shortcut_dataset": ".benchmark",
    "generate_linear_shortcut": ".datasets",
    "generate_multiclass_shortcut": ".datasets",
    "generate_no_shortcut": ".datasets",
    "generate_nonlinear_shortcut": ".datasets",
    "get_embedding_registry": ".model_registry",
    "list_embedding_models": ".model_registry",
    "metrics_registry": ".metrics",
    "register_condition": ".conditions",
    "run_benchmark": ".benchmark",
    "run_paper_benchmark": ".benchmark",
    "set_seed": ".utils",
    "train_test_split": ".utils",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
