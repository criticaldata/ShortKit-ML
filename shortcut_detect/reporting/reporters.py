"""Reporter helpers that encapsulate method-specific logic."""

from __future__ import annotations


class ReporterBase:
    """Base class for method-specific reporting hooks."""

    method: str | None = None

    def applies(self, results: dict[str, dict]) -> bool:
        if self.method is None:
            return False
        return bool(results.get(self.method, {}).get("success"))

    def extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        if self.applies(results):
            self._extend_recommendations(results, recommendations)

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        pass


class HBACReporter(ReporterBase):
    method = "hbac"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        shortcut_info = results["hbac"]["report"]["has_shortcut"]
        if shortcut_info["exists"]:
            recommendations.append(
                "**HBAC detected shortcuts:** Consider investigating the identified "
                "important dimensions. You may want to remove or mask these dimensions."
            )


class ProbeReporter(ReporterBase):
    method = "probe"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        metric_name = results["probe"]["results"]["metrics"]["metric"]
        accuracy = results["probe"]["results"]["metrics"]["metric_value"]
        if accuracy > 0.7:
            recommendations.append(
                f"**High probe {metric_name} ({accuracy:.1%}):** Group information is easily "
                "recoverable from embeddings. Consider adversarial training or "
                "dimension masking techniques."
            )


class StatisticalReporter(ReporterBase):
    method = "statistical"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        stat = results["statistical"]
        significant = stat.get("significant_features")
        if significant is None:
            # Multi-attribute mode: aggregate across attributes
            by_attr = stat.get("by_attribute", {})
            n_sig = 0
            for attr_result in by_attr.values():
                if not isinstance(attr_result, dict):
                    continue
                sf = attr_result.get("significant_features", {})
                if sf:
                    n_sig += sum(1 for v in sf.values() if v is not None and len(v) > 0)
        else:
            n_sig = sum(1 for v in significant.values() if v is not None and len(v) > 0)
        if n_sig > 0:
            recommendations.append(
                f"**{n_sig} group comparisons show significant differences:** "
                "Review these dimensions for potential biases. Consider dimension "
                "removal or reweighting."
            )


class GeometricReporter(ReporterBase):
    method = "geometric"


class EqualizedOddsReporter(ReporterBase):
    method = "equalized_odds"


class EarlyEpochClusteringReporter(ReporterBase):
    method = "early_epoch_clustering"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        report = results["early_epoch_clustering"]["report"]
        if report.risk_level == "high":
            recommendations.append(
                "**Early-epoch clustering detected highly imbalanced clusters:** "
                "Consider resampling or reweighting to reduce reliance on spurious signals."
            )
        elif report.risk_level == "moderate":
            recommendations.append(
                "**Early-epoch clustering shows imbalance:** Review cluster composition and "
                "consider targeted data augmentation or sampling."
            )


class FrequencyReporter(ReporterBase):
    method = "frequency"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        report = results["frequency"].get("report", {})
        detail = report.get("report", {})
        shortcut_classes = detail.get("shortcut_classes", [])
        if shortcut_classes:
            recommendations.append(
                f"**Frequency detector flagged {len(shortcut_classes)} shortcut class(es):** "
                "Review the concentrated embedding dimensions and consider dimension "
                "masking or adversarial training to mitigate shortcut reliance."
            )


class CausalEffectReporter(ReporterBase):
    method = "causal_effect"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        metrics = results["causal_effect"].get("metrics", {})
        n_spurious = metrics.get("n_spurious", 0)
        if n_spurious > 0:
            recommendations.append(
                f"**Causal effect flagged {n_spurious} spurious attribute(s):** "
                "Consider removing or downweighting these attributes in training."
            )


class VAEReporter(ReporterBase):
    method = "vae"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        metrics = results["vae"].get("metrics", {})
        n_flagged = metrics.get("n_flagged", 0)
        if n_flagged > 0:
            recommendations.append(
                f"**VAE flagged {n_flagged} latent dimension(s) as shortcut candidates:** "
                "Review these dimensions for potential spurious correlations."
            )


class CAVReporter(ReporterBase):
    method = "cav"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        n_flagged = sum(
            1
            for row in results["cav"].get("report", {}).get("per_concept", [])
            if row.get("flagged")
        )
        if n_flagged > 0:
            recommendations.append(
                f"**CAV flagged {n_flagged} concept(s) as shortcut-sensitive:** "
                "Review these concepts for potential spurious correlations."
            )


class SISReporter(ReporterBase):
    method = "sis"

    def _extend_recommendations(self, results: dict[str, dict], recommendations: list[str]) -> None:
        shortcut = results["sis"].get("shortcut_detected")
        if shortcut:
            metrics = results["sis"].get("metrics", {})
            mean_sis = metrics.get("mean_sis_size")
            frac_dim = metrics.get("frac_dimensions")
            ms = f"{mean_sis:.1f}" if mean_sis is not None else "N/A"
            fd = f"{frac_dim:.1%}" if frac_dim is not None else "N/A"
            recommendations.append(
                f"**SIS detected small sufficient subsets (mean size {ms}, {fd} of dims):** "
                "Model may rely on few dimensions; consider dimension masking or regularization."
            )


REPORTER_CLASSES = (
    HBACReporter,
    ProbeReporter,
    StatisticalReporter,
    GeometricReporter,
    EqualizedOddsReporter,
    EarlyEpochClusteringReporter,
    FrequencyReporter,
    CausalEffectReporter,
    VAEReporter,
    CAVReporter,
    SISReporter,
)
