"""
Embedding Shortcut Detection Tool
Based on an improved HBAC algorithm, designed to detect shortcuts in model embeddings.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..detector_base import DetectorBase

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class HBACConfig:
    max_iterations: int = 3
    min_cluster_size: float = 0.01
    test_size: float = 0.2
    random_state: int = 42


class EmbeddingShortcutDetector(DetectorBase):
    """
    Detects shortcuts within embeddings.

    Core idea:
    1. If a model relies on shortcuts, the embedding space will form simple linear or nonlinear separations.
    2. Use hierarchical clustering to identify whether simple features can already separate classes.
    3. Analyze feature importance to determine whether the model relies excessively on certain dimensions.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        min_cluster_size: float = 0.01,
        test_size: float = 0.2,
        random_state: int = 42,
        config: HBACConfig | None = None,
    ):
        """
        Parameters:
            max_iterations: Maximum number of hierarchical clustering iterations.
            min_cluster_size: Minimum cluster size (as a proportion of total samples).
            test_size: Proportion of data used for testing.
            random_state: Random seed.
            config: Optional HBACConfig object. If provided, it takes precedence.
        """
        super().__init__(method="hbac")
        # Backward compatibility:
        # - old positional style: HBACDetector(3, 0.05, ...)
        # - config object style:  HBACDetector(HBACConfig(...))
        if isinstance(max_iterations, HBACConfig) and config is None:
            config = max_iterations
            max_iterations = 3
            min_cluster_size = 0.01
            test_size = 0.2
            random_state = 42

        cfg = config or HBACConfig(
            max_iterations=int(max_iterations),
            min_cluster_size=float(min_cluster_size),
            test_size=float(test_size),
            random_state=int(random_state),
        )
        self.config = cfg

        self.max_iterations = cfg.max_iterations
        self.min_cluster_size = cfg.min_cluster_size
        self.test_size = cfg.test_size
        self.random_state = cfg.random_state
        self.scaler = StandardScaler()
        self.clusters_ = []
        self.cluster_labels_ = None
        self.shortcut_report_ = {}

    def fit(
        self, embeddings: np.ndarray, labels: np.ndarray, feature_names: list[str] | None = None
    ) -> "EmbeddingShortcutDetector":
        """
        Fit the shortcut detector.

        Parameters:
            embeddings: np.ndarray of shape (n_samples, embedding_dim).
            labels: np.ndarray of shape (n_samples,).
            feature_names: Optional list of dimension names.
        """
        # Data preprocessing
        self.embeddings = embeddings
        self.labels_raw_ = labels
        unique_labels, labels_encoded = np.unique(labels, return_inverse=True)
        self.label_values_ = unique_labels
        self.labels = labels_encoded
        self.n_samples = len(embeddings)
        self.embedding_dim = embeddings.shape[1]
        self.feature_names = feature_names or [f"dim_{i}" for i in range(self.embedding_dim)]

        # Standardize embeddings
        self.embeddings_scaled = self.scaler.fit_transform(embeddings)

        # Train/test split
        self._train_test_split()

        # Perform Hierarchical Bias-Aware Clustering (HBAC)
        self._hierarchical_bias_aware_clustering()

        # Analyze detected shortcuts
        self._analyze_shortcuts()

        self._finalize_results()
        self._is_fitted = True
        return self

    def _train_test_split(self):
        """Split embeddings into training and testing sets."""
        np.random.seed(self.random_state)
        n_test = int(self.n_samples * self.test_size)
        indices = np.random.permutation(self.n_samples)

        self.test_indices = indices[:n_test]
        self.train_indices = indices[n_test:]

        self.X_train = self.embeddings_scaled[self.train_indices]
        self.y_train = self.labels[self.train_indices]
        self.X_test = self.embeddings_scaled[self.test_indices]
        self.y_test = self.labels[self.test_indices]

    def _hierarchical_bias_aware_clustering(self):
        """
        Hierarchical Bias-Aware Clustering (HBAC)
        Improved for detecting shortcuts rather than bias.
        """
        min_samples = max(int(len(self.X_train) * self.min_cluster_size), 2)

        # Initialize with the entire dataset as one cluster
        current_clusters = [
            {"data_indices": np.arange(len(self.X_train)), "labels": self.y_train, "depth": 0}
        ]

        for _iteration in range(self.max_iterations):
            # Select the cluster with the highest label variance for splitting
            max_std = -1
            cluster_to_split = None

            for cluster in current_clusters:
                label_std = np.std(cluster["labels"])
                if label_std > max_std and len(cluster["data_indices"]) >= 2 * min_samples:
                    max_std = label_std
                    cluster_to_split = cluster

            if cluster_to_split is None:
                break

            # Apply K-means to split the selected cluster
            cluster_data = self.X_train[cluster_to_split["data_indices"]]
            kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_data)

            # Create two subclusters
            new_subclusters = []
            for sub_label in [0, 1]:
                mask = sub_labels == sub_label
                sub_indices = cluster_to_split["data_indices"][mask]

                if len(sub_indices) >= min_samples:
                    new_cluster = {
                        "data_indices": sub_indices,
                        "labels": self.y_train[sub_indices],
                        "depth": cluster_to_split["depth"] + 1,
                        "centroid": kmeans.cluster_centers_[sub_label],
                    }
                    new_subclusters.append(new_cluster)

            # Remove the split cluster and add new subclusters
            # Use list comprehension to avoid numpy array comparison issues
            current_clusters = [c for c in current_clusters if c is not cluster_to_split]
            current_clusters.extend(new_subclusters)

        self.clusters_ = current_clusters

        # Assign cluster labels to training samples
        self.cluster_labels_train = np.zeros(len(self.X_train), dtype=int)
        for i, cluster in enumerate(self.clusters_):
            self.cluster_labels_train[cluster["data_indices"]] = i

    def _analyze_shortcuts(self):
        """Analyze whether shortcuts exist in the embedding space."""
        # 1. Cluster purity
        cluster_purities = []
        for i, cluster in enumerate(self.clusters_):
            labels = cluster["labels"]
            unique, counts = np.unique(labels, return_counts=True)
            dominant_label = self.label_values_[unique[np.argmax(counts)]] if len(unique) else None
            purity = np.max(counts) / len(labels)
            cluster_purities.append(
                {
                    "cluster_id": i,
                    "size": len(labels),
                    "purity": purity,
                    "dominant_label": dominant_label,
                }
            )

        # 2. Dimension importance via ANOVA
        dim_importance = self._compute_dimension_importance()

        # 3. Linearity test
        linearity_score = self._compute_linearity_score()

        # 4. Statistical test between clusters
        statistical_tests = self._perform_statistical_tests()

        # Aggregate report
        self.shortcut_report_ = {
            "cluster_purities": cluster_purities,
            "dimension_importance": dim_importance,
            "linearity_score": linearity_score,
            "statistical_tests": statistical_tests,
            "has_shortcut": self._determine_shortcut_existence(
                cluster_purities, dim_importance, linearity_score
            ),
        }

    def _compute_dimension_importance(self) -> pd.DataFrame:
        """
        Compute the importance of each dimension for clustering
        using ANOVA F-statistics.
        """
        f_scores = []
        p_values = []

        for dim in range(self.embedding_dim):
            dim_values = self.X_train[:, dim]
            groups = [dim_values[cluster["data_indices"]] for cluster in self.clusters_]

            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                f_scores.append(f_stat)
                p_values.append(p_val)
            else:
                f_scores.append(0)
                p_values.append(1)

        importance_df = pd.DataFrame(
            {"dimension": self.feature_names, "f_score": f_scores, "p_value": p_values}
        ).sort_values("f_score", ascending=False)

        return importance_df

    def _compute_linearity_score(self) -> float:
        """
        Evaluate linear separability of the embedding space.
        High linear separability implies the presence of simple shortcuts.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # Train a simple linear classifier
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        lr.fit(self.X_train, self.y_train)

        # Evaluate accuracy
        train_acc = accuracy_score(self.y_train, lr.predict(self.X_train))
        test_acc = accuracy_score(self.y_test, lr.predict(self.X_test))

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfitting": train_acc - test_acc,
        }

    def _perform_statistical_tests(self) -> dict:
        """
        Perform statistical tests between the most deviant cluster and the rest.
        """
        cluster_purities = [
            np.max(np.bincount(c["labels"])) / len(c["labels"]) for c in self.clusters_
        ]
        most_deviant_idx = np.argmax(cluster_purities)
        most_deviant_cluster = self.clusters_[most_deviant_idx]

        deviant_labels = most_deviant_cluster["labels"]
        rest_indices = np.setdiff1d(
            np.arange(len(self.y_train)), most_deviant_cluster["data_indices"]
        )
        rest_labels = self.y_train[rest_indices]

        # Z-test for proportions
        p1 = np.mean(deviant_labels)
        p2 = np.mean(rest_labels)
        n1 = len(deviant_labels)
        n2 = len(rest_labels)

        p_pool = (n1 * p1 + n2 * p2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        z_score = (p1 - p2) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "most_deviant_cluster": most_deviant_idx,
            "deviant_label_mean": p1,
            "rest_label_mean": p2,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    def _determine_shortcut_existence(
        self, cluster_purities: list[dict], dim_importance: pd.DataFrame, linearity_score: dict
    ) -> dict:
        """
        Integrate multiple signals to determine whether shortcuts exist.
        """
        high_purity_clusters = sum(1 for c in cluster_purities if c["purity"] > 0.8)
        few_important_dims = sum(dim_importance["p_value"] < 0.01) < self.embedding_dim * 0.1
        high_linearity = linearity_score["test_accuracy"] > 0.85

        has_shortcut = high_purity_clusters >= 2 or (few_important_dims and high_linearity)

        shortcut_type = []
        if high_purity_clusters >= 2:
            shortcut_type.append("clustering_based")
        if few_important_dims:
            shortcut_type.append("dimension_specific")
        if high_linearity:
            shortcut_type.append("linear_separable")

        return {
            "exists": has_shortcut,
            "confidence": (
                "high" if len(shortcut_type) >= 2 else "medium" if shortcut_type else "low"
            ),
            "types": shortcut_type,
            "evidence": {
                "high_purity_clusters": high_purity_clusters,
                "important_dims_ratio": sum(dim_importance["p_value"] < 0.01) / self.embedding_dim,
                "linear_test_accuracy": linearity_score["test_accuracy"],
            },
        }

    def visualize(self, method: str = "pca", save_path: str | None = None):
        """
        Visualize the embedding space and detected shortcuts.

        Parameters:
            method: 'pca' or 'tsne'
            save_path: Optional save path for the visualization
        """
        import matplotlib.pyplot as plt

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=self.random_state)
        else:
            reducer = TSNE(n_components=2, random_state=self.random_state)

        embeddings_2d = reducer.fit_transform(self.embeddings_scaled)

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot 1: Original labels
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1], c=self.labels, cmap="viridis", alpha=0.6
        )
        axes[0].set_title("Original Labels")
        axes[0].set_xlabel(f"{method.upper()} 1")
        axes[0].set_ylabel(f"{method.upper()} 2")
        plt.colorbar(scatter1, ax=axes[0])

        # Subplot 2: Cluster assignments
        train_2d = embeddings_2d[self.train_indices]
        scatter2 = axes[1].scatter(
            train_2d[:, 0], train_2d[:, 1], c=self.cluster_labels_train, cmap="tab10", alpha=0.6
        )
        axes[1].set_title("Detected Clusters")
        axes[1].set_xlabel(f"{method.upper()} 1")
        axes[1].set_ylabel(f"{method.upper()} 2")
        plt.colorbar(scatter2, ax=axes[1])

        # Subplot 3: Top important dimensions
        top_dims = self.shortcut_report_["dimension_importance"].head(10)
        axes[2].barh(range(len(top_dims)), top_dims["f_score"])
        axes[2].set_yticks(range(len(top_dims)))
        axes[2].set_yticklabels(top_dims["dimension"])
        axes[2].set_xlabel("F-score")
        axes[2].set_title("Top 10 Important Dimensions")
        axes[2].invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def _finalize_results(self) -> None:
        shortcut_info = self.shortcut_report_.get("has_shortcut", {})
        exists = shortcut_info.get("exists")
        confidence = shortcut_info.get("confidence", "low")
        confidence_map = {"high": "high", "medium": "moderate", "low": "low"}
        risk_level = confidence_map.get(str(confidence).lower(), "unknown")
        if exists is False:
            risk_level = "low"

        evidence = shortcut_info.get("evidence", {})
        metrics = {
            "n_clusters": len(self.clusters_),
            "linearity_score": evidence.get("linear_test_accuracy"),
            "important_dims_ratio": evidence.get("important_dims_ratio"),
            "high_purity_clusters": evidence.get("high_purity_clusters"),
        }
        metadata = {
            "n_samples": getattr(self, "n_samples", None),
            "embedding_dim": getattr(self, "embedding_dim", None),
            "max_iterations": self.max_iterations,
            "min_cluster_size": self.min_cluster_size,
            "test_size": self.test_size,
        }

        self.shortcut_detected_ = exists
        self._set_results(
            shortcut_detected=exists,
            risk_level=risk_level,
            metrics=metrics,
            notes="HBAC clustering-based shortcut detection.",
            metadata=metadata,
            report=self.shortcut_report_,
        )

    def get_report(self) -> dict:
        return super().get_report()

    def get_report_text(self) -> str:
        """
        Generate a readable text report of shortcut detection results.
        """
        report = []
        report.append("=" * 60)
        report.append("Embedding Shortcut Detection Report")
        report.append("=" * 60)
        report.append(f"\nDataset: {self.n_samples} samples, {self.embedding_dim} dimensions")
        report.append(f"Clusters found: {len(self.clusters_)}")

        # Shortcut detection summary
        shortcut_info = self.shortcut_report_["has_shortcut"]
        report.append(f"\n{'='*60}")
        report.append("SHORTCUT DETECTION RESULT")
        report.append(f"{'='*60}")
        report.append(f"Shortcuts detected: {'YES' if shortcut_info['exists'] else 'NO'}")
        report.append(f"Confidence: {shortcut_info['confidence']}")
        if shortcut_info["types"]:
            report.append(f"Shortcut types: {', '.join(shortcut_info['types'])}")

        # Evidence
        report.append("\nEvidence:")
        evidence = shortcut_info["evidence"]
        report.append(f"  - High purity clusters: {evidence['high_purity_clusters']}")
        report.append(f"  - Important dimensions ratio: {evidence['important_dims_ratio']:.2%}")
        report.append(f"  - Linear separability: {evidence['linear_test_accuracy']:.2%}")

        # Cluster analysis
        report.append(f"\n{'='*60}")
        report.append("CLUSTER ANALYSIS")
        report.append(f"{'='*60}")
        for cp in self.shortcut_report_["cluster_purities"]:
            report.append(
                f"Cluster {cp['cluster_id']}: "
                f"size={cp['size']}, purity={cp['purity']:.2%}, "
                f"dominant_label={cp['dominant_label']}"
            )

        # Top important dimensions
        report.append(f"\n{'='*60}")
        report.append("TOP IMPORTANT DIMENSIONS")
        report.append(f"{'='*60}")
        top_dims = self.shortcut_report_["dimension_importance"].head(5)
        for _, row in top_dims.iterrows():
            report.append(
                f"{row['dimension']}: F-score={row['f_score']:.2f}, "
                f"p-value={row['p_value']:.4f}"
            )

        # Statistical tests
        report.append(f"\n{'='*60}")
        report.append("STATISTICAL TESTS")
        report.append(f"{'='*60}")
        stats_test = self.shortcut_report_["statistical_tests"]
        report.append(f"Most deviant cluster: {stats_test['most_deviant_cluster']}")
        report.append(f"Z-score: {stats_test['z_score']:.2f}")
        report.append(f"P-value: {stats_test['p_value']:.4f}")
        report.append(f"Significant: {stats_test['significant']}")

        return "\n".join(report)


# # Example usage
# if __name__ == "__main__":
#     np.random.seed(42)
#     n_samples = 1000
#     embedding_dim = 128

#     # Simulated embeddings with shortcuts
#     embeddings = np.random.randn(n_samples, embedding_dim)
#     labels = np.random.randint(0, 2, n_samples)

#     # Inject shortcut correlation into first few dimensions
#     embeddings[labels == 0, 0] += 2
#     embeddings[labels == 1, 0] -= 2
#     embeddings[labels == 0, 1] += 1.5
#     embeddings[labels == 1, 1] -= 1.5

#     detector = EmbeddingShortcutDetector(
#         max_iterations=3,
#         min_cluster_size=0.05,
#         random_state=42
#     )

#     detector.fit(embeddings, labels)
#     print(detector.get_report())
#     detector.visualize(method='pca')
