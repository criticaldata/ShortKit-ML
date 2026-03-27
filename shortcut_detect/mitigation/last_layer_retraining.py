"""
Last Layer Retraining (M06 DFR) - Kirichenko et al. 2023.

Retrain only the last linear layer on a group-balanced subset of embeddings.
Fixes spurious-correlation bias with minimal complexity (sklearn only).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LastLayerRetraining:
    """
    Last Layer Retraining (DFR) per Kirichenko et al. 2023.

    Retrains only the last linear layer (logistic regression) on a group-balanced
    subset of embeddings. The embeddings stay frozen; only the classifier is retrained.
    This simple approach can match or outperform more complex debiasing methods.

    Parameters
    ----------
    C : float
        Inverse regularization strength. Smaller values = stronger regularization.
        Default 1.0.
    penalty : str
        Regularization penalty: "l1" or "l2". Default "l1".
    solver : str
        Solver for LogisticRegression. "liblinear" works for both L1 and L2.
        Default "liblinear".
    class_weight : str or dict
        Class weights for imbalanced task labels. "balanced" or None.
        Default "balanced".
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l1",
        solver: str = "liblinear",
        class_weight: str | dict | None = "balanced",
        random_state: int | None = None,
    ):
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.C = float(C)
        self.penalty = penalty
        self.solver = solver
        self.class_weight = class_weight
        self.random_state = random_state

        self._scaler: StandardScaler | None = None
        self._classifier: LogisticRegression | None = None
        self._task_map: dict | None = None  # original label -> index
        self._embed_dim: int | None = None
        self._n_balanced: int | None = None
        self._n_groups: int | None = None
        self._fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        task_labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> LastLayerRetraining:
        """
        Build group-balanced subset and fit the logistic regression classifier.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, embed_dim).
        task_labels : np.ndarray
            Shape (n_samples,) – task/target labels to predict.
        group_labels : np.ndarray
            Shape (n_samples,) – protected/group labels for balancing.

        Returns
        -------
        self : LastLayerRetraining
        """
        X = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(task_labels)
        g = np.asarray(group_labels)

        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, embed_dim)")
        if y.ndim != 1 or g.ndim != 1:
            raise ValueError("task_labels and group_labels must be 1D")
        if X.shape[0] != y.shape[0] or X.shape[0] != g.shape[0]:
            raise ValueError("embeddings, task_labels, and group_labels must have same length")

        # Map task labels to 0..n_classes-1
        y_uniq = np.unique(y)
        task_map = {v: i for i, v in enumerate(y_uniq)}
        y_idx = np.array([task_map[v] for v in y], dtype=np.int64)

        # Map group labels to 0..n_groups-1
        g_uniq = np.unique(g)
        g_map = {v: i for i, v in enumerate(g_uniq)}
        g_idx = np.array([g_map[v] for v in g], dtype=np.int64)

        n_groups = len(g_uniq)
        g_indices = [np.where(g_idx == grp)[0] for grp in range(n_groups)]
        min_g = min(len(gi) for gi in g_indices)

        if min_g == 0:
            raise ValueError(
                "At least one group has 0 samples. "
                "Cannot build balanced subset. Check group_labels."
            )

        # Build balanced subset: take min_g samples per group
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()

        balanced_indices = []
        for gi in g_indices:
            idx = gi.copy()
            rng.shuffle(idx)
            balanced_indices.extend(idx[:min_g])

        X_bal = X[balanced_indices]
        y_bal = y_idx[balanced_indices]

        # Preprocess
        scaler = StandardScaler()
        X_bal_scaled = scaler.fit_transform(X_bal)

        # Fit classifier (liblinear is binary-only; saga supports multiclass+L1, lbfgs supports multiclass+L2)
        n_classes = len(np.unique(y_bal))
        solver = self.solver
        if n_classes > 2:
            solver = "saga" if self.penalty == "l1" else "lbfgs"

        clf = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=solver,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=1000,
        )
        clf.fit(X_bal_scaled, y_bal)

        self._scaler = scaler
        self._classifier = clf
        self._task_map = task_map
        self._embed_dim = X.shape[1]
        self._n_balanced = len(balanced_indices)
        self._n_groups = n_groups
        self._fitted = True
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict task labels for given embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, embed_dim). Must match embed_dim from fit.

        Returns
        -------
        np.ndarray
            Predicted task labels (original label values, not indices).
        """
        if not self._fitted or self._classifier is None or self._scaler is None:
            raise ValueError("LastLayerRetraining has not been fitted")
        X = np.asarray(embeddings, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if X.shape[1] != self._embed_dim:
            raise ValueError(
                f"embed_dim {X.shape[1]} does not match fitted embed_dim {self._embed_dim}"
            )

        X_scaled = self._scaler.transform(X)
        pred_idx = self._classifier.predict(X_scaled)

        # Map indices back to original label values
        inv_map = {i: v for v, i in self._task_map.items()}
        return np.array([inv_map[int(p)] for p in pred_idx])

    def fit_predict(
        self,
        embeddings: np.ndarray,
        task_labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and predict in one call."""
        self.fit(embeddings, task_labels, group_labels)
        return self.predict(embeddings)

    @property
    def scaler_(self) -> StandardScaler | None:
        """Fitted StandardScaler (None if not fitted)."""
        return self._scaler

    @property
    def classifier_(self) -> LogisticRegression | None:
        """Fitted LogisticRegression (None if not fitted)."""
        return self._classifier
