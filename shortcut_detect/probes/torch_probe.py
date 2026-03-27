# shortcut_detect/detectors/torch_probe.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from shortcut_detect.detector_base import DetectorBase
from shortcut_detect.utils import validate_embeddings_labels

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

    from shortcut_detect.training.data_adapters import (
        DataSpec,
        extract_xy_batch,
        is_iterable_dataset,
        resolve_data_spec,
        safe_len,
    )
    from shortcut_detect.training.loader_hooks import (
        LoaderFactory,
        LoaderRequest,
        StageLoaderOverrides,
        build_loader,
    )
except Exception as e:  # pragma: no cover
    raise ImportError("TorchProbe requires PyTorch. Install it with: pip install torch") from e


MetricName = Literal["accuracy", "f1", "roc_auc", "loss"]


def _is_binary(y: np.ndarray) -> bool:
    return np.unique(y).shape[0] == 2


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # positive class assumed to be 1
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0.0 or denom_r == 0.0:
        return 0.0
    p = tp / denom_p
    r = tp / denom_r
    denom = p + r
    return float(0.0 if denom == 0.0 else (2.0 * p * r) / denom)


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    scores = []
    for c in classes:
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        scores.append(_f1_binary(yt, yp))
    return float(np.mean(scores)) if scores else 0.0


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # AUC via rank statistic (equivalent to Mann–Whitney U)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks of all scores (average ranks for ties)
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)

    # handle ties
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i + 1
        while j < sorted_scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = float(np.sum(ranks[:n_pos]))  # pos are first in concatenation
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


@dataclass(frozen=True)
class TorchProbeConfig:
    metric: MetricName = "accuracy"
    threshold: float = 0.70
    evaluation: Literal["holdout"] = "holdout"  # keep simple; can add "cv" later
    test_size: float = 0.2
    random_state: int = 0

    # training
    epochs: int = 10
    batch_size: int = 128
    num_workers: int = 0
    early_stopping: int | None = None
    use_amp: bool = False
    verbose: bool = False


class TorchProbe(DetectorBase):
    """Probe-based shortcut detector using a PyTorch model.

    Fits a torch model to predict a demographic target from embeddings and flags a shortcut if
    the chosen metric exceeds a threshold.

    Parameters
    ----------
    model:
        torch.nn.Module that maps embeddings -> logits (classification) or scalar (regression).
        For classification, return shape (N, C) logits.
    loss_fn:
        Loss function (e.g., nn.CrossEntropyLoss()).
    optimizer_class / optimizer_kwargs:
        Optimizer configuration.
    device:
        "cpu" or "cuda"; defaults to CUDA if available.
    metric:
        One of: "accuracy", "f1", "roc_auc", "loss".
        For multiclass, "f1" uses macro averaging; "roc_auc" only supported for binary.
    threshold:
        shortcut_detected is True when metric_value > threshold (except for "loss", where
        shortcut_detected is True when loss < threshold if you choose to use loss; see notes).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Any,
        *,
        optimizer_class: Any = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        device: str | None = None,
        metric: MetricName = "accuracy",
        threshold: float = 0.70,
        test_size: float = 0.2,
        random_state: int = 0,
        epochs: int = 10,
        batch_size: int = 128,
        num_workers: int = 0,
        early_stopping: int | None = None,
        use_amp: bool = False,
        verbose: bool = False,
        loader_factory: LoaderFactory | None = None,
        stage_loader_overrides: StageLoaderOverrides = None,
    ):
        super().__init__(method="torch_probe")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config = TorchProbeConfig(
            metric=metric,
            threshold=float(threshold),
            test_size=float(test_size),
            random_state=int(random_state),
            epochs=int(epochs),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            early_stopping=early_stopping,
            use_amp=bool(use_amp),
            verbose=bool(verbose),
        )

        self.shortcut_detected_: bool | None = None
        self.metric_value_: float | None = None

        self._optimizer: torch.optim.Optimizer | None = None
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self.loader_factory = loader_factory
        self.stage_loader_overrides = stage_loader_overrides

    def fit(self, embeddings: np.ndarray, target: np.ndarray) -> TorchProbe:
        X, y = validate_embeddings_labels(embeddings, target, min_samples=4)

        # holdout split
        rng = np.random.RandomState(self.config.random_state)
        n = X.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)

        # stratify for classification (best-effort)
        # if too small or many classes, fall back to simple split
        test_n = max(1, int(round(self.config.test_size * n)))
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # train
        train_loss_hist, best_val_loss, best_epoch = self._train_holdout(X_tr, y_tr, X_te, y_te)

        # evaluate
        eval_metrics = self._evaluate_np(X_te, y_te)
        metric = self.config.metric
        if metric not in eval_metrics:
            raise ValueError(
                f"Metric '{metric}' not available. Available: {sorted(eval_metrics.keys())}"
            )
        metric_value = float(eval_metrics[metric])
        self.metric_value_ = metric_value

        # detection rule
        if metric == "loss":
            shortcut = bool(metric_value < self.config.threshold)
            notes_rule = "shortcut_detected is True when loss < threshold."
        else:
            shortcut = bool(metric_value > self.config.threshold)
            notes_rule = f"shortcut_detected is True when {metric} > threshold."

        self.shortcut_detected_ = shortcut

        # risk mapping
        if metric == "loss":
            if metric_value <= min(self.config.threshold, 0.35):
                risk_level = "high"
            elif metric_value < self.config.threshold:
                risk_level = "moderate"
            else:
                risk_level = "low"
        else:
            if metric_value >= max(self.config.threshold, 0.85):
                risk_level = "high"
            elif metric_value >= self.config.threshold:
                risk_level = "moderate"
            else:
                risk_level = "low"

        classes, counts = np.unique(y, return_counts=True)

        self._set_results(
            shortcut_detected=shortcut,
            risk_level=risk_level,
            metrics={
                "metric": metric,
                "metric_value": metric_value,
                "threshold": self.config.threshold,
                "protocol": "holdout",
            },
            notes=(
                "Trained a PyTorch probe model to predict the provided target from embeddings. "
                + notes_rule
            ),
            metadata={
                "device": self.device,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "optimizer": getattr(self.optimizer_class, "__name__", str(self.optimizer_class)),
                "optimizer_kwargs": dict(self.optimizer_kwargs),
                "class_distribution": {
                    str(c): int(nc) for c, nc in zip(classes, counts, strict=False)
                },
                "n_train": int(X_tr.shape[0]),
                "n_test": int(X_te.shape[0]),
            },
            report={
                "protocol": "holdout",
                "test_size": self.config.test_size,
                "train_loss_history": train_loss_hist,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
            },
        )

        self._is_fitted = True
        return self

    def fit_dataset(
        self,
        dataset: Dataset,
        *,
        val_dataset: Dataset | None = None,
        target_extractor: Callable[[Any], Any] | None = None,
        data_spec: DataSpec | dict[str, Any] | None = None,
    ) -> TorchProbe:
        """Train using map-style or iterable datasets without materializing full arrays."""
        resolved_spec = resolve_data_spec(data_spec)
        is_iterable = is_iterable_dataset(dataset)

        if is_iterable and val_dataset is None:
            raise ValueError("IterableDataset requires `val_dataset` for TorchProbe.fit_dataset.")

        if val_dataset is None:
            n = safe_len(dataset)
            if n is None or n < 4:
                raise ValueError("Dataset must expose len() and have at least 4 samples.")
            rng = np.random.RandomState(self.config.random_state)
            idx = np.arange(n)
            rng.shuffle(idx)
            test_n = max(1, int(round(self.config.test_size * n)))
            val_idx = idx[:test_n]
            train_idx = idx[test_n:]
            train_dataset = Subset(dataset, train_idx.tolist())
            eval_dataset = Subset(dataset, val_idx.tolist())
        else:
            train_dataset = dataset
            eval_dataset = val_dataset

        train_loader = self._make_loader_from_dataset(
            train_dataset,
            stage="train",
            shuffle=not is_iterable_dataset(train_dataset),
        )
        val_loader = self._make_loader_from_dataset(
            eval_dataset,
            stage="val",
            shuffle=False,
        )
        return self.fit_loaders(
            train_loader,
            val_loader=val_loader,
            target_extractor=target_extractor,
            data_spec=resolved_spec,
        )

    def fit_loaders(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader,
        target_extractor: Callable[[Any], Any] | None = None,
        data_spec: DataSpec | dict[str, Any] | None = None,
    ) -> TorchProbe:
        """Train/evaluate from user-provided loaders."""
        resolved_spec = resolve_data_spec(data_spec)
        device = torch.device(self.device)

        train_loss_hist, best_val_loss, best_epoch = self._train_with_loaders(
            train_loader,
            val_loader,
            target_extractor=target_extractor,
        )
        eval_metrics = self._evaluate_loader(
            val_loader,
            device,
            target_extractor=target_extractor,
        )

        metric = self.config.metric
        if metric not in eval_metrics:
            raise ValueError(
                f"Metric '{metric}' not available. Available: {sorted(eval_metrics.keys())}"
            )
        metric_value = float(eval_metrics[metric])
        self.metric_value_ = metric_value

        if metric == "loss":
            shortcut = bool(metric_value < self.config.threshold)
            notes_rule = "shortcut_detected is True when loss < threshold."
            if metric_value <= min(self.config.threshold, 0.35):
                risk_level = "high"
            elif metric_value < self.config.threshold:
                risk_level = "moderate"
            else:
                risk_level = "low"
        else:
            shortcut = bool(metric_value > self.config.threshold)
            notes_rule = f"shortcut_detected is True when {metric} > threshold."
            if metric_value >= max(self.config.threshold, 0.85):
                risk_level = "high"
            elif metric_value >= self.config.threshold:
                risk_level = "moderate"
            else:
                risk_level = "low"

        self.shortcut_detected_ = shortcut
        n_train = (
            resolved_spec.train_size
            if resolved_spec is not None
            else safe_len(getattr(train_loader, "dataset", None))
        )
        n_test = (
            resolved_spec.val_size
            if resolved_spec is not None
            else safe_len(getattr(val_loader, "dataset", None))
        )

        self._set_results(
            shortcut_detected=shortcut,
            risk_level=risk_level,
            metrics={
                "metric": metric,
                "metric_value": metric_value,
                "threshold": self.config.threshold,
                "protocol": "loader",
            },
            notes=("Trained a PyTorch probe model from provided data loaders. " + notes_rule),
            metadata={
                "device": self.device,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "optimizer": getattr(self.optimizer_class, "__name__", str(self.optimizer_class)),
                "optimizer_kwargs": dict(self.optimizer_kwargs),
                "n_train": n_train,
                "n_test": n_test,
            },
            report={
                "protocol": "loader",
                "test_size": self.config.test_size,
                "train_loss_history": train_loss_hist,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
            },
        )
        self._is_fitted = True
        return self

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        stage: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool = False,
    ) -> DataLoader:
        X_t = torch.from_numpy(np.asarray(X)).float()
        y_t = torch.from_numpy(np.asarray(y))
        ds = TensorDataset(X_t, y_t)
        req = LoaderRequest(
            stage=stage,
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.device.startswith("cuda"),
            drop_last=drop_last,
        )
        return build_loader(
            req,
            loader_factory=self.loader_factory,
            stage_loader_overrides=self.stage_loader_overrides,
        )

    def _make_loader_from_dataset(
        self,
        dataset: Dataset,
        *,
        stage: str,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        req = LoaderRequest(
            stage=stage,
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=bool(shuffle and not is_iterable_dataset(dataset)),
            num_workers=self.config.num_workers,
            pin_memory=self.device.startswith("cuda"),
            drop_last=drop_last,
        )
        return build_loader(
            req,
            loader_factory=self.loader_factory,
            stage_loader_overrides=self.stage_loader_overrides,
        )

    def _train_holdout(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        y_te: np.ndarray,
    ) -> tuple[list[float], float, int]:
        train_loader = self._make_loader(
            X_tr,
            y_tr,
            stage="train",
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = self._make_loader(
            X_te,
            y_te,
            stage="val",
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        return self._train_with_loaders(train_loader, val_loader)

    def _train_with_loaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        target_extractor: Callable[[Any], Any] | None = None,
    ) -> tuple[list[float], float, int]:
        device = torch.device(self.device)
        self.model.to(device)

        self._optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        if self.config.use_amp and self.device.startswith("cuda"):
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

        best_val = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        loss_hist: list[float] = []

        for epoch in range(self.config.epochs):
            self.model.train()
            total = 0.0
            n_seen = 0

            for batch in train_loader:
                xb, yb = extract_xy_batch(batch, target_extractor=target_extractor)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                self._optimizer.zero_grad()

                if self._scaler is not None:
                    with torch.cuda.amp.autocast():
                        out = self.model(xb)
                        loss = self.loss_fn(out, yb)
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
                    loss.backward()
                    self._optimizer.step()

                bs = int(xb.shape[0])
                total += float(loss.item()) * bs
                n_seen += bs

            train_loss = total / max(1, n_seen)
            loss_hist.append(float(train_loss))

            val_metrics = self._evaluate_loader(
                val_loader,
                device,
                target_extractor=target_extractor,
            )
            val_loss = float(val_metrics.get("loss", float("nan")))

            if self.config.verbose:
                print(
                    f"[Epoch {epoch+1}/{self.config.epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

            if self.config.early_stopping is not None and best_epoch >= 0:
                if (epoch - best_epoch) >= self.config.early_stopping:
                    break

        if best_state is not None:
            state = {k: v.to(device) for k, v in best_state.items()}
            self.model.load_state_dict(state)

        return loss_hist, float(best_val), int(best_epoch)

    def _evaluate_loader(
        self,
        loader: DataLoader,
        device: torch.device,
        *,
        target_extractor: Callable[[Any], Any] | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_seen = 0
        logits_list = []
        y_list = []
        with torch.no_grad():
            for batch in loader:
                xb, yb = extract_xy_batch(batch, target_extractor=target_extractor)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = self.model(xb)
                loss = self.loss_fn(out, yb)
                bs = int(xb.shape[0])
                total_loss += float(loss.item()) * bs
                n_seen += bs
                logits_list.append(out.detach().cpu().numpy())
                y_list.append(yb.detach().cpu().numpy())

        logits = np.concatenate(logits_list, axis=0)
        y_true = np.concatenate(y_list, axis=0)
        metrics: dict[str, float] = {"loss": float(total_loss / max(1, n_seen))}

        # classification: logits shape (N, C)
        if logits.ndim == 2 and logits.shape[1] > 1:
            probs = _softmax_np(logits)
            y_pred = probs.argmax(axis=1)
            metrics["accuracy"] = _accuracy(y_true, y_pred)
            metrics["f1"] = (
                _f1_binary(y_true, y_pred) if _is_binary(y_true) else _f1_macro(y_true, y_pred)
            )
            if _is_binary(y_true):
                metrics["roc_auc"] = _roc_auc_binary(y_true, probs[:, 1])
        else:
            # regression-ish; users can use loss as metric
            pass

        return metrics

    def _evaluate_np(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        device = torch.device(self.device)
        loader = self._make_loader(
            X,
            y,
            stage="eval",
            batch_size=max(64, min(256, int(X.shape[0]))),
            shuffle=False,
            num_workers=0,
        )
        return self._evaluate_loader(loader, device)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for embeddings (requires prior fit)."""
        self._ensure_fitted()
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features). Got shape={X_arr.shape}.")
        device = torch.device(self.device)
        self.model.to(device)
        self.model.eval()
        loader = self._make_loader(
            X_arr,
            np.zeros(X_arr.shape[0], dtype=np.int64),
            stage="predict",
            batch_size=max(64, min(256, int(X_arr.shape[0]))),
            shuffle=False,
            num_workers=0,
        )
        preds_list = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device, non_blocking=True)
                out = self.model(xb)
                if out.dim() == 2 and out.shape[1] > 1:
                    preds_list.append(out.argmax(dim=1).detach().cpu().numpy())
                else:
                    preds_list.append(out.squeeze(-1).detach().cpu().numpy().astype(np.int64))
        return np.concatenate(preds_list, axis=0)
