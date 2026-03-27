"""Factory for creating probe detectors from unified API context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sklearn_probe import SKLearnProbe


@dataclass
class ProbeFactoryContext:
    """Context passed to the probe factory (seed + kwargs from unified builder)."""

    seed: int
    kwargs: dict[str, Any]


class ProbeDetectorFactory:
    """Creates probe detectors by backend name."""

    @staticmethod
    def create(backend: str, ctx: ProbeFactoryContext):
        """
        Create a probe detector for the given backend.

        Parameters
        ----------
        backend : str
            "sklearn" or "torch".
        ctx : ProbeFactoryContext
            Seed and kwargs from the unified detector builder.

        Returns
        -------
        SKLearnProbe or TorchProbe
        """
        kwargs = ctx.kwargs
        seed = ctx.seed

        if backend == "sklearn":
            estimator = kwargs.get("probe_estimator")
            return SKLearnProbe(
                estimator=estimator,
                metric=kwargs.get("probe_metric", "f1"),
                threshold=float(kwargs.get("probe_threshold", 0.70)),
                average=kwargs.get("probe_average", "macro"),
                evaluation=kwargs.get("probe_evaluation", "holdout"),
                test_size=float(kwargs.get("probe_test_size", 0.2)),
                cv_folds=int(kwargs.get("probe_cv_folds", 5)),
                random_state=seed,
            )
        if backend == "torch":
            import torch

            from .torch_probe import TorchProbe

            model = kwargs.get("probe_model")
            loss_fn = kwargs.get("probe_loss_fn")
            if model is None or loss_fn is None:
                raise ValueError(
                    "probe_backend='torch' requires 'probe_model' and 'probe_loss_fn' in kwargs."
                )
            return TorchProbe(
                model=model,
                loss_fn=loss_fn,
                optimizer_class=kwargs.get("probe_optimizer_class") or torch.optim.Adam,
                optimizer_kwargs=kwargs.get("probe_optimizer_kwargs") or {"lr": 1e-3},
                device=kwargs.get("probe_device"),
                metric=kwargs.get("probe_metric", "accuracy"),
                threshold=float(kwargs.get("probe_threshold", 0.70)),
                test_size=float(kwargs.get("probe_test_size", 0.2)),
                random_state=seed,
                epochs=int(kwargs.get("probe_epochs", 10)),
                batch_size=int(kwargs.get("probe_batch_size", 128)),
                num_workers=int(kwargs.get("probe_num_workers", 0)),
                early_stopping=kwargs.get("probe_early_stopping"),
                use_amp=bool(kwargs.get("probe_use_amp", False)),
                verbose=bool(kwargs.get("probe_verbose", False)),
                loader_factory=kwargs.get("probe_loader_factory"),
                stage_loader_overrides=kwargs.get("probe_stage_loader_overrides"),
            )
        raise ValueError(f"Unknown probe_backend={backend!r}. Use 'sklearn' or 'torch'.")
