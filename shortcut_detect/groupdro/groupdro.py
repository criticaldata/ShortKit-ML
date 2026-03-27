# shortcut_detect/groupdro.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from ..detector_base import DetectorBase
from ..training.data_adapters import (
    DataSpec,
    extract_xyg_batch,
    is_iterable_dataset,
    resolve_data_spec,
    safe_len,
)
from ..training.loader_hooks import LoaderFactory, LoaderRequest, StageLoaderOverrides, build_loader

# -----------------------
# Embedding dataset wrapper
# -----------------------


class EmbeddingGroupDataset(Dataset):
    """
    Provides (x, y, g) triples and metadata required by GroupDRO LossComputer:
      - n_groups
      - group_counts()
      - group_str(idx)

    If group_ids is provided, it defines the fixed universe of groups and the mapping
    is consistent across datasets (e.g., train and external val).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        *,
        group_ids: np.ndarray | None = None,
    ):
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"labels must be 1D, got {y.shape}")
        if g.ndim != 1:
            raise ValueError(f"group_labels must be 1D, got {g.shape}")
        if not (len(X) == len(y) == len(g)):
            raise ValueError("embeddings, labels, group_labels must have same length")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        # remap arbitrary group ids -> contiguous 0..G-1 (matches original code assumptions)
        uniq = np.unique(g)
        self.group_ids = uniq
        self.group_to_index = {v: i for i, v in enumerate(self.group_ids.tolist())}
        g_idx = np.array([self.group_to_index[v] for v in g], dtype=np.int64)
        self.g = torch.from_numpy(g_idx).long()

        # Fixed universe of group ids (recommended when using external val split)
        if group_ids is None:
            uniq = np.unique(g)
            self.group_ids = uniq.astype(np.int64)
        else:
            self.group_ids = np.asarray(group_ids, dtype=np.int64)
            if self.group_ids.ndim != 1:
                raise ValueError("group_ids must be 1D array-like.")
            if len(np.unique(self.group_ids)) != len(self.group_ids):
                raise ValueError("group_ids contains duplicates.")

        self.group_to_index = {int(v): i for i, v in enumerate(self.group_ids.tolist())}

        # map; error if unknown group appears
        try:
            g_idx = np.array([self.group_to_index[int(v)] for v in g], dtype=np.int64)
        except KeyError as e:
            raise ValueError(
                f"Found group id {e.args[0]} not present in provided group_ids."
            ) from None

        self.g = torch.from_numpy(g_idx).long()
        self.n_groups = len(self.group_ids)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.g[idx]

    def group_counts(self) -> torch.Tensor:
        return torch.bincount(self.g, minlength=self.n_groups).float()

    def group_str(self, group_idx: int) -> str:
        # original group id for readability
        gid = self.group_ids[group_idx]
        return f"group={gid}"


class GroupMetadata:
    """Minimal group metadata adapter used by LossComputer in loader-native paths."""

    def __init__(self, group_ids: np.ndarray, group_counts: np.ndarray):
        gids = np.asarray(group_ids, dtype=np.int64)
        counts = np.asarray(group_counts, dtype=np.float32)
        if gids.ndim != 1:
            raise ValueError("group_ids must be 1D.")
        if counts.ndim != 1 or counts.shape[0] != gids.shape[0]:
            raise ValueError("group_counts must be 1D with same length as group_ids.")
        self.group_ids = gids
        self.n_groups = int(gids.shape[0])
        self._counts = torch.from_numpy(counts).float()

    def group_counts(self) -> torch.Tensor:
        return self._counts

    def group_str(self, group_idx: int) -> str:
        return f"group={int(self.group_ids[group_idx])}"


# -----------------------
# LossComputer (ported)
# -----------------------


class LossComputer:
    """
    Faithful port of the original GroupDRO LossComputer, but device-agnostic.
    """

    def __init__(
        self,
        criterion,
        is_robust: bool,
        dataset: Any,
        alpha: float | None = None,
        gamma: float = 0.1,
        adj: np.ndarray | None = None,
        min_var_weight: float = 0.0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
        device: torch.device | None = None,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(device)
        self.group_frac = self.group_counts / self.group_counts.sum().clamp_min(1.0)
        self.group_str = dataset.group_str
        self.device = device

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(device)

        if is_robust:
            if alpha is None:
                raise ValueError("GroupDRO robust mode requires alpha (as in original repo).")

        # maintained throughout training
        self.adv_probs = torch.ones(self.n_groups, device=device) / max(self.n_groups, 1)
        self.exp_avg_loss = torch.zeros(self.n_groups, device=device)
        self.exp_avg_initialized = torch.zeros(self.n_groups, dtype=torch.uint8, device=device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training: bool = False):
        # per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, _ = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)
        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss = adjusted_loss + self.adj / torch.sqrt(self.group_counts.clamp_min(1.0))

        if self.normalize_loss:
            denom = adjusted_loss.sum().clamp_min(1e-12)
            adjusted_loss = adjusted_loss / denom

        # adv_probs update
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.detach())
        self.adv_probs = self.adv_probs / self.adv_probs.sum().clamp_min(1e-12)

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.group_counts.clamp_min(1.0))
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        # matches original greedy selection by alpha mass
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= float(self.alpha)
        weights = mask.float() * sorted_frac / float(self.alpha)
        last_idx = int(mask.sum().item())
        if last_idx < len(weights):
            weights[last_idx] = 1.0 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # unsort weights
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # original uses group_map = (group_idx == arange(n_groups)[:,None]).float()
        device = losses.device
        group_map = (
            group_idx == torch.arange(self.n_groups, device=device).unsqueeze(1).long()
        ).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups, device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups, device=self.device)
        self.avg_group_acc = torch.zeros(self.n_groups, device=self.device)
        self.avg_per_sample_loss = torch.tensor(0.0, device=self.device)
        self.avg_actual_loss = torch.tensor(0.0, device=self.device)
        self.avg_acc = torch.tensor(0.0, device=self.device)
        self.batch_count = 0

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        denom = self.processed_data_counts + group_count
        denom = denom + (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom

        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        denom_b = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom_b) * self.avg_actual_loss + (
            1 / denom_b
        ) * actual_loss

        self.processed_data_counts += group_count
        if self.is_robust and weights is not None:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()

        self.batch_count += 1

        group_frac = self.processed_data_counts / self.processed_data_counts.sum().clamp_min(1.0)
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_stats(self) -> dict[str, float]:
        stats = {}
        for idx in range(self.n_groups):
            stats[f"avg_loss_group:{idx}"] = float(self.avg_group_loss[idx].item())
            stats[f"exp_avg_loss_group:{idx}"] = float(self.exp_avg_loss[idx].item())
            stats[f"avg_acc_group:{idx}"] = float(self.avg_group_acc[idx].item())
            stats[f"processed_data_count_group:{idx}"] = float(
                self.processed_data_counts[idx].item()
            )
            stats[f"update_data_count_group:{idx}"] = float(self.update_data_counts[idx].item())
            stats[f"update_batch_count_group:{idx}"] = float(self.update_batch_counts[idx].item())

        stats["avg_actual_loss"] = float(self.avg_actual_loss.item())
        stats["avg_per_sample_loss"] = float(self.avg_per_sample_loss.item())
        stats["avg_acc"] = float(self.avg_acc.item())
        # useful extras
        stats["worst_group_acc"] = (
            float(self.avg_group_acc.min().item()) if self.n_groups > 0 else float("nan")
        )
        stats["adv_probs_entropy"] = float(
            -(self.adv_probs * (self.adv_probs + 1e-12).log()).sum().item()
        )
        return stats


# -----------------------
# Detector (your package-facing wrapper)
# -----------------------


@dataclass
class GroupDROConfig:
    # training
    n_epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-5
    momentum: float = 0.9
    num_workers: int = 0
    loader_factory: LoaderFactory | None = None
    stage_loader_overrides: StageLoaderOverrides = None

    # robust objective
    robust: bool = True
    alpha: float = 0.2
    gamma: float = 0.1
    robust_step_size: float = 0.01
    use_normalized_loss: bool = False
    btl: bool = False
    minimum_variational_weight: float = 0.0
    generalization_adjustment: list[float] | None = None  # len=1 or len=n_groups
    automatic_adjustment: bool = False  # optional parity with original (updates adj)

    # model on embeddings
    hidden_dim: int | None = None
    dropout: float = 0.0

    # splitting/eval
    val_fraction: float = 0.1

    # misc
    seed: int = 0
    device: str | None = None


class GroupDRODetector(DetectorBase):
    """
    Embedding-based GroupDRO detector.
    Trains a small classifier on embeddings under GroupDRO, then reports
    worst-group vs average accuracy and group-wise metrics.
    """

    def __init__(self, config: GroupDROConfig | None = None):
        super().__init__(method="groupdro")
        self.config = config or GroupDROConfig()
        self.model_: nn.Module | None = None
        self.report_: dict[str, Any] = {}

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, d: int, n_classes: int) -> nn.Module:
        if self.config.hidden_dim is None:
            return nn.Linear(d, n_classes)
        return nn.Sequential(
            nn.Linear(d, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, n_classes),
        )

    def _split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        # deterministic split via numpy RNG already seeded
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_val = int(round(self.config.val_fraction * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_comp: LossComputer,
        batch_parser: (
            Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None,
    ) -> dict[str, float]:
        model.eval()
        loss_comp.reset_stats()
        for batch in loader:
            if batch_parser is None:
                x, y, g = batch
            else:
                x, y, g = batch_parser(batch)
            x, y, g = x.to(loss_comp.device), y.to(loss_comp.device), g.to(loss_comp.device)
            logits = model(x)
            _ = loss_comp.loss(logits, y, g, is_training=False)
        return loss_comp.get_stats()

    def fit(
        self, embeddings: np.ndarray, labels: np.ndarray, group_labels: np.ndarray
    ) -> GroupDRODetector:
        """
        Default behavior: internal train/val split.
        """
        self._set_seed()
        device = self._device()

        ds = EmbeddingGroupDataset(embeddings, labels, group_labels)
        n = len(ds)
        train_idx, val_idx = self._split_indices(n)

        train_ds = Subset(ds, train_idx.tolist())
        val_ds = Subset(ds, val_idx.tolist())

        result = self._fit_from_datasets(
            ds_full=ds, train_ds=train_ds, val_ds=val_ds, device=device
        )
        return result

    def get_shortcut_detected(self, report):
        q = np.asarray(report["final_adv_probs"])

        # Invert group_id_map to locate indices
        # group_id_map: original_gid -> idx
        gid_map = report["group_id_map"]
        idx_easy = gid_map[0]
        idx_hard = gid_map[1]

        # Expect higher q on harder group (not guaranteed in every random draw, but should hold here)
        return q[idx_hard] > q[idx_easy]

    def fit_with_val(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        train_group_labels: np.ndarray,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray,
        val_group_labels: np.ndarray,
        *,
        group_ids: np.ndarray | None = None,
    ) -> GroupDRODetector:
        """
        External validation split:
        - trains on provided train set
        - selects best checkpoint on provided val set
        - supports fixed group-id mapping via group_ids
        """
        self._set_seed()
        device = self._device()

        train_ds_full = EmbeddingGroupDataset(
            train_embeddings, train_labels, train_group_labels, group_ids=group_ids
        )
        val_ds_full = EmbeddingGroupDataset(
            val_embeddings, val_labels, val_group_labels, group_ids=train_ds_full.group_ids
        )

        # For the LossComputer stats, we want dataset metadata:
        # - training LossComputer should use training dataset counts
        # - validation LossComputer should use validation dataset counts
        return self._fit_from_train_val(train_ds_full, val_ds_full, device=device)

    def fit_dataset(
        self,
        dataset: Dataset,
        *,
        val_dataset: Dataset | None = None,
        target_extractor: Callable[[Any], Any] | None = None,
        group_extractor: Callable[[Any], Any] | None = None,
        data_spec: DataSpec | dict[str, Any] | None = None,
    ) -> GroupDRODetector:
        """Train/evaluate GroupDRO from datasets without materializing full arrays."""
        self._set_seed()
        resolved_spec = resolve_data_spec(data_spec)
        is_iter = is_iterable_dataset(dataset)

        if is_iter and val_dataset is None:
            raise ValueError("IterableDataset requires `val_dataset` for GroupDRO.fit_dataset.")

        if val_dataset is None:
            n = safe_len(dataset)
            if n is None or n < 2:
                raise ValueError("Dataset must expose len() and contain at least 2 samples.")
            train_idx, val_idx = self._split_indices(n)
            train_dataset = Subset(dataset, train_idx.tolist())
            eval_dataset = Subset(dataset, val_idx.tolist())
        else:
            train_dataset = dataset
            eval_dataset = val_dataset

        train_loader = self._build_dataset_loader(
            train_dataset, stage="train", shuffle=not is_iterable_dataset(train_dataset)
        )
        val_loader = self._build_dataset_loader(eval_dataset, stage="val", shuffle=False)
        return self.fit_loaders(
            train_loader,
            val_loader=val_loader,
            target_extractor=target_extractor,
            group_extractor=group_extractor,
            data_spec=resolved_spec,
            split_mode="external",
        )

    def fit_loaders(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader,
        target_extractor: Callable[[Any], Any] | None = None,
        group_extractor: Callable[[Any], Any] | None = None,
        data_spec: DataSpec | dict[str, Any] | None = None,
        split_mode: str = "external",
    ) -> GroupDRODetector:
        """Train/evaluate GroupDRO from user-provided loaders."""
        self._set_seed()
        device = self._device()
        cfg = self.config
        resolved_spec = resolve_data_spec(data_spec)

        train_iterable = is_iterable_dataset(getattr(train_loader, "dataset", None))
        val_iterable = is_iterable_dataset(getattr(val_loader, "dataset", None))
        if (train_iterable or val_iterable) and resolved_spec is None:
            raise ValueError(
                "IterableDataset loaders require `data_spec` with n_features, n_classes, and n_groups."
            )

        d, n_classes, group_ids, train_counts, val_counts = self._resolve_loader_schema(
            train_loader=train_loader,
            val_loader=val_loader,
            target_extractor=target_extractor,
            group_extractor=group_extractor,
            data_spec=resolved_spec,
        )
        group_to_index = {int(v): i for i, v in enumerate(group_ids.tolist())}

        def parse_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x, y, g = extract_xyg_batch(
                batch,
                target_extractor=target_extractor,
                group_extractor=group_extractor,
            )
            mapped = torch.full_like(g, fill_value=-1)
            for gid, idx in group_to_index.items():
                mapped[g == gid] = int(idx)
            if torch.any(mapped < 0):
                bad = torch.unique(g[mapped < 0]).detach().cpu().numpy().tolist()
                raise ValueError(
                    f"Found unknown group ids {bad}; expected subset of {group_ids.tolist()}."
                )
            return x, y, mapped

        model = self._build_model(d, n_classes).to(device)
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")
        adj = self._prepare_adjustment(cfg, n_groups=len(group_ids))

        train_meta = GroupMetadata(group_ids=group_ids, group_counts=train_counts)
        val_meta = GroupMetadata(group_ids=group_ids, group_counts=val_counts)

        train_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=train_meta,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )

        history, best_state = self._train_loop(
            model=model,
            opt=opt,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            train_loss_comp=train_loss_comp,
            ds_for_val_loss_comp=val_meta,
            adj=adj,
            device=device,
            batch_parser=parse_batch,
        )

        if best_state is not None:
            model.load_state_dict(best_state)

        val_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=val_meta,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )
        final_val_stats = self._eval_epoch(
            model, val_loader, val_loss_comp, batch_parser=parse_batch
        )

        self.model_ = model
        self.report_ = {
            "success": True,
            "method": "groupdro",
            "n_groups": len(group_ids),
            "group_id_map": {int(v): i for i, v in enumerate(group_ids.tolist())},
            "history": history,
            "final": final_val_stats,
            "final_adv_probs": val_loss_comp.adv_probs.detach().cpu().numpy(),
            "split_mode": split_mode,
        }

        self._finalize_results()
        self._is_fitted = True
        return self

    def _build_dataset_loader(self, dataset: Dataset, *, stage: str, shuffle: bool) -> DataLoader:
        cfg = self.config
        device = self._device()
        return build_loader(
            LoaderRequest(
                stage=stage,
                dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=bool(shuffle and not is_iterable_dataset(dataset)),
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )

    def _scan_loader(
        self,
        loader: DataLoader,
        *,
        target_extractor: Callable[[Any], Any] | None,
        group_extractor: Callable[[Any], Any] | None,
    ) -> dict[str, Any]:
        n_samples = 0
        n_features = None
        max_label = -1
        group_counts: dict[int, int] = {}
        for batch in loader:
            x, y, g = extract_xyg_batch(
                batch,
                target_extractor=target_extractor,
                group_extractor=group_extractor,
            )
            if x.ndim < 2:
                raise ValueError("Expected batched feature tensor with ndim >= 2.")
            if n_features is None:
                n_features = int(x.shape[1])
            n_samples += int(y.shape[0])
            max_label = max(max_label, int(y.max().item()))
            groups_np = g.detach().cpu().numpy().astype(np.int64)
            ids, counts = np.unique(groups_np, return_counts=True)
            for gid, cnt in zip(ids.tolist(), counts.tolist(), strict=False):
                group_counts[int(gid)] = group_counts.get(int(gid), 0) + int(cnt)
        if n_features is None or n_samples == 0:
            raise ValueError("Loader produced no samples.")
        return {
            "n_features": int(n_features),
            "n_classes": int(max_label + 1),
            "group_counts": group_counts,
            "n_samples": int(n_samples),
        }

    def _resolve_loader_schema(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_extractor: Callable[[Any], Any] | None,
        group_extractor: Callable[[Any], Any] | None,
        data_spec: DataSpec | None,
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        if data_spec is not None:
            if data_spec.n_groups is None and data_spec.group_ids is None:
                raise ValueError("data_spec must provide n_groups or group_ids for GroupDRO.")
            if data_spec.group_ids is not None:
                group_ids = np.asarray(data_spec.group_ids, dtype=np.int64)
            else:
                group_ids = np.arange(int(data_spec.n_groups), dtype=np.int64)
            n_groups = len(group_ids)
            train_size = (
                data_spec.train_size or safe_len(getattr(train_loader, "dataset", None)) or n_groups
            )
            val_size = (
                data_spec.val_size or safe_len(getattr(val_loader, "dataset", None)) or n_groups
            )
            train_counts = np.full(
                n_groups, fill_value=max(train_size / n_groups, 1.0), dtype=np.float32
            )
            val_counts = np.full(
                n_groups, fill_value=max(val_size / n_groups, 1.0), dtype=np.float32
            )
            return (
                int(data_spec.n_features),
                int(data_spec.n_classes),
                group_ids,
                train_counts,
                val_counts,
            )

        train_stats = self._scan_loader(
            train_loader,
            target_extractor=target_extractor,
            group_extractor=group_extractor,
        )
        val_stats = self._scan_loader(
            val_loader,
            target_extractor=target_extractor,
            group_extractor=group_extractor,
        )
        all_groups = sorted(
            set(train_stats["group_counts"].keys()) | set(val_stats["group_counts"].keys())
        )
        group_ids = np.asarray(all_groups, dtype=np.int64)
        n_groups = len(group_ids)
        train_counts = np.zeros(n_groups, dtype=np.float32)
        val_counts = np.zeros(n_groups, dtype=np.float32)
        for i, gid in enumerate(group_ids.tolist()):
            train_counts[i] = float(train_stats["group_counts"].get(gid, 0))
            val_counts[i] = float(val_stats["group_counts"].get(gid, 0))
        train_counts = np.maximum(train_counts, 1.0)
        val_counts = np.maximum(val_counts, 1.0)
        d = int(train_stats["n_features"])
        n_classes = int(max(train_stats["n_classes"], val_stats["n_classes"]))
        return d, n_classes, group_ids, train_counts, val_counts

    def _fit_from_datasets(
        self,
        *,
        ds_full: EmbeddingGroupDataset,
        train_ds: Dataset,
        val_ds: Dataset,
        device: torch.device,
    ) -> GroupDRODetector:
        """
        Internal split path (existing behavior). ds_full provides group universe.
        train_ds/val_ds are subsets of ds_full.
        """
        cfg = self.config

        train_loader = build_loader(
            LoaderRequest(
                stage="train",
                dataset=train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )
        val_loader = build_loader(
            LoaderRequest(
                stage="val",
                dataset=val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )

        d = ds_full.X.shape[1]
        n_classes = int(torch.max(ds_full.y).item()) + 1
        model = self._build_model(d, n_classes).to(device)

        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

        criterion = nn.CrossEntropyLoss(reduction="none")

        adj = self._prepare_adjustment(cfg, n_groups=ds_full.n_groups)

        train_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=ds_full,  # OK: same group universe, counts from full set
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )

        history, best_state = self._train_loop(
            model=model,
            opt=opt,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            train_loss_comp=train_loss_comp,
            ds_for_val_loss_comp=ds_full,  # same as before
            adj=adj,
            device=device,
        )

        if best_state is not None:
            model.load_state_dict(best_state)

        # final eval on full ds_full like existing fit()
        full_loader = build_loader(
            LoaderRequest(
                stage="full",
                dataset=ds_full,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )
        full_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=ds_full,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )
        full_stats = self._eval_epoch(model, full_loader, full_loss_comp)

        self.model_ = model
        self.report_ = {
            "success": True,
            "method": "groupdro",
            "n_groups": ds_full.n_groups,
            "group_id_map": ds_full.group_to_index,  # original-id -> 0..G-1
            "history": history,
            "final": full_stats,
            "final_adv_probs": full_loss_comp.adv_probs.detach().cpu().numpy(),
            "split_mode": "internal",
        }
        self._finalize_results()
        self._is_fitted = True

        return self

    def _fit_from_train_val(
        self,
        train_ds: EmbeddingGroupDataset,
        val_ds: EmbeddingGroupDataset,
        *,
        device: torch.device,
    ) -> GroupDRODetector:
        """
        External split path. Uses separate datasets but fixed group_ids mapping.
        """
        cfg = self.config

        train_loader = build_loader(
            LoaderRequest(
                stage="train",
                dataset=train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )
        val_loader = build_loader(
            LoaderRequest(
                stage="val",
                dataset=val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )

        d = train_ds.X.shape[1]
        n_classes = int(max(train_ds.y.max().item(), val_ds.y.max().item())) + 1
        model = self._build_model(d, n_classes).to(device)

        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

        criterion = nn.CrossEntropyLoss(reduction="none")
        adj = self._prepare_adjustment(cfg, n_groups=train_ds.n_groups)

        train_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=train_ds,  # counts from train
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )

        history, best_state = self._train_loop(
            model=model,
            opt=opt,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            train_loss_comp=train_loss_comp,
            ds_for_val_loss_comp=val_ds,  # counts from val
            adj=adj,
            device=device,
        )

        if best_state is not None:
            model.load_state_dict(best_state)

        # final eval: report val as "final" by default (SSA-friendly); optionally also train
        val_loss_comp = LossComputer(
            criterion=criterion,
            is_robust=cfg.robust,
            dataset=val_ds,
            alpha=cfg.alpha,
            gamma=cfg.gamma,
            adj=adj,
            min_var_weight=cfg.minimum_variational_weight,
            step_size=cfg.robust_step_size,
            normalize_loss=cfg.use_normalized_loss,
            btl=cfg.btl,
            device=device,
        )
        final_val_stats = self._eval_epoch(model, val_loader, val_loss_comp)

        self.model_ = model
        self.report_ = {
            "success": True,
            "method": "groupdro",
            "n_groups": train_ds.n_groups,
            "group_id_map": train_ds.group_to_index,
            "history": history,
            "final": final_val_stats,
            "final_adv_probs": val_loss_comp.adv_probs.detach().cpu().numpy(),
            "split_mode": "external",
        }

        self._finalize_results()
        self._is_fitted = True
        return self

    def _prepare_adjustment(self, cfg: GroupDROConfig, n_groups: int) -> np.ndarray | None:
        adj = None
        if cfg.generalization_adjustment is not None:
            arr = np.array(cfg.generalization_adjustment, dtype=np.float32)
            if arr.size == 1:
                adj = np.repeat(arr, n_groups)
            elif arr.size == n_groups:
                adj = arr
            else:
                raise ValueError("generalization_adjustment must have length 1 or n_groups.")
        return adj

    def _train_loop(
        self,
        *,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        criterion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_loss_comp: LossComputer,
        ds_for_val_loss_comp: Any,
        adj: np.ndarray | None,
        device: torch.device,
        batch_parser: (
            Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None,
    ) -> tuple[list[dict[str, float]], dict[str, torch.Tensor] | None]:
        cfg = self.config
        history: list[dict[str, float]] = []
        best_val_worst = -1.0
        best_state = None

        for epoch in range(cfg.n_epochs):
            model.train()
            train_loss_comp.reset_stats()

            for batch in train_loader:
                if batch_parser is None:
                    x, y, g = batch
                else:
                    x, y, g = batch_parser(batch)
                x, y, g = x.to(device), y.to(device), g.to(device)
                logits = model(x)
                loss = train_loss_comp.loss(logits, y, g, is_training=True)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            train_stats = train_loss_comp.get_stats()

            val_loss_comp = LossComputer(
                criterion=criterion,
                is_robust=cfg.robust,
                dataset=ds_for_val_loss_comp,
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                adj=adj,
                min_var_weight=cfg.minimum_variational_weight,
                step_size=cfg.robust_step_size,
                normalize_loss=cfg.use_normalized_loss,
                btl=cfg.btl,
                device=device,
            )
            val_stats = self._eval_epoch(
                model,
                val_loader,
                val_loss_comp,
                batch_parser=batch_parser,
            )

            row = {
                "epoch": float(epoch),
                **{f"train_{k}": float(v) for k, v in train_stats.items()},
                **{f"val_{k}": float(v) for k, v in val_stats.items()},
            }
            history.append(row)

            curr_val_worst = val_stats["worst_group_acc"]
            if curr_val_worst > best_val_worst:
                best_val_worst = curr_val_worst
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if cfg.automatic_adjustment:
                gen_gap = val_loss_comp.avg_group_loss - train_loss_comp.exp_avg_loss
                adjustments = gen_gap * torch.sqrt(train_loss_comp.group_counts.clamp_min(1.0))
                train_loss_comp.adj = adjustments.detach()

        return history, best_state

    def get_report(self) -> dict[str, Any]:
        return super().get_report()

    def _finalize_results(self) -> None:
        final = self.report_.get("final", {})
        metrics = {
            "avg_acc": final.get("avg_acc"),
            "worst_group_acc": final.get("worst_group_acc"),
        }
        metadata = {
            "n_groups": self.report_.get("n_groups"),
            "n_epochs": self.config.n_epochs,
            "batch_size": self.config.batch_size,
            "robust": self.config.robust,
            "split_mode": self.report_.get("split_mode"),
        }
        self.shortcut_detected_ = None
        self._set_results(
            shortcut_detected=None,
            risk_level="unknown",
            metrics=metrics,
            notes="GroupDRO reports worst-group vs average accuracy; interpret gaps manually.",
            metadata=metadata,
            report=self.report_,
        )
