# shortcut_detect/ssa.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from shortcut_detect.detector_base import DetectorBase
from shortcut_detect.groupdro.groupdro import GroupDROConfig, GroupDRODetector
from shortcut_detect.training.loader_hooks import (
    LoaderFactory,
    LoaderRequest,
    StageLoaderOverrides,
    build_loader,
)

# -----------------------
# Datasets
# -----------------------


class EmbeddingUnlabeledDataset(Dataset):
    """Returns (x, y, idx) for DU."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"labels must be 1D, got {y.shape}")
        if len(X) != len(y):
            raise ValueError("embeddings and labels must have same length")
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], idx


class EmbeddingLabeledAttrDataset(Dataset):
    """Returns (x, y, a) for DL."""

    def __init__(self, X: np.ndarray, y: np.ndarray, a: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"labels must be 1D, got {y.shape}")
        if a.ndim != 1:
            raise ValueError(f"spurious attribute labels must be 1D, got {a.shape}")
        if not (len(X) == len(y) == len(a)):
            raise ValueError("embeddings, labels, spurious_labels must have same length")
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.a = torch.from_numpy(a).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.a[idx]


# -----------------------
# Model: spurious attribute predictor fk(x; θk)
# -----------------------


class AttrPredictor(nn.Module):
    def __init__(self, d: int, n_a: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(d, n_a)
        else:
            self.net = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_a),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------
# SSA config
# -----------------------


@dataclass
class SSAConfig:
    # Algorithm 1: K splits, T iterations
    K: int = 3
    T: int = 2000  # iterations (not epochs)

    # pseudo-labeler training
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # SGD as in many paper settings

    # pseudo-labeler model
    hidden_dim: int | None = None
    dropout: float = 0.0

    # Eq. (5–8): adaptive thresholds
    tau_gmin: float = 0.95
    threshold_update_every: int = 200
    threshold_update_max_items: int | None = 20000  # subsample DU^o for threshold update (speed)

    # DL split for pseudo-labeler tuning (paper: split DL into two halves; fixed)
    dl_val_fraction: float = 0.5

    # misc
    seed: int = 0
    device: str | None = None
    loader_factory: LoaderFactory | None = None
    stage_loader_overrides: StageLoaderOverrides = None

    # Phase 2: GroupDRO config (robust training)
    groupdro: GroupDROConfig = field(default_factory=GroupDROConfig)

    # hard threshold to determine if a shortcut is detected or not
    ssa_gap_threshold = 0.10


# -----------------------
# SSA detector
# -----------------------


class SSADetector(DetectorBase):
    """
    Embeddings-first SSA:
      Phase 1: pseudo-label spurious attribute a_hat(x) for DU using DL + DU with adaptive thresholds.
      Phase 2: run GroupDRO on (X_U, y_U, g_hat=(y,a_hat)) with validation on DL (true groups).
    """

    def __init__(self, config: SSAConfig | None = None):
        super().__init__(method="ssa")
        self.config = config or SSAConfig()
        self.report_: dict[str, Any] = {}
        self.groupdro_: GroupDRODetector | None = None
        self.attr_models_: list[nn.Module] = []  # one per fold (optional to keep)
        self.a_hat_: np.ndarray | None = None  # pseudo a for DU

        self.seed = self.config.seed
        self._is_fitted = False
        self.shortcut_detected_ = None
        self.results_ = {}

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _encode_group(y: np.ndarray, a: np.ndarray, n_a: int) -> np.ndarray:
        return (y.astype(np.int64) * int(n_a) + a.astype(np.int64)).astype(np.int64)

    @staticmethod
    def _kfold_indices(n: int, K: int, rng: np.random.RandomState) -> list[np.ndarray]:
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, K)
        return [f.astype(np.int64) for f in folds]

    @staticmethod
    def _split_indices(
        n: int, val_fraction: float, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(round(val_fraction * n))
        val_idx = idx[:n_val].astype(np.int64)
        train_idx = idx[n_val:].astype(np.int64)
        return train_idx, val_idx

    @torch.no_grad()
    def _eval_attr_worst_group_acc(
        self,
        model: nn.Module,
        loader: DataLoader,
        n_y: int,
        n_a: int,
        device: torch.device,
    ) -> float:
        """
        Worst-group accuracy for spurious attribute prediction on DL^bullet.
        Groups are defined by (y,a) (true a).
        """
        model.eval()
        correct = {}
        total = {}
        for x, y, a in loader:
            x = x.to(device)
            y_np = y.numpy()
            a_np = a.numpy()
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            g = self._encode_group(y_np, a_np, n_a)
            for gi, pi, ai in zip(g, pred, a_np, strict=False):
                total[gi] = total.get(gi, 0) + 1
                correct[gi] = correct.get(gi, 0) + int(pi == ai)

        if not total:
            return float("nan")
        accs = [correct[g] / total[g] for g in total.keys()]
        return float(min(accs))

    @torch.no_grad()
    def _compute_groupwise_thresholds(
        self,
        model: nn.Module,
        du_loader: DataLoader,
        dl_group_counts: np.ndarray,
        n_a: int,
        tau_gmin: float,
        max_items: int | None,
        device: torch.device,
    ) -> np.ndarray:
        """
        Implements Eq. (5–7) using DU^o predictions:
          gmin = argmin_g |DL^o(g)|  (Eq. 5)
          cap  = |DL^o(gmin)| + |DU^o(gmin, tau_gmin)|  (Eq. 7 RHS)
          for g != gmin, choose smallest tau_g s.t. |DL^o(g)| + |DU^o(g, tau_g)| <= cap  (Eq. 7)

        Returns tau per group_id in [0, n_y*n_a).
        """
        model.eval()

        G = int(dl_group_counts.shape[0])
        # Eq. (5): smallest-population group in DL^o
        # Prefer positive-count groups if any exist; otherwise allow zeros.
        pos = np.where(dl_group_counts > 0)[0]
        if len(pos) > 0:
            gmin = int(pos[np.argmin(dl_group_counts[pos])])
        else:
            gmin = int(np.argmin(dl_group_counts))

        conf_by_group: list[list[float]] = [[] for _ in range(G)]

        seen = 0
        for x, y, _idx in du_loader:
            x = x.to(device)
            y_np = y.numpy().astype(np.int64)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, a_hat = probs.max(dim=1)

            conf_np = conf.detach().cpu().numpy()
            a_hat_np = a_hat.detach().cpu().numpy().astype(np.int64)

            g_hat = (y_np * n_a + a_hat_np).astype(np.int64)

            for gi, ci in zip(g_hat, conf_np, strict=False):
                if 0 <= gi < G:
                    conf_by_group[int(gi)].append(float(ci))

            seen += len(y_np)
            if max_items is not None and seen >= max_items:
                break

        # compute cap using gmin with fixed tau_gmin
        confs_gmin = np.array(conf_by_group[gmin], dtype=np.float32)
        du_gmin_count = int(np.sum(confs_gmin >= float(tau_gmin)))
        cap = int(dl_group_counts[gmin]) + du_gmin_count

        tau = np.zeros(G, dtype=np.float32)
        tau[gmin] = float(tau_gmin)

        eps = 1e-6
        for g in range(G):
            if g == gmin:
                continue

            allowed = cap - int(dl_group_counts[g])
            if allowed <= 0:
                tau[g] = 1.0  # exclude all (conf in [0,1])
                continue

            confs = np.array(conf_by_group[g], dtype=np.float32)
            if confs.size <= allowed:
                tau[g] = 0.0  # include all
                continue

            # We want |{conf >= tau}| <= allowed and tau minimal.
            # With >=, ties can inflate counts. Use tau just above the allowed-th largest confidence.
            confs_sorted = np.sort(confs)[::-1]  # descending
            kth = float(confs_sorted[allowed - 1])
            tau[g] = min(1.0, kth + eps)

        return tau

    def fit(
        self,
        # DU: group-unlabeled
        du_embeddings: np.ndarray,
        du_labels: np.ndarray,
        # DL: group-labeled (has spurious attribute labels)
        dl_embeddings: np.ndarray,
        dl_labels: np.ndarray,
        dl_spurious: np.ndarray,
    ) -> SSADetector:
        """
        End-to-end SSA:
          Phase 1: pseudo-label spurious attribute for DU (Algorithm 1, Eq. 5–9).
          Phase 2: GroupDRO on pseudo-groups with validation on DL.
        """
        self._set_seed()
        cfg = self.config
        device = self._device()
        rng = np.random.RandomState(cfg.seed)

        # infer cardinalities
        n_y = int(max(np.max(du_labels), np.max(dl_labels))) + 1
        n_a = int(np.max(dl_spurious)) + 1
        G = n_y * n_a

        # Phase 1: Algorithm 1 pseudo-labeling with K folds
        du_ds = EmbeddingUnlabeledDataset(du_embeddings, du_labels)
        dl_full_ds = EmbeddingLabeledAttrDataset(dl_embeddings, dl_labels, dl_spurious)

        dl_train_idx, dl_val_idx = self._split_indices(len(dl_full_ds), cfg.dl_val_fraction, rng)
        dl_train_ds = torch.utils.data.Subset(dl_full_ds, dl_train_idx.tolist())
        dl_val_ds = torch.utils.data.Subset(dl_full_ds, dl_val_idx.tolist())

        dl_train_loader = build_loader(
            LoaderRequest(
                stage="dl_train",
                dataset=dl_train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )
        dl_val_loader = build_loader(
            LoaderRequest(
                stage="dl_val",
                dataset=dl_val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            ),
            loader_factory=cfg.loader_factory,
            stage_loader_overrides=cfg.stage_loader_overrides,
        )

        folds = self._kfold_indices(len(du_ds), cfg.K, rng)
        a_hat_all = np.zeros(len(du_ds), dtype=np.int64)

        # Precompute DL^o group counts: |DL^o(g)| for g=(y,a)
        dl_y_train = dl_labels[dl_train_idx]
        dl_a_train = dl_spurious[dl_train_idx]
        dl_g_train = self._encode_group(dl_y_train, dl_a_train, n_a)
        dl_group_counts = np.bincount(dl_g_train, minlength=G).astype(np.int64)

        self.attr_models_.clear()

        for k in range(cfg.K):
            du_bullet_idx = folds[k]  # D_U^(k)
            du_circ_idx = np.concatenate([folds[j] for j in range(cfg.K) if j != k]).astype(
                np.int64
            )

            du_circ_ds = torch.utils.data.Subset(du_ds, du_circ_idx.tolist())
            du_bullet_ds = torch.utils.data.Subset(du_ds, du_bullet_idx.tolist())

            du_circ_loader = build_loader(
                LoaderRequest(
                    stage="du_train",
                    dataset=du_circ_ds,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=False,
                ),
                loader_factory=cfg.loader_factory,
                stage_loader_overrides=cfg.stage_loader_overrides,
            )
            # a second loader (no shuffle) for threshold updates so subsampling is stable-ish
            du_circ_loader_eval = build_loader(
                LoaderRequest(
                    stage="du_eval",
                    dataset=du_circ_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                ),
                loader_factory=cfg.loader_factory,
                stage_loader_overrides=cfg.stage_loader_overrides,
            )
            du_bullet_loader = build_loader(
                LoaderRequest(
                    stage="du_bullet",
                    dataset=du_bullet_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=False,
                ),
                loader_factory=cfg.loader_factory,
                stage_loader_overrides=cfg.stage_loader_overrides,
            )

            d = du_embeddings.shape[1]
            model = AttrPredictor(d=d, n_a=n_a, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(
                device
            )

            opt = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
            ce = nn.CrossEntropyLoss(reduction="none")

            # initialize thresholds
            tau = np.full(G, cfg.tau_gmin, dtype=np.float32)

            # train loop for T iterations (Algorithm 1 lines 7–12) :contentReference[oaicite:3]{index=3}
            it_dl = iter(dl_train_loader)
            it_du = iter(du_circ_loader)

            best_worst = -1.0
            best_state = None

            for t in range(1, cfg.T + 1):
                model.train()

                try:
                    xL, yL, aL = next(it_dl)
                except StopIteration:
                    it_dl = iter(dl_train_loader)
                    xL, yL, aL = next(it_dl)

                try:
                    xU, yU, _idxU = next(it_du)
                except StopIteration:
                    it_du = iter(du_circ_loader)
                    xU, yU, _idxU = next(it_du)

                xL = xL.to(device)
                aL = aL.to(device)
                xU = xU.to(device)

                # supervised loss on DL^o (Eq. 3/4 supervised term)
                logitsL = model(xL)
                loss_sup = ce(logitsL, aL).mean()

                # unsupervised loss on DU^o with group-wise threshold (Eq. 8)
                logitsU = model(xU)
                probsU = torch.softmax(logitsU, dim=1)
                confU, a_hatU = probsU.max(dim=1)

                # pseudo-group g_hat=(y, a_hat(x))
                yU_np = yU.numpy().astype(np.int64)
                a_hatU_np = a_hatU.detach().cpu().numpy().astype(np.int64)
                g_hat_np = (yU_np * n_a + a_hatU_np).astype(np.int64)

                tau_batch = torch.from_numpy(tau[g_hat_np]).to(device)
                mask = (confU >= tau_batch).float()

                loss_unsup = (mask * ce(logitsU, a_hatU)).sum() / mask.sum().clamp_min(1.0)

                loss = loss_sup + loss_unsup

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                # update thresholds periodically (Algorithm 1 line 10; Eq. 7) :contentReference[oaicite:4]{index=4}
                if t % max(1, cfg.threshold_update_every) == 0:
                    tau = self._compute_groupwise_thresholds(
                        model=model,
                        du_loader=du_circ_loader_eval,
                        dl_group_counts=dl_group_counts,
                        n_a=n_a,
                        tau_gmin=cfg.tau_gmin,
                        max_items=cfg.threshold_update_max_items,
                        device=device,
                    )

                # optional: track best pseudo-labeler via worst-group attr acc on DL^bullet (Appendix A.3)
                if t % 200 == 0:
                    worst = self._eval_attr_worst_group_acc(model, dl_val_loader, n_y, n_a, device)
                    if worst > best_worst:
                        best_worst = worst
                        best_state = {
                            kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()
                        }

            if best_state is not None:
                model.load_state_dict(best_state)

            self.attr_models_.append(model)

            # Predict pseudo-attributes on D_U^bullet (Algorithm 1 lines 13–14) :contentReference[oaicite:5]{index=5}
            model.eval()
            for xU, _yU, idxU in du_bullet_loader:
                xU = xU.to(device)
                logits = model(xU)
                pred_a = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                idxU_np = idxU.numpy().astype(np.int64)
                a_hat_all[idxU_np] = pred_a

        # Eq. (9): final pseudo-labels for all DU without threshold :contentReference[oaicite:6]{index=6}
        self.a_hat_ = a_hat_all

        # Phase 2: robust training on pseudo-groups, validate on DL (Section 4.3) :contentReference[oaicite:7]{index=7}
        g_train = self._encode_group(du_labels, a_hat_all, n_a)
        g_val = self._encode_group(dl_labels, dl_spurious, n_a)

        group_ids = np.arange(G, dtype=np.int64)

        gdro = GroupDRODetector(cfg.groupdro)
        gdro.fit_with_val(
            train_embeddings=du_embeddings,
            train_labels=du_labels,
            train_group_labels=g_train,
            val_embeddings=dl_embeddings,
            val_labels=dl_labels,
            val_group_labels=g_val,
            group_ids=group_ids,
        )

        self.groupdro_ = gdro

        gdro_report = gdro.get_report()
        gdro_detail = gdro_report.get("report", {})
        final = gdro_detail.get("final", {})
        avg_acc = final.get("avg_acc", float("nan"))
        worst_acc = final.get("worst_group_acc", float("nan"))
        gap = (
            avg_acc - worst_acc if np.isfinite(avg_acc) and np.isfinite(worst_acc) else float("nan")
        )

        self.shortcut_detected_ = self.get_shortcut_detected(avg_acc, worst_acc)

        self.report_ = {
            "pseudo_attr_hat": a_hat_all,  # length |DU|
            "groupdro_report": gdro_detail,
        }
        metrics = {
            "n_labeled": int(len(dl_embeddings)),
            "n_unlabeled": int(len(du_embeddings)),
            "avg_acc": avg_acc,
            "worst_group_acc": worst_acc,
            "gap": gap,
        }
        metadata = {
            "K": cfg.K,
            "T": cfg.T,
            "tau_gmin": cfg.tau_gmin,
            "n_y": n_y,
            "n_a": n_a,
            "n_groups": G,
            "ssa_gap_threshold": cfg.ssa_gap_threshold,
        }
        if self.shortcut_detected_ is None:
            risk_level = "unknown"
        elif self.shortcut_detected_:
            risk_level = "moderate"
        else:
            risk_level = "low"

        self._set_results(
            shortcut_detected=self.shortcut_detected_,
            risk_level=risk_level,
            metrics=metrics,
            notes="SSA pseudo-labeling + GroupDRO gap-based detection.",
            metadata=metadata,
            report=self.report_,
        )
        self._is_fitted = True
        return self

    def get_shortcut_detected(self, avg_acc: float, worst_acc: float) -> bool | None:
        """Detect a shortcut if worst-group gap exceeds threshold."""
        gap = avg_acc - worst_acc
        gap_thresh = self.config.ssa_gap_threshold
        if not np.isfinite(gap):
            return None
        return gap >= gap_thresh

    def get_report(self) -> dict:
        return super().get_report()
