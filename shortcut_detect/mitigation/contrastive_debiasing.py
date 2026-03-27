"""
Contrastive Debiasing (M07) - Zhang et al. 2022 (Correct-n-Contrast).

Uses contrastive learning to separate shortcuts: align same-class/different-group
representations while pushing different-class apart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.utils.data  # noqa: F401 - required for PyTorch
except ImportError:
    raise ImportError(
        "ContrastiveDebiasing requires PyTorch. Install with: pip install torch"
    ) from None


class ContrastiveDebiasing:
    """
    Contrastive debiasing (Correct-n-Contrast) per Zhang et al. 2022.

    Trains an encoder on embeddings via contrastive learning: anchors are from
    one (task, group) slice; positives are same task, different group; negatives
    are different task. This aligns representations to be invariant to spurious
    (group) signals while preserving task-relevant structure.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension of the encoder output. If None, uses min(64, embed_dim).
    temperature : float
        Temperature for InfoNCE contrastive loss. Lower = sharper.
        Default 0.05.
    contrastive_weight : float
        Weight for contrastive loss vs CE (when use_task_loss=True).
        1.0 = pure contrastive. Default 0.75.
    use_task_loss : bool
        If True, jointly minimize CE loss for task prediction to preserve utility.
        Default True.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr : float
        Learning rate.
    dropout : float
        Dropout rate in the encoder.
    device : str or torch.device, optional
        Device to train on. Defaults to cuda if available else cpu.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int | None = None,
        temperature: float = 0.05,
        contrastive_weight: float = 0.75,
        use_task_loss: bool = True,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        device: str | torch.device | None = None,
        random_state: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.contrastive_weight = float(contrastive_weight)
        self.use_task_loss = bool(use_task_loss)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.device = device
        self.random_state = random_state

        self._encoder: nn.Module | None = None
        self._task_head: nn.Module | None = None
        self._embed_dim: int | None = None
        self._n_task: int | None = None
        self._task_map: dict | None = None
        self._fitted = False

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_seed(self) -> None:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _build_contrastive_batches(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        g_idx: np.ndarray,
        n_task: int,
        n_group: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Build (anchor_ix, pos_ix, neg_ix, task_labels) for each valid anchor."""
        n = X.shape[0]
        # Index by (task, group)
        by_slice: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            key = (int(y_idx[i]), int(g_idx[i]))
            if key not in by_slice:
                by_slice[key] = []
            by_slice[key].append(i)

        batches = []
        for (t, g), anchor_indices in by_slice.items():
            # Positives: same task t, different group
            pos_indices = []
            for g_other in range(n_group):
                if g_other != g:
                    key = (t, g_other)
                    if key in by_slice:
                        pos_indices.extend(by_slice[key])
            if not pos_indices:
                continue  # need at least one positive

            # Negatives: different task
            neg_indices = []
            for t_other in range(n_task):
                if t_other != t:
                    for g_other in range(n_group):
                        key = (t_other, g_other)
                        if key in by_slice:
                            neg_indices.extend(by_slice[key])
            if not neg_indices:
                continue

            pos_arr = np.array(pos_indices, dtype=np.int64)
            neg_arr = np.array(neg_indices, dtype=np.int64)
            for a_ix in anchor_indices:
                batches.append((a_ix, pos_arr, neg_arr, y_idx))

        return batches

    def fit(
        self,
        embeddings: np.ndarray,
        task_labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> ContrastiveDebiasing:
        """
        Fit the contrastive debiasing model.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, embed_dim).
        task_labels : np.ndarray
            Shape (n_samples,) – task/target labels.
        group_labels : np.ndarray
            Shape (n_samples,) – protected/group labels (spurious).

        Returns
        -------
        self : ContrastiveDebiasing
        """
        self._setup_seed()
        X = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray(task_labels)
        g = np.asarray(group_labels)

        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, embed_dim)")
        if y.ndim != 1 or g.ndim != 1:
            raise ValueError("task_labels and group_labels must be 1D")
        if X.shape[0] != y.shape[0] or X.shape[0] != g.shape[0]:
            raise ValueError("embeddings, task_labels, and group_labels must have same length")

        y_uniq = np.unique(y)
        g_uniq = np.unique(g)
        n_task = len(y_uniq)
        n_group = len(g_uniq)

        if n_group < 2:
            raise ValueError(
                "Contrastive Debiasing requires at least 2 groups. "
                "Positives are same-task, different-group samples."
            )
        if n_task < 2:
            raise ValueError(
                "Contrastive Debiasing requires at least 2 task classes for negatives."
            )

        y_map = {v: i for i, v in enumerate(y_uniq)}
        g_map = {v: i for i, v in enumerate(g_uniq)}
        y_idx = np.array([y_map[v] for v in y], dtype=np.int64)
        g_idx = np.array([g_map[v] for v in g], dtype=np.int64)

        batches = self._build_contrastive_batches(X, y_idx, g_idx, n_task, n_group)
        if not batches:
            raise ValueError(
                "Could not build contrastive batches. "
                "Ensure each task has samples in at least 2 groups."
            )

        embed_dim = X.shape[1]
        h_dim = self.hidden_dim
        if h_dim is None:
            h_dim = min(64, embed_dim)

        device = self._device()
        encoder = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ).to(device)
        task_head = nn.Linear(h_dim, n_task).to(device) if self.use_task_loss else None

        params = list(encoder.parameters())
        if task_head is not None:
            params.extend(list(task_head.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        ce = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y_idx).long().to(device)

        for _ in range(self.n_epochs):
            encoder.train()
            if task_head is not None:
                task_head.train()
            perm = np.random.permutation(len(batches))
            for idx in perm:
                a_ix, pos_ix, neg_ix, _ = batches[idx]
                n_pos = min(32, len(pos_ix))
                n_neg = min(32, len(neg_ix))
                p_perm = np.random.choice(len(pos_ix), size=n_pos, replace=n_pos > len(pos_ix))
                n_perm = np.random.choice(len(neg_ix), size=n_neg, replace=n_neg > len(neg_ix))
                pos_sel = pos_ix[p_perm]
                neg_sel = neg_ix[n_perm]

                anchor = X_t[a_ix : a_ix + 1]
                positives = X_t[pos_sel]
                negatives = X_t[neg_sel]

                all_vec = torch.cat([anchor, positives, negatives], dim=0)
                hidden = encoder(all_vec)
                z = F.normalize(hidden, dim=1)

                anchor_z = z[0:1]
                pos_z = z[1 : 1 + n_pos]
                neg_z = z[1 + n_pos :]

                sim_pos = torch.matmul(anchor_z, pos_z.T) / self.temperature
                sim_neg = torch.matmul(anchor_z, neg_z.T) / self.temperature
                logits = torch.cat([sim_pos, sim_neg], dim=1)
                labels_cl = torch.zeros(1, dtype=torch.long, device=device)
                loss_cl = F.cross_entropy(logits, labels_cl)

                loss = self.contrastive_weight * loss_cl
                if self.use_task_loss and task_head is not None:
                    batch_ix = np.concatenate([[a_ix], pos_sel, neg_sel])
                    y_batch = y_t[torch.from_numpy(batch_ix).to(device)]
                    task_logits = task_head(hidden)
                    loss_ce = ce(task_logits, y_batch)
                    loss = loss + (1.0 - self.contrastive_weight) * loss_ce

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self._encoder = encoder
        self._task_head = task_head
        self._embed_dim = embed_dim
        self._n_task = n_task
        self._task_map = y_map
        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to debiased representations.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, embed_dim). Must match embed_dim from fit.

        Returns
        -------
        np.ndarray
            Debiased embeddings, shape (n_samples, hidden_dim).
        """
        if not self._fitted or self._encoder is None:
            raise ValueError("ContrastiveDebiasing has not been fitted")
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if X.shape[1] != self._embed_dim:
            raise ValueError(
                f"embed_dim {X.shape[1]} does not match fitted embed_dim {self._embed_dim}"
            )

        device = self._device()
        self._encoder.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(device)
            hidden = self._encoder(x_t)
            out = hidden.cpu().numpy()
        return out.astype(np.float64)

    def fit_transform(
        self,
        embeddings: np.ndarray,
        task_labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings, task_labels, group_labels)
        return self.transform(embeddings)
