"""
Adversarial Debiasing (M04) - Zhang et al. 2018.

Adversarial training to remove demographic encoding from embeddings.
Uses a Gradient Reversal Layer so the encoder learns representations that
predict the task while being uninformative for the protected attribute.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError(
        "AdversarialDebiasing requires PyTorch. Install with: pip install torch"
    ) from None


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer: forward pass identity, backward pass negates and scales gradients."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def _grl(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Apply gradient reversal in the forward pass."""
    return GradientReversalLayer.apply(x, lambda_)


class AdversarialDebiasing:
    """
    Adversarial debiasing to remove demographic encoding from embeddings (Zhang et al. 2018).

    Trains an encoder that maps embeddings to a hidden representation, with an adversary
    (via Gradient Reversal Layer) trying to predict the protected attribute from that
    representation. The encoder learns to be informative for the task while uninformative
    for the protected attribute.

    Parameters
    ----------
    hidden_dim : int, optional
        Hidden dimension of the encoder. If None, uses min(64, embed_dim).
    adversary_weight : float
        Weight (lambda) for the adversarial loss. Higher values push harder to remove
        protected-attribute information. Default 0.5.
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
        adversary_weight: float = 0.5,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        dropout: float = 0.1,
        device: str | torch.device | None = None,
        random_state: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.adversary_weight = float(adversary_weight)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.device = device
        self.random_state = random_state

        self._encoder: nn.Module | None = None
        self._adversary: nn.Module | None = None
        self._task_head: nn.Module | None = None
        self._embed_dim: int | None = None
        self._n_protected: int | None = None
        self._n_task: int | None = None
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

    def fit(
        self,
        embeddings: np.ndarray,
        protected_labels: np.ndarray,
        task_labels: np.ndarray | None = None,
    ) -> AdversarialDebiasing:
        """
        Fit the adversarial debiasing model.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_samples, embed_dim).
        protected_labels : np.ndarray
            Shape (n_samples,) – demographic/protected attribute labels.
        task_labels : np.ndarray, optional
            Shape (n_samples,) – task labels. If provided, the encoder also
            minimizes task loss to preserve utility.

        Returns
        -------
        self : AdversarialDebiasing
        """
        self._setup_seed()
        X = np.asarray(embeddings, dtype=np.float32)
        s = np.asarray(protected_labels)
        if X.ndim != 2:
            raise ValueError("embeddings must be 2D (n_samples, embed_dim)")
        if s.ndim != 1:
            raise ValueError("protected_labels must be 1D")
        if X.shape[0] != s.shape[0]:
            raise ValueError("embeddings and protected_labels must have same length")

        embed_dim = X.shape[1]
        uniq = np.unique(s)
        n_protected = len(uniq)
        s_map = {v: i for i, v in enumerate(uniq)}
        s_idx = np.array([s_map[v] for v in s], dtype=np.int64)

        n_task: int | None = None
        y_idx: np.ndarray | None = None
        if task_labels is not None:
            y = np.asarray(task_labels)
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError("task_labels must be 1D with same length as embeddings")
            y_uniq = np.unique(y)
            n_task = len(y_uniq)
            y_map = {v: i for i, v in enumerate(y_uniq)}
            y_idx = np.array([y_map[v] for v in y], dtype=np.int64)

        h_dim = self.hidden_dim
        if h_dim is None:
            h_dim = min(64, embed_dim)

        device = self._device()
        encoder = nn.Sequential(
            nn.Linear(embed_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        ).to(device)
        adversary = nn.Linear(h_dim, n_protected).to(device)
        task_head = nn.Linear(h_dim, n_task).to(device) if n_task is not None else None

        opt = torch.optim.Adam(
            list(encoder.parameters())
            + list(adversary.parameters())
            + (list(task_head.parameters()) if task_head is not None else []),
            lr=self.lr,
        )
        ce = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X).float().to(device)
        s_t = torch.from_numpy(s_idx).long().to(device)
        if y_idx is not None:
            y_t = torch.from_numpy(y_idx).long().to(device)

        ds = TensorDataset(
            X_t,
            s_t,
            *((y_t,) if y_idx is not None else ()),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        for _ in range(self.n_epochs):
            encoder.train()
            adversary.train()
            if task_head is not None:
                task_head.train()
            for batch in loader:
                x_b = batch[0]
                s_b = batch[1]
                y_b = batch[2] if len(batch) > 2 else None

                hidden = encoder(x_b)
                adv_in = _grl(hidden, self.adversary_weight)
                adv_logits = adversary(adv_in)
                loss_adv = ce(adv_logits, s_b)

                if task_head is not None and y_b is not None:
                    task_logits = task_head(hidden)
                    loss_task = ce(task_logits, y_b)
                    loss = loss_task + loss_adv
                else:
                    loss = loss_adv

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        self._encoder = encoder
        self._adversary = adversary
        self._task_head = task_head
        self._embed_dim = embed_dim
        self._n_protected = n_protected
        self._n_task = n_task
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
            raise ValueError("AdversarialDebiasing has not been fitted")
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
        protected_labels: np.ndarray,
        task_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings, protected_labels, task_labels=task_labels)
        return self.transform(embeddings)
