"""Group DRO (Sagawa et al. 2020) with the shared CNN.

Keeps a weight per (task, attribute) group, raises it exponentially with the
group's loss, and minimizes the weighted sum: the worst group drives training.
"""

import torch
from torch import nn

from shortcut_detect.leaderboard.image_models import TorchPredictor, make_cnn, seed_everything

METADATA = {
    "name": "group-dro-cnn",
    "title": "Group DRO (shared CNN)",
    "track": "image-mitigation",
    "authors": ["ShortKit-ML maintainers"],
    "paper": "https://arxiv.org/abs/1911.08731",
    "description": "Online group DRO (step size 0.01) over (task, attribute) groups, 8 epochs, Adam.",
    "baseline": True,
}


def fit(images, attribute, task_labels, seed=0, epochs=8, eta=0.01, batch_size=64):
    seed_everything(seed)
    model = make_cnn()
    X = torch.as_tensor(images)
    Y = torch.as_tensor(task_labels, dtype=torch.long)
    G = torch.as_tensor(task_labels * 2 + attribute, dtype=torch.long)
    q = torch.ones(4) / 4
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(Y), generator=gen)
        for s in range(0, len(Y), batch_size):
            idx = order[s : s + batch_size]
            loss = nn.functional.cross_entropy(model(X[idx]), Y[idx], reduction="none")
            g = G[idx]
            group_loss = torch.stack(
                [loss[g == k].mean() if (g == k).any() else loss.new_zeros(()) for k in range(4)]
            )
            q = q * torch.exp(eta * group_loss.detach())
            q = q / q.sum()
            opt.zero_grad()
            (q * group_loss).sum().backward()
            opt.step()
    model.eval()
    return TorchPredictor(model)
