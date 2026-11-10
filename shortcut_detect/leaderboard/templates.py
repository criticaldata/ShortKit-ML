"""Starter ``method.py`` files written by ``python -m shortcut_detect.leaderboard new``."""

from __future__ import annotations

import json

HEADER = '''"""{title}

TODO: two or three sentences on what the method does and the idea behind it.
Cite the paper in METADATA["paper"].
"""

'''

META = """METADATA = {{
    "name": {name},
    "title": {title},
    "track": {track},
    "authors": ["TODO Your Name"],
    "paper": None,  # "https://arxiv.org/abs/..." or a DOI link; a list if the method builds on several
    "code": None,  # your own repository, if any
    "description": "TODO one sentence shown on the leaderboard.",
}}

"""

BODIES = {
    "embedding-detection": '''import numpy as np


def detect(embeddings, attribute, task_labels, seed=0):
    """Score whether ``embeddings`` carry ``attribute`` beyond what ``task_labels`` explains.

    embeddings   float32 array (n, d); column order is shuffled per dataset
    attribute    int array (n,) with values 0/1
    task_labels  int array (n,) with values 0/1, correlated with the attribute
    seed         int; use it for any randomness so scores are reproducible

    Return a float (higher = more evidence of a shortcut), or a dict
    {"score": float, "flag": bool} to also be scored on false/true positive rates.
    """
    raise NotImplementedError
''',
    "embedding-mitigation": '''import numpy as np


class Model:
    def predict(self, embeddings):
        """Return task-label predictions (n,) for test embeddings."""
        raise NotImplementedError

    # Optional: define transform(embeddings) -> array (n, k) if your method changes
    # the representation; the benchmark then also reports residual attribute leakage.


def fit(embeddings, attribute, task_labels, seed=0):
    """Train on embeddings where the attribute predicts the task label (rho >= 0.9).

    The test set has the attribute independent of the label; you are scored on
    worst-group accuracy over the four (task, attribute) groups there.
    Return an object with ``predict(embeddings)``.
    """
    raise NotImplementedError
''',
    "image-detection": '''import numpy as np
import torch


def detect(images, task_labels, model, seed=0):
    """Score whether ``model`` relies on a hidden shortcut in ``images``.

    images       float32 array (n, 3, 32, 32) in [0, 1]
    task_labels  int array (n,)
    model        torch.nn.Module in eval mode, logits of shape (n, 2)
    seed         int; use it for any randomness

    No attribute labels are given: the method has to find the shortcut itself.
    Return a float, or a dict with "score" and optionally "flag" (bool) and
    "heatmaps" (array (n, 32, 32), higher = more responsible for the prediction).
    """
    raise NotImplementedError
''',
    "image-mitigation": '''import numpy as np

from shortcut_detect.leaderboard.image_models import TorchPredictor, make_cnn, train_classifier


def fit(images, attribute, task_labels, seed=0):
    """Train an image classifier that does not lean on the shortcut.

    images       float32 array (n, 3, 32, 32) in [0, 1]
    attribute    int array (n,), 1 = shortcut present; agrees with the label 90-97% of the time
    task_labels  int array (n,)

    You may use any model; make_cnn() / train_classifier() give the shared
    backbone the baselines use. Return an object with ``predict(images)``,
    for example TorchPredictor(model).
    """
    raise NotImplementedError
''',
}


def _docstring_text(s: str) -> str:
    return " ".join(s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').split())


def render(name: str, track: str, title: str | None = None) -> str:
    title = title or name.replace("-", " ").title()

    def lit(s: str) -> str:  # a Python string literal for any text
        return json.dumps(s, ensure_ascii=False)

    return (
        HEADER.format(title=_docstring_text(title))
        + BODIES[track].split("\n\n\n")[0]
        + "\n\n"
        + META.format(name=lit(name), title=lit(title), track=lit(track))
        + "\n\n".join(BODIES[track].split("\n\n\n")[1:])
        + "\n"
    )
