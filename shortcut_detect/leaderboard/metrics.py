"""Small metric helpers shared by the tracks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores_pos, scores_null) -> float:
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_null, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    if not np.all(np.isfinite(s)):
        raise ValueError("a detection score was NaN or infinite")
    return float(roc_auc_score(y, s))


def parse_detection(out) -> tuple[float, bool | None, dict]:
    """Split a ``detect()`` return value into (score, flag, other keys).

    ``detect`` returns a float, or a dict with a ``score`` and optionally a ``flag``
    and track-specific extras (``heatmaps`` on the image track).
    """
    extras = {}
    if isinstance(out, dict):
        extras = {k: v for k, v in out.items() if k not in ("score", "flag")}
        score, flag = out.get("score"), out.get("flag")
    else:
        score, flag = out, None
    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError("detect() must return a float score or a dict with a 'score'") from exc
    if not np.isfinite(score):
        raise ValueError("detect() returned a NaN or infinite score")
    return score, (None if flag is None else bool(flag)), extras


def group_accuracies(pred, y, a) -> dict[str, float]:
    pred, y, a = (np.asarray(v).astype(int) for v in (pred, y, a))
    out = {}
    for yy in np.unique(y):
        for aa in np.unique(a):
            m = (y == yy) & (a == aa)
            if m.any():
                out[f"y{yy}_a{aa}"] = float((pred[m] == yy).mean())
    return out


def worst_and_mean(pred, y, a) -> tuple[float, float, dict[str, float]]:
    g = group_accuracies(pred, y, a)
    vals = list(g.values())
    return float(min(vals)), float(np.mean(vals)), g
