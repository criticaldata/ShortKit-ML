"""Tracks, metadata contract and benchmark version."""

from __future__ import annotations

from dataclasses import dataclass

# A detection flag that fires on more than this share of null datasets is not a usable
# decision rule: such entries rank below every entry whose flag stays under it.
MAX_FLAG_FPR = 0.2


@dataclass(frozen=True)
class Track:
    key: str
    title: str
    entry: str  # function a submission's method.py must define
    primary: str  # metric that ranks the table (higher is better)
    columns: tuple[tuple[str, str], ...]  # (metric key, column header) shown on the board
    tolerance: float  # max abs difference allowed by `verify`
    summary: str
    # Bump when the track's suite changes in a way that changes scores; results scored on
    # an older version are shown as stale until they are rescored.
    version: str = "1.0"
    detection: bool = False  # ranked with the flag false-positive rule (MAX_FLAG_FPR)


TRACKS: dict[str, Track] = {
    t.key: t
    for t in (
        Track(
            key="embedding-detection",
            title="Embedding detection",
            entry="detect",
            primary="auroc",
            columns=(
                ("auroc", "AUROC"),
                ("auroc_linear", "linear"),
                ("auroc_xor", "xor"),
                ("auroc_variance", "variance"),
                ("fpr", "FPR"),
                ("tpr", "TPR"),
            ),
            tolerance=1e-6,
            summary=(
                "Given embeddings E, an attribute A and a task label Y, score whether E carries A "
                "beyond what Y explains. Null datasets have A correlated with Y but no extra "
                "attribute signal, so a method that only finds the correlation scores 0.5."
            ),
            version="1.1",
            detection=True,
        ),
        Track(
            key="embedding-mitigation",
            title="Embedding mitigation",
            entry="fit",
            primary="worst_group_acc",
            columns=(
                ("worst_group_acc", "Worst-group acc"),
                ("mean_acc", "Mean acc"),
                ("gain_over_erm", "Gain over ERM"),
                ("leakage_auroc", "Residual leakage"),
            ),
            # torch-trained entries (e.g. contrastive debiasing) move by ~0.002 across CI CPUs
            tolerance=0.01,
            summary=(
                "Train a task classifier on embeddings where the attribute predicts the label in "
                "training (rho >= 0.9) but not at test time (rho = 0.5). Scored by worst-group "
                "accuracy on the test set."
            ),
        ),
        Track(
            key="image-detection",
            title="Image detection",
            entry="detect",
            primary="auroc",
            columns=(
                ("auroc", "AUROC"),
                ("reliance_spearman", "Spearman vs reliance"),
                ("pointing_game", "Pointing game"),
                ("fpr", "FPR"),
                ("tpr", "TPR"),
            ),
            tolerance=0.02,
            summary=(
                "Given images, task labels and a CNN trained on them (no attribute labels), score "
                "whether the model relies on a hidden shortcut; optionally return per-image "
                "heatmaps, scored by whether their peak lands on the shortcut."
            ),
            detection=True,
        ),
        Track(
            key="image-mitigation",
            title="Image mitigation",
            entry="fit",
            primary="worst_group_acc",
            columns=(
                ("worst_group_acc", "Worst-group acc"),
                ("mean_acc", "Mean acc"),
                ("gain_over_erm", "Gain over ERM"),
            ),
            tolerance=0.02,
            summary=(
                "Train an image classifier where a colored patch predicts the label in training "
                "but not at test time. Scored by worst-group accuracy on the test set."
            ),
        ),
    )
}

REQUIRED_METADATA = ("name", "title", "track", "authors", "description")
OPTIONAL_METADATA = ("paper", "code", "venue", "year", "baseline", "contact", "license", "notes")
