"""Community leaderboard for shortcut detection and mitigation methods.

Researchers add a method as a folder under ``contrib/`` (a ``method.py`` with a
``METADATA`` dict and one entry function). This package scores it on a shared,
procedurally generated benchmark with planted shortcuts of known strength, so
every method on a track is measured the same way.

Four tracks, scored separately and never ranked against each other:

``embedding-detection``  does the embedding carry the attribute beyond the task label?
``embedding-mitigation`` train a task classifier on embeddings that does not lean on the attribute
``image-detection``      does a trained image model rely on a hidden shortcut, and where is it?
``image-mitigation``     train an image classifier that does not lean on the shortcut

The leaderboard is separate from the paper's audit
(:func:`shortcut_detect.audit.conditional_audit`): nothing here changes what the
audit reports.

Command line: ``python -m shortcut_detect.leaderboard --help``.
"""

from .loader import Submission, discover_submissions, load_submission
from .runner import score_submission
from .spec import MAX_FLAG_FPR, TRACKS, Track

__all__ = [
    "MAX_FLAG_FPR",
    "TRACKS",
    "Track",
    "Submission",
    "load_submission",
    "discover_submissions",
    "score_submission",
]
