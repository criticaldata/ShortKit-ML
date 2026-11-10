"""Load and validate a submission folder under ``contrib/``."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .spec import OPTIONAL_METADATA, REQUIRED_METADATA, TRACKS

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


def _find_root() -> Path:
    """The checkout that holds ``contrib/`` and ``docs/leaderboard/``.

    ``SHORTKIT_LEADERBOARD_ROOT`` wins; then the source tree this module sits in (an
    editable install or a clone); then the current directory, so that a pip-installed
    package scaffolds into ``./contrib`` rather than into site-packages.
    """
    env = os.environ.get("SHORTKIT_LEADERBOARD_ROOT")
    if env:
        return Path(env).resolve()
    source = Path(__file__).resolve().parents[2]
    if (source / "contrib").is_dir():
        return source
    return Path.cwd()


REPO_ROOT = _find_root()
CONTRIB_DIR = REPO_ROOT / "contrib"
# Every submission folder loaded in this process, so that helper modules one
# submission imported can be told apart from another's (see submission_imports).
_FOLDERS: set[Path] = set()


class SubmissionError(ValueError):
    pass


@dataclass
class Submission:
    path: Path
    module: ModuleType
    metadata: dict

    @property
    def name(self) -> str:
        return self.metadata["name"]

    @property
    def track(self):
        return TRACKS[self.metadata["track"]]

    @property
    def entry(self):
        return getattr(self.module, self.track.entry)

    @property
    def results_path(self) -> Path:
        return self.path / "results.json"

    def load_results(self) -> dict | None:
        if not self.results_path.exists():
            return None
        return json.loads(self.results_path.read_text())


def validate_metadata(meta: dict, folder: Path) -> list[str]:
    problems = []
    if not isinstance(meta, dict):
        return ["METADATA must be a dict"]
    for key in REQUIRED_METADATA:
        if not meta.get(key):
            problems.append(f"METADATA is missing '{key}'")
    unknown = set(meta) - set(REQUIRED_METADATA) - set(OPTIONAL_METADATA)
    if unknown:
        problems.append(f"METADATA has unknown keys {sorted(unknown)}")
    name = meta.get("name", "")
    if name and not NAME_RE.match(name):
        problems.append("name must be lowercase letters, digits and hyphens (2-63 chars)")
    if name and name != folder.name:
        problems.append(f"name '{name}' must match the folder name '{folder.name}'")
    if meta.get("track") and meta["track"] not in TRACKS:
        problems.append(f"track must be one of {sorted(TRACKS)}")
    authors = meta.get("authors")
    if authors is not None and (
        not isinstance(authors, (list, tuple)) or not all(isinstance(a, str) and a for a in authors)
    ):
        problems.append("authors must be a list of names")
    code = meta.get("code")
    if code and not str(code).startswith(("http://", "https://")):
        problems.append("code must be a URL")
    paper = meta.get("paper")
    if paper and not all(
        isinstance(u, str) and u.startswith(("http://", "https://")) for u in paper_links(meta)
    ):
        problems.append("paper must be a URL or a list of URLs")
    return problems


def paper_links(meta: dict) -> list[str]:
    """METADATA["paper"] as a list: one URL, or several (the method's paper first)."""
    paper = meta.get("paper")
    if not paper:
        return []
    return [paper] if isinstance(paper, str) else list(paper)


def _owner(module) -> Path | None:
    file = getattr(module, "__file__", None)
    if not isinstance(file, str):
        return None
    return next((f for f in _FOLDERS if file.startswith(f"{f}{os.sep}")), None)


@contextmanager
def submission_imports(folder: Path) -> Iterator[None]:
    """Make ``folder``'s helper modules importable, and only this folder's.

    Helpers are imported under their plain names (``import helpers``), so two
    submissions with a ``helpers.py`` would share whichever was imported first.
    Modules another submission imported are dropped from ``sys.modules`` on entry,
    and the folder stays on ``sys.path`` for the whole block, so imports inside
    ``detect`` or ``fit`` work too.
    """
    folder = Path(folder).resolve()
    _FOLDERS.add(folder)
    for name, mod in list(sys.modules.items()):
        owner = _owner(mod)
        if owner is not None and owner != folder:
            del sys.modules[name]
    sys.path.insert(0, str(folder))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(folder))
        except ValueError:
            pass


def load_submission(path: str | Path) -> Submission:
    folder = Path(path).resolve()
    if folder.is_file():
        folder = folder.parent
    method_file = folder / "method.py"
    if not method_file.exists():
        raise SubmissionError(f"{folder} has no method.py")
    mod_name = f"shortkit_contrib_{folder.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, method_file)
    module = importlib.util.module_from_spec(spec)
    with submission_imports(folder):
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise SubmissionError(f"importing {method_file} failed: {exc!r}") from exc
    meta = getattr(module, "METADATA", None)
    problems = validate_metadata(meta, folder)
    if not problems:
        entry = TRACKS[meta["track"]].entry
        if not callable(getattr(module, entry, None)):
            problems.append(f"track '{meta['track']}' needs a function `{entry}` in method.py")
    if problems:
        raise SubmissionError(f"{folder.name}: " + "; ".join(problems))
    return Submission(path=folder, module=module, metadata=dict(meta))


def discover_submissions(root: str | Path | None = None) -> list[Path]:
    root = Path(root) if root else CONTRIB_DIR
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "method.py").exists()
    )
