# contrib: leaderboard submissions

Each folder is one method on the [ShortKit-ML leaderboard](https://criticaldata.github.io/ShortKit-ML/leaderboard/):
a `method.py` (a `METADATA` dict plus `detect` or `fit`) and the `results.json` it scored.
Folders marked `"baseline": True` are reference methods maintained with the package.
The `shortkit-*` folders wrap ShortKit-ML's own detectors and mitigations
(`shortcut_detect/leaderboard/package_methods.py`), so the leaderboard measures them the
way users call them. Two package methods are not listed: SSA (Spread Spurious Attribute)
needs its own labelled/unlabelled splits, and background randomization needs foreground masks,
which the image benchmark does not provide.

```bash
python -m shortcut_detect.leaderboard new my-method --track embedding-detection
python -m shortcut_detect.leaderboard validate contrib/my-method
python -m shortcut_detect.leaderboard score contrib/my-method
python -m shortcut_detect.leaderboard build
python -m shortcut_detect.leaderboard list
```

Full guide: [Submit a method](../docs/leaderboard/submit.md). Track rules and metrics: [Tracks](../docs/leaderboard/tracks.md).
