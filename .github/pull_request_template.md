## What this changes

<!-- One or two sentences. -->

## Method submission (delete if this PR is not one)

- **Track:** <!-- embedding-detection / embedding-mitigation / image-detection / image-mitigation -->
- **Paper:** <!-- link, or "unpublished" -->
- **Implementation:** <!-- original authors' code, or a reimplementation (of whose method?) -->

- [ ] One folder `contrib/<name>/` with `method.py`; `METADATA["name"]` equals the folder name
- [ ] `python -m shortcut_detect.leaderboard validate contrib/<name>` passes
- [ ] `python -m shortcut_detect.leaderboard score contrib/<name>` ran and `results.json` is committed
- [ ] `python -m shortcut_detect.leaderboard build` ran and `docs/leaderboard/` is committed
- [ ] Every random choice uses the `seed` argument
- [ ] The method does not read the benchmark generator's internals or special-case seeds
- [ ] No network access or file access outside the folder at scoring time
- [ ] Extra dependencies, if any, are in `contrib/<name>/requirements.txt`

See [Submit a method](https://criticaldata.github.io/ShortKit-ML/leaderboard/submit/).
