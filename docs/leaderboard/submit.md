# Submit a method

A submission is one folder under [`contrib/`](https://github.com/criticaldata/ShortKit-ML/tree/main/contrib)
with a `method.py` in it. It is scored on its track's benchmark, and when the pull
request is merged it is listed on the [leaderboard](index.md) with your name and a
link to your paper. Published methods, new methods, reimplementations of other people's
methods (credit them in `authors`) and negative results are all welcome.

## 1. Scaffold

```bash
git clone https://github.com/criticaldata/ShortKit-ML.git && cd ShortKit-ML
uv venv --python 3.10 && source .venv/bin/activate && uv pip install -e .
python -m shortcut_detect.leaderboard new my-method --track embedding-detection
```

The tracks are `embedding-detection`, `embedding-mitigation`, `image-detection` and
`image-mitigation` ([what each one asks](tracks.md)). The command writes
`contrib/my-method/method.py` with the function you need to fill in and its documented inputs.

## 2. Fill in `method.py`

```python
METADATA = {
    "name": "my-method",                 # must equal the folder name
    "title": "My Method",                # shown on the leaderboard
    "track": "embedding-detection",
    "authors": ["Ada Lovelace", "Alan Turing"],
    "paper": "https://arxiv.org/abs/...",  # optional; a list of links if several, your paper first
    "code": "https://github.com/...",      # optional, your own repository
    "description": "One sentence on the idea.",
}


def detect(embeddings, attribute, task_labels, seed=0):
    ...
    return {"score": score, "flag": p_value < 0.05}
```

- One entry function per track: `detect` for detection tracks, `fit` for mitigation tracks.
- Use `seed` for every random choice; the benchmark checks that scores reproduce.
- Helper modules next to `method.py` can be imported normally.
- Extra packages go in `contrib/my-method/requirements.txt`; CI installs them. Keep them
  pip-installable on CPU.
- Keep your method's own cost modest. A full scoring run on an embedding track takes
  minutes. A full run on an image track takes longer, because the benchmark trains the small
  shared CNN once per dataset (22 datasets for image detection, 6 settings plus their ERM
  references for image mitigation), and that alone can take well over half an hour on a laptop
  CPU. Use `validate` while developing; it runs the quick suite.
- The method may wrap code that lives in your own repository (installed through
  `requirements.txt`): the folder only needs the adapter.

## 3. Check and score

```bash
python -m shortcut_detect.leaderboard validate contrib/my-method   # quick suite, about a minute
python -m shortcut_detect.leaderboard score contrib/my-method      # full suite, writes results.json
python -m shortcut_detect.leaderboard build                         # updates the leaderboard page
```

Commit `method.py`, `results.json`, any helpers, and the updated `docs/leaderboard/`.

## 4. Open a pull request

Use the **method submission** checklist in the pull request template. CI then

1. validates `METADATA` and runs your method on the quick suite,
2. rescores it on the full suite and checks that the numbers match your `results.json`, and
3. posts the scores in the run summary, next to the current leaders of your track.

**Scores are reproduced on the CI runner, not on your machine.** Methods built on numpy and
scikit-learn usually match to the last digit everywhere. Methods that train with torch, and
everything on the image tracks (which train a CNN), can come out differently on another
operating system or processor even with the same seed. If step 2 fails only because of that,
download the `scored-results-<track>` artifact from the CI run, replace your `results.json`
with the one in it, and push. The leaderboard lists the scores CI computed. CI uses the
package versions in `uv.lock`, Python 3.11 and CPU-only torch.

A maintainer reviews the code before merging: CI runs your code, so the review checks it
does nothing beyond the method (no network calls, no reading files outside its folder).
Once merged, the leaderboard on the docs site updates with the next deploy.

## Rules

- **No peeking.** Methods may not read the benchmark's generator internals (planted
  dimensions, patch corner, rho) or special-case the seeds. Column order and patch position
  are randomized per dataset for that reason, and maintainers rescore entries on fresh
  seeds (`score --seed-offset`) before a benchmark release; an entry whose ranking only holds on
  the public seeds is removed.
- **Detection tracks see no answers.** Embedding detection gets the attribute (the question
  is whether the embedding carries it beyond the label); image detection gets no attribute
  labels at all.
- **Mitigation tracks may use group labels** on the training set, as most published
  mitigation methods do. Methods that do not need them are noted in `description`.
- **One entry per method.** Variants with different hyperparameters belong in one folder
  with the settings you recommend. A substantially different method gets its own folder.
- **Stale scores.** When a track's benchmark version changes, older results on that track are
  shown as stale until they are rescored; maintainers rescore baselines and ask authors about
  the rest.
- **Flags must mean something.** On detection tracks, an entry whose flag fires on more than
  20% of null datasets ranks below every entry whose flag does not, whatever its AUROC
  ([details](tracks.md#ranking-the-detection-tracks)).

## Not ready to write code?

Open a [method proposal issue](https://github.com/criticaldata/ShortKit-ML/issues/new?template=method-proposal.yml)
with the paper and a reference implementation, and the community or the maintainers can port it,
with you credited as an author.
