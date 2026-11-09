"""Tests for the community leaderboard (shortcut_detect.leaderboard)."""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from shortcut_detect.leaderboard import (
    TRACKS,
    board,
    discover_submissions,
    embedding_tracks,
    image_data,
    image_tracks,
    load_submission,
)
from shortcut_detect.leaderboard.loader import CONTRIB_DIR, SubmissionError
from shortcut_detect.leaderboard.metrics import auroc, worst_and_mean
from shortcut_detect.leaderboard.runner import compare, score_submission
from shortcut_detect.leaderboard.templates import render


def write_method(folder, body: str):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "method.py").write_text(textwrap.dedent(body))
    return folder


GOOD = """
METADATA = {
    "name": "%s",
    "title": "Test",
    "track": "embedding-detection",
    "authors": ["A. Person"],
    "description": "test",
}

def detect(embeddings, attribute, task_labels, seed=0):
    import numpy as np
    return float(np.abs(np.corrcoef(embeddings.T, attribute)[-1, :-1]).max())
"""


def test_load_valid_submission(tmp_path):
    sub = load_submission(write_method(tmp_path / "corr-max", GOOD % "corr-max"))
    assert sub.name == "corr-max"
    assert sub.track.key == "embedding-detection"
    assert callable(sub.entry)


@pytest.mark.parametrize(
    "folder,body,needle",
    [
        ("other", GOOD % "corr-max", "must match the folder name"),
        (
            "bad-track",
            (GOOD % "bad-track").replace("embedding-detection", "nope"),
            "track must be one of",
        ),
        (
            "no-entry",
            (GOOD % "no-entry").replace("def detect", "def other"),
            "needs a function `detect`",
        ),
        (
            "no-authors",
            (GOOD % "no-authors").replace('"authors": ["A. Person"],', ""),
            "missing 'authors'",
        ),
        (
            "bad-url",
            (GOOD % "bad-url").replace('"description"', '"paper": "arxiv 123", "description"'),
            "URL",
        ),
    ],
)
def test_invalid_metadata(tmp_path, folder, body, needle):
    with pytest.raises(SubmissionError, match=needle):
        load_submission(write_method(tmp_path / folder, body))


@pytest.mark.parametrize("track", sorted(TRACKS))
def test_templates_load(tmp_path, track):
    folder = tmp_path / "fresh-method"
    folder.mkdir()
    (folder / "method.py").write_text(render("fresh-method", track))
    sub = load_submission(folder)
    assert sub.metadata["track"] == track
    labels = np.zeros(4, int)
    data = np.zeros((4, 3, 32, 32)) if track.startswith("image") else np.zeros((4, 8))
    args = (data, labels, None) if track == "image-detection" else (data, labels, labels)
    with pytest.raises(NotImplementedError):
        sub.entry(*args)


def test_embedding_suite_hides_dimensions():
    a, b = [ds for ds, _ in zip(embedding_tracks.detection_suite("quick"), range(2), strict=False)]
    assert a.embeddings.shape == (600, 64)
    # same generator settings, different seeds: the attribute columns move
    assert a.seed != b.seed


def test_embedding_detection_scores_oracle_and_noise():
    def oracle(E, A, Y, seed=0):  # uses the attribute directly: mean |corr| with A within Y
        return float(
            np.mean([np.abs(np.corrcoef(E[Y == y].T, A[Y == y])[-1, :-1]).max() for y in (0, 1)])
        )

    def noise(E, A, Y, seed=0):
        return {"score": float(np.random.default_rng(seed).random()), "flag": False}

    good = embedding_tracks.score_detection(oracle, suite="quick")["metrics"]
    rand = embedding_tracks.score_detection(noise, suite="quick")["metrics"]
    assert good["auroc_linear"] > 0.9
    assert rand["fpr"] == 0.0 and rand["tpr"] == 0.0
    assert set(good) >= {"auroc", "auroc_linear", "auroc_xor", "auroc_variance"}


def test_embedding_mitigation_reweighting_beats_erm():
    from sklearn.linear_model import LogisticRegression

    def reweight(E, A, Y, seed=0):
        g = Y * 2 + A
        w = 1.0 / np.bincount(g, minlength=4)[g]
        return LogisticRegression(max_iter=2000).fit(E, Y, sample_weight=w / w.mean())

    m = embedding_tracks.score_mitigation(reweight, suite="quick")["metrics"]
    assert m["gain_over_erm"] > 0.1
    assert 0 <= m["worst_group_acc"] <= m["mean_acc"] <= 1


def test_shapes_generator():
    ds = image_data.generate_shapes(200, 1.0, seed=3)
    assert ds.images.shape == (200, 3, 32, 32) and ds.images.dtype == np.float32
    assert 0.0 <= ds.images.min() and ds.images.max() <= 1.0
    assert np.array_equal(ds.attribute, ds.task_labels)  # rho = 1
    r0, r1, c0, c1 = ds.patch_box
    patched = ds.images[ds.attribute == 1][:, 0, r0:r1, c0:c1]
    assert np.allclose(patched, 0.95)
    again = image_data.generate_shapes(200, 1.0, seed=3)
    assert np.array_equal(ds.images, again.images)


def test_pointing_game():
    heat = np.zeros((2, 32, 32))
    r0, r1, c0, c1 = image_data.patch_box("top-left")
    heat[0, r0, c0] = 1.0
    heat[1, 20, 20] = 1.0
    assert image_tracks.pointing_hits(heat, np.array([1, 1]), (r0, r1, c0, c1)) == [True, False]


def test_metrics():
    assert auroc([2, 3], [0, 1]) == 1.0
    wga, mean, groups = worst_and_mean([1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0])
    assert wga == 1.0 and len(groups) == 4


def test_compare_flags_drift():
    old = {"benchmark_version": "1.0", "metrics": {"auroc": 0.8}}
    assert compare(old, {"benchmark_version": "1.0", "metrics": {"auroc": 0.8}}, 1e-6) == []
    assert compare(old, {"benchmark_version": "1.0", "metrics": {"auroc": 0.9}}, 1e-6)
    assert compare(old, {"benchmark_version": "2.0", "metrics": {"auroc": 0.8}}, 1e-6)


def test_build_board(tmp_path):
    folder = write_method(tmp_path / "corr-max", GOOD % "corr-max")
    sub = load_submission(folder)
    res = score_submission(sub, suite="quick")
    res["suite"] = "full"  # pretend, so the board lists it
    (folder / "results.json").write_text(json.dumps(res))
    page = tmp_path / "index.md"
    page.write_text(f"intro\n{board.START}\nold\n{board.END}\nfooter\n")
    data = tmp_path / "leaderboard.json"
    board.build(root=tmp_path, page_path=page, data_path=data)
    text = page.read_text()
    assert (
        "corr-max" in text and "old" not in text and text.startswith("intro") and "footer" in text
    )
    assert "No entries yet" in text  # the other tracks are empty here
    assert board.build(check=True, root=tmp_path, page_path=page, data_path=data)
    assert (
        json.loads(data.read_text())["tracks"]["embedding-detection"]["entries"][0]["name"]
        == "corr-max"
    )


def test_bundled_submissions_are_valid_and_scored():
    folders = discover_submissions(CONTRIB_DIR)
    assert len(folders) >= 12
    tracks = set()
    for f in folders:
        sub = load_submission(f)
        tracks.add(sub.metadata["track"])
        res = sub.load_results()
        assert res is not None, f"{f.name} has no results.json"
        assert res["suite"] == "full" and res["track"] == sub.metadata["track"]
    assert tracks == set(TRACKS)


def test_leaderboard_page_is_current():
    assert board.build(check=True), "run `python -m shortcut_detect.leaderboard build`"


def test_image_detection_labels_by_measured_reliance():
    assert image_tracks.label(0.5, 0.9) == "null"
    assert image_tracks.label(0.9, image_tracks.RELIANCE_POSITIVE) == "shortcut"
    assert image_tracks.label(1.0, 0.05) == "ambiguous"

    class PatchReader:  # predicts 1 exactly when the top-left patch is red
        def eval(self):
            return self

        def __call__(self, X):
            import torch

            r = X[:, 0, 2:6, 2:6].mean((1, 2))
            return torch.stack([0.5 - r, r - 0.5], 1)

    ds = image_data.generate_shapes(100, 1.0, seed=0, corner="top-left")
    assert image_tracks.patch_reliance(PatchReader(), ds) == 1.0


def test_cli_messages_point_submitters_at_the_fix(tmp_path):
    from shortcut_detect.leaderboard.cli import describe, todo_fields

    assert "fill in the template" in describe(NotImplementedError(), tmp_path)
    folder = write_method(
        tmp_path / "boom", (GOOD % "boom").replace("return float", "return undefined_name + float")
    )
    sub = load_submission(folder)
    try:
        sub.entry(np.zeros((4, 3)), np.zeros(4), np.zeros(4))
    except NameError as exc:
        msg = describe(exc, folder)
    assert msg.startswith("NameError") and "method.py:" in msg
    meta = load_submission(
        write_method(tmp_path / "fresh", render("fresh", "embedding-detection"))
    ).metadata
    assert todo_fields(meta) == ["authors", "description"]


def test_locate_shortcut_finds_the_patch_not_the_task_signal():
    from shortcut_detect.leaderboard.image_data import generate_shapes
    from shortcut_detect.leaderboard.package_methods import locate_shortcut

    for seed in (1, 2, 3):
        ds = generate_shapes(400, 0.9, seed)
        mask = locate_shortcut(ds.images, ds.attribute, ds.task_labels)
        r0, r1, c0, c1 = ds.patch_box
        assert mask[r0:r1, c0:c1].all()
        assert mask.sum() <= 36


def test_detection_ranking_demotes_flags_that_fire_on_nulls(tmp_path):
    track = TRACKS["embedding-detection"]
    entries = {
        "flag-all": {"auroc": 0.9, "fpr": 1.0, "tpr": 1.0},
        "calibrated": {"auroc": 0.6, "fpr": 0.05, "tpr": 0.3},
        "score-only": {"auroc": 0.7},
    }
    for name, metrics in entries.items():
        folder = write_method(tmp_path / name, GOOD % name)
        res = {
            "method": name,
            "track": track.key,
            "benchmark_version": track.version,
            "suite": "full",
            "seed_offset": 0,
            "metrics": metrics,
        }
        (folder / "results.json").write_text(json.dumps(res))
    order = [r["name"] for r in board.collect(tmp_path)["embedding-detection"]]
    assert order == ["calibrated", "score-only", "flag-all"]
    page = tmp_path / "index.md"
    page.write_text(f"{board.START}\n{board.END}\n")
    board.build(root=tmp_path, page_path=page, data_path=tmp_path / "lb.json")
    assert "flag FPR > 0.2" in page.read_text()
    # mitigation tracks have no flags and are ranked by their metric alone
    assert board.flag_tier(TRACKS["embedding-mitigation"], {"fpr": 1.0}) == 0


def test_paper_accepts_a_list_of_links(tmp_path):
    body = GOOD.replace(
        '"description": "test",',
        '"description": "test",\n    "paper": ["https://arxiv.org/abs/1", "https://doi.org/10.1/x"],',
    )
    sub = load_submission(write_method(tmp_path / "two-papers", body % "two-papers"))
    assert (
        board._paper_cell(sub.metadata)
        == "[1](https://arxiv.org/abs/1), [2](https://doi.org/10.1/x)"
    )
    bad = GOOD.replace(
        '"description": "test",', '"description": "test",\n    "paper": ["not a url"],'
    )
    with pytest.raises(SubmissionError, match="paper must be a URL"):
        load_submission(write_method(tmp_path / "bad-paper", bad % "bad-paper"))


def test_every_bundled_entry_links_a_paper():
    from shortcut_detect.leaderboard.loader import paper_links

    preprint = "https://doi.org/10.64898/2026.04.29.26352053"
    for f in discover_submissions(CONTRIB_DIR):
        meta = load_submission(f).metadata
        links = paper_links(meta)
        assert links, f"{f.name} has no paper link"
        if f.name.startswith("shortkit-"):
            assert links[-1] == preprint, f"{f.name} should end with the ShortKit-ML preprint"


def test_helper_modules_stay_with_their_submission(tmp_path):
    bodies = {}
    for name, value in (("helper-one", 1.0), ("helper-two", 2.0)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "helpers.py").write_text(f"VALUE = {value}\n")
        # imported inside detect(), after method.py has loaded
        body = (GOOD % name).replace(
            "    import numpy as np\n    return float(np.abs(np.corrcoef(embeddings.T, attribute)[-1, :-1]).max())",
            "    from helpers import VALUE\n    return VALUE",
        )
        bodies[name] = write_method(folder, body)
    one, two = (load_submission(bodies[n]) for n in ("helper-one", "helper-two"))
    from shortcut_detect.leaderboard.loader import submission_imports

    args = (np.zeros((4, 3)), np.zeros(4), np.zeros(4))
    with submission_imports(one.path):
        assert one.entry(*args) == 1.0
    with submission_imports(two.path):
        assert two.entry(*args) == 2.0
    with submission_imports(one.path):
        assert one.entry(*args) == 1.0


def test_compare_reports_metrics_that_disappeared():
    old = {"benchmark_version": "1.0", "metrics": {"auroc": 0.8, "fpr": 0.0}}
    problems = compare(old, {"benchmark_version": "1.0", "metrics": {"auroc": 0.8}}, 1e-6)
    assert problems == ["fpr: committed 0.0, no longer produced"]


def test_zero_scores_rank_by_value_not_as_missing():
    track = TRACKS["embedding-mitigation"]

    def row(name, wga):
        return {"name": name, "stale": False, "flag_tier": 0, "metrics": {track.primary: wga}}

    rows = [row("a-missing", None), row("b-zero", 0.0), row("c-negative", -0.1)]
    order = [r["name"] for r in sorted(rows, key=lambda r: board.rank_key(track, r))]
    assert order == ["b-zero", "c-negative", "a-missing"]


def test_template_and_table_survive_quotes_and_pipes(tmp_path):
    title = 'The "Clever Hans" probe | v2 \\ [beta]'
    sub = load_submission(
        write_method(tmp_path / "odd-title", render("odd-title", "embedding-detection", title))
    )
    assert sub.metadata["title"] == title
    r = {"meta": sub.metadata, "folder": "odd-title", "stale": False}
    cell = board._method_cell(r)
    assert "|" not in cell.replace("\\|", "")
    assert cell.startswith('[The "Clever Hans" probe \\| v2 \\\\ \\[beta\\]](')


def test_image_detection_gives_each_submission_its_own_model(monkeypatch):
    import torch

    from shortcut_detect.leaderboard import image_models

    # tiny, fast stand-in for the shared CNN training
    monkeypatch.setattr(image_tracks, "DETECTION_SEEDS", {"quick": {0.5: 1, 1.0: 1}})
    monkeypatch.setattr(image_tracks, "_cache", {})
    monkeypatch.setattr(
        image_tracks,
        "train_classifier",
        lambda X, y, seed: image_models.make_cnn().eval(),
    )

    def vandal(images, task_labels, model, seed=0):
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        return 0.0

    def reader(images, task_labels, model, seed=0):
        return float(sum(p.abs().sum() for p in model.parameters()))

    image_tracks.score_detection(vandal, suite="quick")
    scores = image_tracks.score_detection(reader, suite="quick")["details"]["per_dataset"]
    assert all(r["score"] > 0 for r in scores)
    with pytest.raises(ValueError, match="dict with a 'score'"):
        image_tracks.score_detection(lambda *a, **k: {"flag": True}, suite="quick")
