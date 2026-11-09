"""``python -m shortcut_detect.leaderboard``: scaffold, validate, score, verify, build."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .loader import CONTRIB_DIR, NAME_RE, SubmissionError, discover_submissions, load_submission
from .spec import MAX_FLAG_FPR, TRACKS


def _targets(paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths] if paths else discover_submissions()


def cmd_new(args) -> int:
    from .templates import render

    if not NAME_RE.match(args.name):
        print("name must be lowercase letters, digits and hyphens", file=sys.stderr)
        return 2
    folder = CONTRIB_DIR / args.name
    if folder.exists():
        print(f"{folder} already exists", file=sys.stderr)
        return 2
    folder.mkdir(parents=True)
    (folder / "method.py").write_text(render(args.name, args.track, args.title))
    print(
        f"created {folder / 'method.py'}\n"
        f"next: fill in METADATA and `{TRACKS[args.track].entry}`, then\n"
        f"  python -m shortcut_detect.leaderboard validate contrib/{args.name}\n"
        f"  python -m shortcut_detect.leaderboard score contrib/{args.name}"
    )
    return 0


REPRO_HINT = (
    "if you changed the method, rerun `score` and commit the new results.json. If not, the "
    "difference comes from your machine: scores are reproduced on the CI runner, so commit "
    "the results.json from this run's `scored-results-<track>` artifact. If the numbers change on "
    "every run, some randomness in the method is not using the `seed` argument"
)


def describe(exc: BaseException, path) -> str:
    """One line a submitter can act on: error type, message, and where in their code."""
    if isinstance(exc, SubmissionError):
        return str(exc)
    if isinstance(exc, NotImplementedError) and not str(exc):
        return "the entry function still raises NotImplementedError: fill in the template"
    folder = str(Path(path).resolve())
    where = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename.startswith(folder):
            where = f" (at {Path(frame.filename).name}:{frame.lineno}: {frame.line})"
            break
    return f"{type(exc).__name__}: {exc}{where}"


def todo_fields(meta: dict) -> list[str]:
    return sorted(k for k, v in meta.items() if "TODO" in str(v))


def cmd_validate(args) -> int:
    from .runner import score_submission

    failed = 0
    for path in _targets(args.paths):
        try:
            sub = load_submission(path)
            todo = todo_fields(sub.metadata)
            if todo:
                raise SubmissionError(f"METADATA still has TODO placeholders in {todo}")
            res = score_submission(sub, suite="quick")
            print(
                f"ok    {sub.name} [{sub.metadata['track']}] quick {sub.track.primary}="
                f"{res['metrics'].get(sub.track.primary)}"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {path}: {describe(exc, path)}")
            if args.verbose:
                traceback.print_exc()
    return 1 if failed else 0


def cmd_score(args) -> int:
    from .runner import score_submission, write_results

    failed = 0
    for path in _targets(args.paths):
        try:
            sub = load_submission(path)
            res = score_submission(sub, suite=args.suite, seed_offset=args.seed_offset)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {path}: {describe(exc, path)}")
            traceback.print_exc()
            continue
        m = res["metrics"]
        print(
            f"{sub.name} [{sub.metadata['track']}] " + ", ".join(f"{k}={v}" for k, v in m.items())
        )
        if args.suite == "full" and args.seed_offset == 0 and not args.no_write:
            write_results(sub, res)
            print(f"  wrote {sub.results_path}")
    return 1 if failed else 0


def cmd_verify(args) -> int:
    from .runner import compare, score_submission

    report, failed = [], 0
    for path in _targets(args.paths):
        name = Path(path).name
        try:
            sub = load_submission(path)
            old = sub.load_results()
            if old is None:
                raise SubmissionError("no results.json: run `score` and commit the file")
            new = score_submission(sub, suite="full")
            if args.write_dir:
                out = Path(args.write_dir) / sub.name / "results.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(new, indent=2) + "\n")
            problems = compare(old, new, sub.track.tolerance)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            msg = describe(exc, path)
            report.append({"name": name, "ok": False, "problems": [msg]})
            print(f"FAIL  {name}: {msg}")
            continue
        ok = not problems
        if problems:
            problems.append(REPRO_HINT)
        failed += not ok
        report.append(
            {
                "name": sub.name,
                "track": sub.metadata["track"],
                "ok": ok,
                "problems": problems,
                "metrics": new["metrics"],
                "primary": sub.track.primary,
            }
        )
        print(
            ("ok    " if ok else "FAIL  ") + sub.name + ("" if ok else ": " + "; ".join(problems))
        )
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if args.markdown:
        with open(args.markdown, "a") as fh:
            fh.write(verify_markdown(report))
    return 1 if failed else 0


def verify_markdown(report: list[dict]) -> str:
    """Run-summary markdown: each entry's scores and where it would rank on its track."""
    from .board import collect, flag_tier, rank_key

    board = collect()
    lines = ["## Leaderboard scoring\n"]
    for r in report:
        if not r.get("track"):
            lines.append(
                f"### ❌ `{r['name']}`\n\n" + "\n".join(f"- {p}" for p in r["problems"]) + "\n"
            )
            continue
        track = TRACKS[r["track"]]
        me = {"name": r["name"], "stale": False, "metrics": r["metrics"]}
        me["flag_tier"] = flag_tier(track, r["metrics"])
        others = [e for e in board[r["track"]] if e["name"] != r["name"] and not e["stale"]]
        rank = 1 + sum(rank_key(track, e) < rank_key(track, me) for e in others)
        status = "✅ scores reproduce" if r["ok"] else "❌ " + "; ".join(r["problems"])
        lines.append(
            f"### `{r['name']}` · {track.title}\n\n{status}\n\n"
            f"Would rank **#{rank} of {len(others) + 1}** by {dict(track.columns)[track.primary]}"
            + (
                f" (flag FPR > {MAX_FLAG_FPR:g}: ranked after calibrated flags)"
                if me["flag_tier"] == 2
                else ""
            )
            + ".\n"
        )
        lines.append("| Metric | Value |\n|---|---:|")
        lines.extend(f"| {k} | {v} |" for k, v in r["metrics"].items())
        top = others[:3]
        if top:
            lines.append(
                "\nCurrent leaders: "
                + ", ".join(
                    f"{e['meta']['title']} ({e['metrics'].get(track.primary)})" for e in top
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_build(args) -> int:
    from .board import PAGE, build

    if args.check:
        if build(check=True):
            print("leaderboard is up to date")
            return 0
        print(
            "leaderboard is out of date: run `python -m shortcut_detect.leaderboard build`",
            file=sys.stderr,
        )
        return 1
    build()
    print(f"wrote {PAGE}")
    return 0


def cmd_list(args) -> int:
    for path in discover_submissions():
        try:
            sub = load_submission(path)
            res = sub.load_results()
            score = res["metrics"].get(sub.track.primary) if res else None
            print(f"{sub.metadata['track']:<22} {sub.name:<32} {score}")
        except SubmissionError as exc:
            print(f"invalid                {path.name:<32} {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m shortcut_detect.leaderboard", description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    n = sp.add_parser("new", help="scaffold contrib/<name>/method.py")
    n.add_argument("name")
    n.add_argument("--track", required=True, choices=sorted(TRACKS))
    n.add_argument("--title")
    n.set_defaults(func=cmd_new)

    v = sp.add_parser("validate", help="load submissions and run them on the quick suite")
    v.add_argument("paths", nargs="*")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)

    s = sp.add_parser("score", help="score on the full suite and write results.json")
    s.add_argument("paths", nargs="*")
    s.add_argument("--suite", choices=("full", "quick"), default="full")
    s.add_argument(
        "--seed-offset", type=int, default=0, help="score on fresh seeds (never written)"
    )
    s.add_argument("--no-write", action="store_true")
    s.set_defaults(func=cmd_score)

    ve = sp.add_parser("verify", help="rescore and compare with the committed results.json")
    ve.add_argument("paths", nargs="*")
    ve.add_argument("--report", help="write a JSON report here")
    ve.add_argument("--markdown", help="append a markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    ve.add_argument(
        "--write-dir", help="also write each recomputed results.json to <dir>/<name>/results.json"
    )
    ve.set_defaults(func=cmd_verify)

    b = sp.add_parser("build", help="regenerate docs/leaderboard/index.md")
    b.add_argument("--check", action="store_true", help="fail if the page is out of date")
    b.set_defaults(func=cmd_build)

    li = sp.add_parser("list", help="list submissions and their primary score")
    li.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)
