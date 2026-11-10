"""Build the leaderboard page from every ``contrib/*/results.json``."""

from __future__ import annotations

import json
from pathlib import Path

from .loader import (
    CONTRIB_DIR,
    REPO_ROOT,
    SubmissionError,
    discover_submissions,
    load_submission,
    paper_links,
)
from .spec import MAX_FLAG_FPR, TRACKS

PAGE = REPO_ROOT / "docs" / "leaderboard" / "index.md"
DATA = REPO_ROOT / "docs" / "leaderboard" / "leaderboard.json"
START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"


def collect(root: Path | None = None) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {k: [] for k in TRACKS}
    for folder in discover_submissions(root or CONTRIB_DIR):
        try:
            sub = load_submission(folder)
        except SubmissionError:
            continue
        res = sub.load_results()
        if not res or res.get("suite") != "full" or res.get("seed_offset", 0) != 0:
            continue
        rows[sub.metadata["track"]].append(
            {
                "name": sub.name,
                "folder": folder.name,
                "meta": sub.metadata,
                "metrics": res["metrics"],
                "stale": res.get("benchmark_version") != TRACKS[sub.metadata["track"]].version,
                "benchmark_version": res.get("benchmark_version"),
            }
        )
    for key, items in rows.items():
        track = TRACKS[key]
        for r in items:
            r["flag_tier"] = flag_tier(track, r["metrics"])
        items.sort(key=lambda r, t=track: rank_key(t, r))
    return rows


def rank_key(track, r: dict) -> tuple:
    """Sort key for a board row: current before stale, flag group, primary metric, name."""
    return (
        r["stale"],
        r["flag_tier"],
        -_or_neg_inf(r["metrics"].get(track.primary)),
        r["name"],
    )


def _or_neg_inf(v) -> float:
    return float("-inf") if v is None else v


def flag_tier(track, metrics: dict) -> int:
    """Detection ranking groups: 0 flag holds FPR <= MAX_FLAG_FPR, 1 no flag, 2 flag fires too often."""
    if not track.detection:
        return 0
    fpr = metrics.get("fpr")
    if fpr is None:
        return 1
    return 0 if fpr <= MAX_FLAG_FPR else 2


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _text(s: str) -> str:
    """Submitter text made safe for a markdown table cell (and for link text)."""
    s = " ".join(str(s).split())
    for ch in "\\|[]":
        s = s.replace(ch, "\\" + ch)
    return s


def _url(u: str) -> str:
    return str(u).strip().replace(" ", "%20").replace("|", "%7C").replace(")", "%29")


def _method_cell(r: dict) -> str:
    meta = r["meta"]
    link = f"https://github.com/criticaldata/ShortKit-ML/tree/main/contrib/{r['folder']}"
    cell = f"[{_text(meta['title'])}]({link})"
    if meta.get("baseline"):
        cell += " <small>baseline</small>"
    if r["stale"]:
        cell += f" <small>stale: v{r['benchmark_version']}</small>"
    if r.get("flag_tier") == 2:
        cell += f" <small>flag FPR > {MAX_FLAG_FPR:g}</small>"
    return cell


def _paper_cell(meta: dict) -> str:
    links = paper_links(meta)
    if len(links) == 1:
        return f"[paper]({_url(links[0])})"
    return ", ".join(f"[{i}]({_url(u)})" for i, u in enumerate(links, 1))


def render_tables(rows: dict[str, list[dict]]) -> str:
    out = []
    for key, track in TRACKS.items():
        out.append(f"## {track.title}\n")
        rank = f"Ranked by **{dict(track.columns)[track.primary]}**"
        if track.detection:
            rank += (
                f", after grouping: entries whose flag fires on at most {MAX_FLAG_FPR:.0%} of null "
                f"datasets first, then entries without a flag, then entries whose flag fires more often"
            )
        out.append(
            f"{track.summary} {rank}. Benchmark v{track.version}. [Track rules](tracks.md#{key}).\n"
        )
        items = rows[key]
        if not items:
            out.append("_No entries yet: [be the first](submit.md)._\n")
            continue
        head = ["#", "Method", "Authors", "Paper"] + [h for _, h in track.columns]
        out.append("| " + " | ".join(head) + " |")
        out.append(
            "|" + "|".join(["---:", "---", "---", "---"] + ["---:"] * len(track.columns)) + "|"
        )
        for i, r in enumerate(items, 1):
            metrics = [_fmt(r["metrics"].get(k)) for k, _ in track.columns]
            if not r["stale"]:
                metrics[[k for k, _ in track.columns].index(track.primary)] = (
                    f"**{metrics[[k for k, _ in track.columns].index(track.primary)]}**"
                )
            cells = [
                str(i),
                _method_cell(r),
                ", ".join(_text(a) for a in r["meta"]["authors"]),
                _paper_cell(r["meta"]),
            ] + metrics
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def build(
    check: bool = False, root: Path | None = None, page_path: Path = PAGE, data_path: Path = DATA
) -> bool:
    """Rewrite the generated part of the page. With ``check``, return False if it is out of date."""
    rows = collect(root)
    tables = render_tables(rows)
    page = page_path.read_text()
    if START not in page or END not in page:
        raise RuntimeError(f"{page_path} lost its {START} / {END} markers")
    pre, rest = page.split(START, 1)
    _, post = rest.split(END, 1)
    new_page = f"{pre}{START}\n\n{tables}\n{END}{post}"
    data = {
        "max_flag_fpr": MAX_FLAG_FPR,
        "tracks": {
            k: {
                "benchmark_version": TRACKS[k].version,
                "entries": [
                    {
                        "rank": i,
                        "name": r["name"],
                        "title": r["meta"]["title"],
                        "authors": r["meta"]["authors"],
                        "papers": paper_links(r["meta"]),
                        "baseline": bool(r["meta"].get("baseline")),
                        "stale": r["stale"],
                        "flag_fpr_too_high": r["flag_tier"] == 2,
                        "metrics": r["metrics"],
                    }
                    for i, r in enumerate(v, 1)
                ],
            }
            for k, v in rows.items()
        },
    }
    new_data = json.dumps(data, indent=2) + "\n"
    if check:
        return page == new_page and data_path.exists() and data_path.read_text() == new_data
    page_path.write_text(new_page)
    data_path.write_text(new_data)
    return True
