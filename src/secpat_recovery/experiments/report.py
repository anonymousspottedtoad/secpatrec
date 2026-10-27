"""Summarise the output folder's scores in <output>/index.html.

Every batch folder <output>/<mode>/<experiment>/<model>/<language>/ with an
evaluation_<stamp>.json is reported on each record's newest score across its
evaluations; its status comes from the last run in <output>/runs.json that ran
it. The experiments rebuild the page after every batch; `secpat report` rebuilds
it by hand.
"""

import html
import json
from collections import defaultdict
from pathlib import Path

MODES = ["evaluate", "interaction", "recover"]
EXPERIMENTS = ["pipeline", "baseline", "ablation_no_grep", "ablation_no_search"]
LEVELS = [("level1", "Level 1 — roles"), ("level2", "Level 2 — structure"),
          ("level3", "Level 3 — interactions")]
QUIET = ("ok", "dry run", "no matching records")

STYLE = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de; --head: #f6f8fa;
  --bad: #cf222e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8d96a0; --line: #30363d; --head: #161b22;
    --bad: #f85149;
  }
}
body {
  background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, sans-serif;
  margin: 24px 16px;
}
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 8px; }
p, .meta { color: var(--muted); } .meta { margin: 0 0 16px; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; min-width: 640px; }
th, td {
  border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left; white-space: nowrap;
}
th { background: var(--head); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.n { color: var(--muted); font-size: 12px; } .bad { color: var(--bad); }
a { color: inherit; }
"""


def latest_scores(batch: Path) -> tuple[list[dict], Path | None]:
    """Each record's newest score across the batch's evaluations, and the newest file.

    Runs filtered by --repo / --pattern score part of a batch each, so no single
    evaluation need cover every record in the folder.
    """
    found = sorted(batch.glob("evaluation_*.json"))
    records = {}
    for evaluation in found:
        for r in json.loads(evaluation.read_text(encoding="utf-8")):
            records[r.get("slug") or f"{r.get('pattern')}__{r.get('repo')}"] = r
    return list(records.values()), (found[-1] if found else None)


def scores(records: list[dict]) -> dict:
    """Mean of each level over the records that level could score."""
    out = {"records": len(records)}
    for key, _ in LEVELS:
        values = [
            m["score"] for r in records for m in r.get("metrics") or []
            if m.get("key") == key and m.get("available", True) and m.get("score") is not None
        ]
        out[key] = (sum(values) / len(values), len(values)) if values else None
    return out


def cells(s: dict | None) -> str:
    if s is None:
        return '<td class="num">—</td>' * (1 + len(LEVELS))
    row = f'<td class="num">{s["records"]}</td>'
    for key, _ in LEVELS:
        v = s[key]
        row += ('<td class="num">—</td>' if v is None else
                f'<td class="num">{v[0]:.3f} <span class="n">({v[1]})</span></td>')
    return row


def sort_key(batch: tuple) -> tuple:
    mode, experiment, model, lang = batch
    rank = lambda v, order: order.index(v) if v in order else len(order)  # noqa: E731
    return (rank(mode, MODES), mode, rank(experiment, EXPERIMENTS), experiment, model, lang)


def build(output: Path) -> Path:
    output = Path(output)
    history_path = output / "runs.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    status = {}
    for run in history:
        for b in run["batches"]:
            status[(run["mode"], b["experiment"], b["model"], b["language"])] = b["status"]
    scored = {tuple(p.parent.relative_to(output).parts)
              for p in output.glob("*/*/*/*/evaluation_*.json")}
    latest = {(history[-1]["mode"], b["experiment"], b["model"], b["language"])
              for b in history[-1]["batches"]} if history else set()

    by_batch, by_pattern = [], []
    for key in sorted(scored | latest, key=sort_key):
        label = "".join(f"<td>{html.escape(v)}</td>" for v in key)
        st = status.get(key, "—")
        cls = "" if st in QUIET + ("—",) else ' class="bad"'
        status_td = f"<td{cls}>{html.escape(st)}</td>"
        records, evaluation = latest_scores(output.joinpath(*key))
        if evaluation is None:
            by_batch.append(f"<tr>{label}{status_td}{cells(None)}<td></td></tr>")
            continue
        page = evaluation.with_suffix(".html")
        href = html.escape(str(page.relative_to(output)))
        link = f'<a href="{href}">{html.escape(page.name)}</a>' if page.exists() else ""
        by_batch.append(f"<tr>{label}{status_td}{cells(scores(records))}<td>{link}</td></tr>")
        per_pattern = defaultdict(list)
        for r in records:
            per_pattern[r.get("pattern", "?")].append(r)
        for pattern, rs in sorted(per_pattern.items()):
            by_pattern.append(f"<tr>{label}<td>{html.escape(pattern)}</td>{cells(scores(rs))}</tr>")

    if history:
        last = history[-1]
        filters = ", ".join(
            f"{k} {', '.join(last[k])}" for k in ("repos", "patterns") if last.get(k)
        )
        meta = (f"{len(history)} run(s) · last: {html.escape(last['mode'])} mode, started "
                f"{html.escape(last['started_at'])}, CPGs {html.escape(last['cpg'])}"
                + (f", {html.escape(filters)}" if filters else "")
                + (", dry run" if last.get("dry_run") else ""))
    else:
        meta = "no runs yet"
    level_heads = "".join(f"<th>{name}</th>" for _, name in LEVELS)
    heads = "<th>Mode</th><th>Experiment</th><th>Model</th><th>Language</th>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>secpat results</title><style>{STYLE}</style></head>
<body>
<h1>secpat results</h1>
<div class="meta">{meta}</div>
<p>Each batch shows every record's newest score; the linked secpat report is the
batch's newest evaluation, which may cover only the records that run selected.
Each level is the mean over the records it could score; the count is in
brackets. A fresh score leaves the baseline's unrun records (prompt over the
input limit) unscored, where the paper counts them as 0.</p>
<h2>By batch</h2>
<div class="wrap"><table>
<tr>{heads}<th>Status</th><th>Records</th>{level_heads}<th>secpat report</th></tr>
{''.join(by_batch)}
</table></div>
<h2>By pattern</h2>
<div class="wrap"><table>
<tr>{heads}<th>Pattern</th><th>Records</th>{level_heads}</tr>
{''.join(by_pattern)}
</table></div>
</body></html>
"""
    out = output / "index.html"
    out.write_text(page, encoding="utf-8")
    return out

