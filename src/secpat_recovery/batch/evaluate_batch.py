#!/usr/bin/env python3
"""`secpat evaluate --batch-path`: score a recovery batch against the ground-truth snapshot.

Runs Levels 1–3 of the pattern-drift framework over every record in a batch
folder and writes one self-contained HTML report.

**Nothing scored here is implemented here.** The metrics and the AST index come
from ``secpat-scoring`` and the interaction-order walk from
``secpat-interaction-order``; ``secpat_scoring.evaluate`` is the single entry
point, called with one ``EvaluationRequest`` per record. ``build_index`` and
``detect`` are the same functions ``secpat recover`` builds its own stage 3
with, so a recovery is scored against an order derived exactly the way the one it
emitted was. Indexing the rows, resolving the role vocabulary, restricting both
sides to it and building the containment tree all happen inside that one call.
The per-level scores are reported as-is; the package's composite
(``overall_drift``, a mean across levels) is deliberately not shown — the levels
are conditioned on one another and measured on different units, so averaging
them, or averaging a level across records, denotes nothing. Every input is local:

    ground truth    <- <groundtruth-path>/<pattern>__<repo>.json     (the snapshot)
    recovery result <- <batch>/<record>/recovery.json
    vocabulary      <- security_patterns/<pattern>/04_pattern.json
    behaviour       <- security_patterns/<pattern>/04_pattern.json
    struct          <- secpat_scoring.workspace.build_index(codebase/<repo>)

**Roles flagged ``required: false`` are not scored** (no flag means required). An optional role a
codebase legitimately does not implement is not drift, and that is now the
scorer's own rule rather than a switch a caller sets — see
``secpat_scoring.evaluation.resolve_roles``.

**The ground truth is given, not chosen.** Each record's snapshot file holds one
role mapping, ``role_mappings``, and that is the ground truth. Which annotation
(or, failing one, which recovery) it is was decided when the snapshot was written
(by whoever prepared the snapshot); scoring neither knows nor asks.

**``--recovery-order stored``** scores the recovery side's Level 3 on the order
its own ``recovery.json`` carries under ``interaction_order`` instead of a walk.
It exists for the one-shot baseline, whose stored order is the model's claim
(``mechanism: "model-claim"``) — the question is how well that claim matches the
ground truth's walked order. The ground truth is still walked; the conditioning is
unchanged, so a claimed interaction counts only when both endpoints are roles both
sides detected. A claimed edge through a role that was not agreed is dropped, not
bridged: the walk can pass through such a role because it follows the code, while
a claim names no code between its endpoints. A role pair claimed twice keeps its
first rank.

Usage (paths relative to where you run it):
    secpat evaluate --manifest-path <manifest.json> --groundtruth-path <snapshot folder> \
        --batch-path <batch folder>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

# The engine's .env (JOERN_SERVICE_URL and the rest) is loaded by this package's
# __init__, before secpat_interaction_order snapshots its config.
from secpat_interaction_order import DetectionResult, Interaction, detect
from secpat_scoring import EvaluationRequest, evaluate
from secpat_scoring.workspace import build_index

from ..interaction.backends import detect_language
from ..manifest import load_records

# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


class Snapshot:
    """The ground-truth snapshot: one ``<pattern>__<repo>.json`` per record."""

    def __init__(self, root: Path):
        self.root = root

    def ground_truth(self, rec: dict) -> list[dict] | None:
        """The record's ground-truth role mapping, or None if it has none."""
        path = self.root / f"{rec['pattern']}__{rec['repo']}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("role_mappings")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def stored_order(envelope: dict | None, mappings: dict[str, dict]) -> DetectionResult:
    """A recovery's stored ``interaction_order``, narrowed to the roles in ``mappings``.

    ``mappings`` is the side Level 3 hands its detector — already conditioned to
    the roles both sides detected — so an interaction is kept only when both of
    its endpoints are among them. Ranks are kept as stored; a role pair stored
    twice keeps its first rank, since Kendall's Tau compares each interaction once.
    """
    envelope = envelope or {}
    if not envelope.get("available", False):
        return DetectionResult(
            reason=envelope.get("reason") or "the recovery stored no interaction order"
        )
    rows = sorted(envelope.get("interactions") or [], key=lambda r: r.get("rank", 0))
    kept, seen = [], set()
    for row in rows:
        pair = (row.get("from", ""), row.get("to", ""))
        if pair in seen or pair[0] not in mappings or pair[1] not in mappings:
            continue
        seen.add(pair)
        kept.append(Interaction(
            source=pair[0],
            target=pair[1],
            file=row.get("file", ""),
            line=int(row.get("line") or 0),
            mechanism=row.get("mechanism") or "stored",
        ))
    return DetectionResult(
        interactions=kept,
        entry_role=kept[0].source if kept else "",
        entry_rule="stored",
    )


def score_record(
    rec: dict, batch_dir: Path, snapshot: Snapshot, recovery_order: str = "derived",
) -> dict:
    """Score one record against its snapshot ground truth.

    Returns a report dict; never raises for data reasons.
    """
    slug = f"{rec['pattern']}__{rec['repo']}"
    out: dict = {
        "slug": slug,
        "id": rec.get("id"),
        "pattern": rec["pattern"],
        "repo": rec["repo"],
        "error": None,
        "metrics": [],
    }

    recovery_path = batch_dir / slug / "recovery.json"
    if not recovery_path.is_file():
        out["error"] = f"no recovery.json in {slug}/ — the run did not produce one"
        return out

    ground_truth_rows = snapshot.ground_truth(rec)
    if ground_truth_rows is None:
        out["error"] = f"no ground truth for {slug} in the snapshot"
        return out

    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery_rows = recovery.get("role_mappings", [])

    # The pattern comes from its local file: it is
    # the definition the recovery actually ran against, and the only one carrying
    # `required`. Its `behaviour` is read from the same document for the same
    # reason — pairing a role set from one revision with a walkthrough from
    # another is incoherent.
    pattern_path = Path(rec["pattern_file"])
    if not pattern_path.is_file():
        out["error"] = f"pattern file missing: {rec['pattern_file']}"
        return out
    pattern = json.loads(pattern_path.read_text(encoding="utf-8"))

    # The language is read off both sides' citations, as stage 2 reads it off
    # its own. It picks both the files indexed and the walk's frontend: a Java
    # repo indexed as Python holds no Java symbol for Level 2 to place, and
    # walked under the Python profile it anchors nothing for Level 3.
    language = detect_language(
        Path(rec["codebase_path"]),
        [row.get("file", "") for row in ground_truth_rows + recovery_rows],
    )
    struct = build_index(Path(rec["codebase_path"]), language)
    if not struct:
        # An empty index is never legitimate for these repos. It means the
        # grammar failed to load, which would degrade Level 2
        # to string identity while still reporting plausible-looking numbers.
        # Refuse rather than mislead.
        out["error"] = (
            f"AST index came back empty for {rec['codebase_path']} — the parser "
            f"failed. Check that secpat-scoring's grammar packages installed."
        )
        return out

    # `repo` is the name the query service resolves its own checkout by, so this
    # scores against an order derived from the same graph the recovery run used.
    # A callable, not a value: Level 3 derives each side from the mapping it has
    # conditioned to the roles both sides detected, and the walkthrough the scorer
    # hands in without the steps touching a `required: false` role.
    #
    # With `--recovery-order stored` the recovery side reads its stored order
    # instead. Level 3 calls this once per side; `evaluate` passes the row dicts
    # through unchanged (no `normalise_row` here), so the recovery side is the
    # call whose rows are the very objects read from recovery.json.
    recovery_row_ids = {id(row) for row in recovery_rows}

    def detect_order(mappings, behaviour):
        if recovery_order == "stored" and mappings and all(
            id(row) in recovery_row_ids for row in mappings.values()
        ):
            return stored_order(recovery.get("interaction_order"), mappings)
        return detect(
            rec["repo"], language, mappings, behaviour=behaviour,
            repo_path=Path(rec["codebase_path"]),
        )

    # The one scoring call. Every consumer of secpat-scoring makes this same call,
    # so there is no switch that could make two of them disagree.
    report = evaluate(
        EvaluationRequest(
            ground_truth_rows=ground_truth_rows,
            recovery_result_rows=recovery_rows,
            pattern=pattern,
            struct=struct,
            detect_order=detect_order,
        )
    )
    if not report.roles.names:
        out["error"] = f"no scorable roles in {rec['pattern_file']}"
        return out

    scored = report.to_dict()
    out.update({
        "metrics": scored["metrics"],
        "vocabulary": scored["roleVocabulary"],
        "role_scope": scored["roleScope"],
        "excluded_roles": scored["excludedRoles"],
        "dropped_ground_truth_rows": scored["droppedGroundTruthRows"],
        "dropped_recovery_rows": scored["droppedRecoveryResultRows"],
        "gt_roles": scored["groundTruthRoles"],
        "rr_roles": scored["recoveryResultRoles"],
        "struct_files": len(struct),
        "recovery_order": recovery_order,
    })
    return out


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light dark;
  --bg:#fff; --fg:#222; --muted:#777; --line:#e0e0e0;
  --panel:#f5f7f8; --chip:#eceff1; --chipfg:#37474f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#16191c; --fg:#e6e6e6; --muted:#9aa4ad; --line:#2c3238;
    --panel:#1d2226; --chip:#252b31; --chipfg:#c3ccd4;
  }
}
:root[data-theme="dark"] {
  --bg:#16191c; --fg:#e6e6e6; --muted:#9aa4ad; --line:#2c3238;
  --panel:#1d2226; --chip:#252b31; --chipfg:#c3ccd4;
}
:root[data-theme="light"] {
  --bg:#fff; --fg:#222; --muted:#777; --line:#e0e0e0;
  --panel:#f5f7f8; --chip:#eceff1; --chipfg:#37474f;
}
* { box-sizing: border-box; }
body {
  margin:0; padding:32px 24px 64px; background:var(--bg); color:var(--fg);
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 4px; }
h2 { font-size:1.05rem; margin:0; }
.sub { color:var(--muted); font-size:.82rem; margin-bottom:24px; }
.banner {
  background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:12px 16px; font-size:.8rem; line-height:1.7; margin-bottom:28px;
}
.rec { border:1px solid var(--line); border-radius:8px; margin-bottom:22px; overflow:hidden; }
.rec > header {
  display:flex; align-items:center; gap:14px; padding:12px 16px;
  background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap;
}
.rec .meta { color:var(--muted); font-size:.74rem; flex:1 1 100%; }
.body { padding:6px 16px 14px; }
details { border-bottom:1px solid var(--line); padding:8px 0; }
details:last-child { border-bottom:0; }
summary {
  cursor:pointer; display:flex; align-items:center; gap:10px;
  font-size:.85rem; font-weight:600; list-style:none;
}
summary::-webkit-details-marker { display:none; }
summary::before { content:"▸"; color:var(--muted); font-size:.7rem; }
details[open] > summary::before { content:"▾"; }
.formula { color:var(--muted); font-weight:400; font-size:.72rem; }
.headline { margin-left:auto; font-weight:700; font-size:.9rem; }
.na { margin-left:auto; font-size:.72rem; color:var(--muted); font-style:italic; }
.inner { padding:10px 0 4px 20px; }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.76rem; margin-top:6px; }
th {
  text-align:left; padding:6px 10px; border-bottom:1px solid var(--line);
  color:var(--muted); font-weight:600; white-space:nowrap;
}
td { padding:6px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
td.mono, .loc { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.loc { color:var(--muted); font-size:.68rem; }
.badge {
  display:inline-block; padding:1px 8px; border-radius:10px;
  font-size:.68rem; font-weight:600;
}
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.chip {
  background:var(--chip); color:var(--chipfg); font-size:.7rem;
  padding:2px 9px; border-radius:10px; white-space:nowrap;
}
.err {
  background:#fbe9e7; border:1px solid #ffab91; color:#bf360c;
  border-radius:6px; padding:10px 14px; font-size:.8rem; margin:10px 16px 14px;
}
@media (prefers-color-scheme: dark) {
  .err { background:#3a1f16; border-color:#7a3b26; color:#ffab91; }
}
.flow { display:grid; grid-template-columns:1fr 1fr; gap:0 16px; font-size:.74rem; margin-top:8px; }
.flow .h {
  color:var(--muted); font-weight:600; padding-bottom:4px;
  border-bottom:1px solid var(--line);
}
.lane { padding:3px 0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.lane.blank { color:var(--muted); opacity:.4; }
"""

STATUS_COLORS = {
    "present": ("#e3f2fd", "#1565c0"),
    "missing": ("#eceff1", "#90a4ae"),
    "insecure": ("#fbe9e7", "#e65100"),
    "external": ("#f3e5f5", "#6a1b9a"),
}
OUTCOME_COLORS = {"agreed": "#2e7d32", "mismatched": "#c62828"}


def comfort(score: float | None) -> str:
    """Green at full agreement through amber to red — the panel's own scale."""
    if score is None:
        return "#90a4ae"
    return "#2e7d32" if score >= 0.9 else "#f57f17" if score >= 0.7 else "#c62828"


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return "—" if v is None else str(v)


def e(v) -> str:
    return escape("" if v is None else str(v))


def status_badge(status) -> str:
    s = str(status or "—")
    bg, fg = STATUS_COLORS.get(s, ("#eceff1", "#546e7a"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{e(s)}</span>'


def counts_label(m: dict) -> str | None:
    """"12 / 14 (85.7%)" when the metric reports an n-of-m count.

    Level 1 counts roles, and its headline *is* that proportion, so the raw
    counts sit beside the percentage. Level 3's headline is a product of two
    factors, not a count, so it gets the percentage alone.
    """
    d = m.get("detail") or {}
    for n, total in ((d.get("agreedRoles"), d.get("totalRoles")),):
        if isinstance(n, int) and isinstance(total, int) and total > 0:
            return f"{n} / {total} ({pct(m.get('score'))})"
    return None


def table(headers: list[str], rows: list[str]) -> str:
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def level1_table(rows: list[dict]) -> str:
    out = []
    for r in rows:
        outcome = str(r.get("outcome", ""))
        color = OUTCOME_COLORS.get(outcome, "inherit")
        out.append(
            f"<tr><td><strong>{e(r.get('patternRole'))}</strong></td>"
            f"<td>{status_badge(r.get('groundTruth'))}</td>"
            f"<td>{status_badge(r.get('recoveryResult'))}</td>"
            f'<td style="color:{color};font-weight:600">{e(outcome)}</td></tr>'
        )
    return table(["Role", "Ground truth", "Recovery result", "Outcome"], out)


def level2_table(rows: list[dict]) -> str:
    out = []
    for r in rows:
        gt = r.get("groundTruth") or {}
        rr = r.get("recoveryResult") or {}
        score = r.get("score") if isinstance(r.get("score"), (int, float)) else None
        out.append(
            f"<tr><td><strong>{e(r.get('patternRole'))}</strong></td>"
            f'<td class="mono">{e(gt.get("entity") or "—")}'
            f'<div class="loc">{e(gt.get("loc") or "—")}</div></td>'
            f'<td class="mono">{e(rr.get("entity") or "—")}'
            f'<div class="loc">{e(rr.get("loc") or "—")}</div></td>'
            f"<td>{'—' if r.get('distance') is None else num(r.get('distance'))}</td>"
            f'<td style="font-weight:700;color:{comfort(score)}">'
            f"{'—' if score is None else f'{score:.3f}'}</td></tr>"
        )
    return table(
        ["Role", "Ground truth", "Recovery result", "d", "Score"], out
    )


def align_flows(gt_seq: list[dict], rr_seq: list[dict]) -> list[tuple]:
    """LCS alignment of the two orderings — the panel's own `alignFlows`."""
    n, m = len(gt_seq), len(rr_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = (
                dp[i + 1][j + 1] + 1
                if gt_seq[i]["interaction"] == rr_seq[j]["interaction"]
                else max(dp[i + 1][j], dp[i][j + 1])
            )
    lanes, i, j = [], 0, 0
    while i < n and j < m:
        if gt_seq[i]["interaction"] == rr_seq[j]["interaction"]:
            lanes.append((gt_seq[i], rr_seq[j]))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            lanes.append((gt_seq[i], None))
            i += 1
        else:
            lanes.append((None, rr_seq[j]))
            j += 1
    lanes += [(gt_seq[k], None) for k in range(i, n)]
    lanes += [(None, rr_seq[k]) for k in range(j, m)]
    return lanes


def level3_view(rows: list[dict]) -> str:
    """The two flows side by side, aligned — what the panel draws for Level 3."""
    gt_seq = sorted(
        (r for r in rows if r.get("groundTruthRank") is not None),
        key=lambda r: r["groundTruthRank"],
    )
    rr_seq = sorted(
        (r for r in rows if r.get("recoveryResultRank") is not None),
        key=lambda r: r["recoveryResultRank"],
    )
    if not gt_seq and not rr_seq:
        return ""

    lanes = align_flows(gt_seq, rr_seq)
    left = ['<div class="h">Ground truth</div>']
    right = ['<div class="h">Recovery result</div>']
    for gt, rr in lanes:
        both = gt is not None and rr is not None
        color = "#37474f" if both else "#c62828" if gt else "#6a1b9a"
        left.append(
            f'<div class="lane" style="color:{color}">{e(gt["interaction"])}</div>'
            if gt
            else '<div class="lane blank">·</div>'
        )
        right.append(
            f'<div class="lane" style="color:{color}">{e(rr["interaction"])}</div>'
            if rr
            else '<div class="lane blank">·</div>'
        )
    grid = (
        '<div class="flow"><div>' + "".join(left) + "</div><div>"
        + "".join(right) + "</div></div>"
    )

    body = []
    for r in rows:
        body.append(
            f'<tr><td class="mono">{e(r.get("interaction"))}</td>'
            f"<td>{e(r.get('groundTruthRank'))}</td>"
            f"<td>{e(r.get('recoveryResultRank'))}</td>"
            f"<td>{e(r.get('outcome'))}</td>"
            f"<td>{e(r.get('discordantWith'))}</td></tr>"
        )
    return grid + table(
        ["Interaction", "GT rank", "RR rank", "Outcome", "Discordant with"], body
    )


def chips(detail: dict) -> str:
    if not detail:
        return ""
    items = "".join(
        f'<span class="chip">{e(k)}: <strong>{num(v)}</strong></span>'
        for k, v in detail.items()
        if not isinstance(v, (list, dict))
    )
    return f'<div class="chips">{items}</div>' if items else ""


def metric_block(m: dict) -> str:
    available = m.get("available")
    score = m.get("score")
    headline = (
        f'<span class="headline" style="color:{comfort(score)}">'
        f"{e(m.get('headlineLabel'))} {e(counts_label(m) or pct(score))}</span>"
        if available
        else '<span class="na">not available</span>'
    )
    inner = [chips(m.get("detail") or {})]
    rows = m.get("perRole") or []
    if available and rows:
        key = m.get("key")
        inner.append(
            level1_table(rows)
            if key == "level1"
            else level2_table(rows)
            if key == "level2"
            else level3_view(rows)
            if key == "level3"
            else ""
        )
    return (
        f"<details{' open' if available else ''}><summary>{e(m.get('level'))}"
        f'<span class="formula">{e(m.get("formula"))}</span>{headline}</summary>'
        f'<div class="inner">{"".join(inner)}</div></details>'
    )


def record_block(r: dict) -> str:
    if r["error"]:
        return (
            f'<section class="rec"><header><h2>{e(r["slug"])}</h2></header>'
            f'<div class="err">{e(r["error"])}</div></section>'
        )
    excluded = r.get("excluded_roles") or []
    # A row for an optional role is dropped too, but it is already named on the
    # "optional, not scored" line — repeating it under "no longer defined" would
    # claim the pattern forgot a role it deliberately marked optional.
    dropped = sorted(
        (
            set(r.get("dropped_ground_truth_rows") or [])
            | set(r.get("dropped_recovery_rows") or [])
        )
        - set(excluded)
    )
    scope = "required roles only" if r.get("role_scope") == "required" else "all roles"
    meta = (
        f"pattern <strong>{e(r['pattern'])}</strong> · repo <strong>{e(r['repo'])}</strong> · "
        f"scoring {len(r['vocabulary'])} roles ({scope}) · "
        f"ground truth {r['gt_roles']} rows · "
        f"recovery result {r['rr_roles']} rows · "
        f"AST index {r['struct_files']} files"
    )
    if excluded:
        meta += (
            f'<div style="margin-top:3px">optional, not scored: '
            f"<em>{e(', '.join(excluded))}</em></div>"
        )
    if dropped:
        meta += (
            f'<div style="margin-top:3px">rows naming a role this pattern no longer '
            f"defines, dropped: <em>{e(', '.join(dropped))}</em></div>"
        )
    return (
        f'<section class="rec"><header><h2>{e(r["slug"])}</h2>'
        f'<div class="meta">{meta}</div></header>'
        f'<div class="body">{"".join(metric_block(m) for m in r["metrics"])}</div></section>'
    )


def render_html(reports: list[dict], batch: str, snapshot: str,
                recovery_order: str = "derived") -> str:
    order_note = (
        '<div style="margin-top:6px"><strong>Level 3 recovery order is the stored '
        "one</strong>: the recovery side&rsquo;s interactions are read from its own "
        "<code>recovery.json</code> <code>interaction_order</code> (for a baseline, the "
        "model&rsquo;s claim), not walked. The ground truth&rsquo;s order is still walked "
        "on Joern.</div>"
        if recovery_order == "stored" else ""
    )
    summary_rows = []
    for r in reports:
        if r["error"]:
            summary_rows.append(
                f'<tr><td class="mono">{e(r["slug"])}</td>'
                f'<td colspan="4" style="color:#c62828">{e(r["error"])}</td></tr>'
            )
            continue
        cells = []
        for key in ("level1", "level2", "level3"):
            m = next((x for x in r["metrics"] if x["key"] == key), None)
            s = m.get("score") if m and m.get("available") else None
            cells.append(
                f'<td style="color:{comfort(s)};font-weight:600">{pct(s)}</td>'
            )
        summary_rows.append(
            f'<tr><td class="mono">{e(r["slug"])}</td>{"".join(cells)}</tr>'
        )

    return f"""<title>Pattern drift — {e(batch)}</title>
<style>{CSS}</style>
<div class="wrap">
<h1>Pattern drift evaluation</h1>
<div class="sub">batch <code>{e(batch)}</code> · {len(reports)} record(s) ·
generated {datetime.now().isoformat(timespec="seconds")}</div>

<div class="banner">
<div><strong>Ground truth</strong> is each record's role mapping in the snapshot
<code>{e(snapshot)}</code>; the <strong>recovery result</strong> is what the pipeline
produced in this batch.
Every score is <code>1</code> at perfect agreement, and <code>drift = 1 − score</code>.</div>
{order_note}
<div style="margin-top:6px;color:var(--muted)">
Levels 1–3 are computed by the <code>secpat-scoring</code> package — not
reimplemented here.</div>
<div style="margin-top:6px;color:var(--muted)">The three levels are reported
separately and never combined into one number: they are conditioned on one another
(Level 2 scores only Level 1 successes, Level 3 only interactions whose endpoints
passed Level 2) and are measured on different units — a set-overlap score, a
structural-distance score, and a rank correlation — so averaging them across levels
or across records would not denote anything. Level 3 is <em>overlap &times; order</em>:
overlap is Jaccard over the two sides&rsquo; interaction sets, and order is Kendall&rsquo;s
Tau over the interactions they <em>both</em> found (Fagin et al.&rsquo;s induced
sub-order) &mdash; 1 when exactly one interaction is shared, 0 when none is.</div>
<div style="margin-top:6px;color:var(--muted)">Roles the pattern flags
<code>required: false</code> are not scored (no flag means required) — an optional
role a codebase legitimately does not implement is not drift. Each record lists what it
excluded. The flag lives on the
local <code>security_patterns/&lt;pattern&gt;/04_pattern.json</code>, so the role
vocabulary and behaviour come from there.</div>
</div>

<h2 style="margin-bottom:6px">Summary</h2>
{table(["Record", "L1 detection", "L2 grounding", "L3 interaction order"], summary_rows)}

{"".join(record_block(r) for r in reports)}
</div>"""


# ---------------------------------------------------------------------------


def run(args) -> int:
    """Score a batch. ``args`` carries ``secpat evaluate``'s batch options by name:
    ``manifest_path``, ``groundtruth_path``, ``batch_path``, ``recovery_order``,
    ``out``."""
    batch_dir = Path(args.batch_path)
    if not batch_dir.is_dir():
        print(f"error: no such batch: {batch_dir}", file=sys.stderr)
        return 2
    snapshot_dir = Path(args.groundtruth_path)
    if not snapshot_dir.is_dir():
        print(f"error: no ground-truth snapshot at {snapshot_dir}", file=sys.stderr)
        return 2

    records = load_records(Path(args.manifest_path))
    snapshot = Snapshot(snapshot_dir)

    reports = []
    for n, rec in enumerate(records, 1):
        print(f"[{n}/{len(records)}] {rec['pattern']}__{rec['repo']}", flush=True)
        report = score_record(rec, batch_dir, snapshot, args.recovery_order)
        if report["error"]:
            print(f"    skipped: {report['error']}", flush=True)
        else:
            parts = " ".join(
                "{}={}".format(
                    m["key"],
                    "n/a" if not m["available"] else f"{m['score']:.3f}",
                )
                for m in report["metrics"]
            )
            print(f"    {parts}", flush=True)
        reports.append(report)

    # The report sits *inside* the batch folder, beside the per-record results it
    # scores. It once sat next to it, so the run folder would open on the readable
    # artifact rather than on eight subfolders; the cost was that reports from
    # every batch piled up in one directory, named only by timestamp, and telling
    # which run produced which report meant opening it. A report belongs with its
    # inputs. The timestamp postfix still keeps successive scoring runs of the
    # *same* batch — a rescore after a metric change — from overwriting each
    # other; pass --out for a fixed name.
    report_dir = batch_dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (
        Path(args.out)
        if args.out
        else report_dir / f"evaluation_{stamp}.html"
    )
    json_path = out_path.with_suffix(".json")
    out_path.write_text(
        render_html(reports, batch_dir.name, str(snapshot_dir), args.recovery_order),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(reports, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nWritten: {out_path}")
    print(f"Written: {json_path}")

    # A record that could not be scored is a defect to fix, not a footnote: exit
    # non-zero so a scripted run stops on it instead of publishing a table with
    # holes in it.
    failed = [r["slug"] for r in reports if r["error"]]
    if failed:
        print(f"\n{len(failed)} of {len(reports)} record(s) failed:", file=sys.stderr)
        for slug in failed:
            print(f"  {slug}", file=sys.stderr)
        return 1
    return 0

