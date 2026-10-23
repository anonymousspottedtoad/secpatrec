#!/usr/bin/env python3
"""Compare the arms of a tool ablation, arm by arm and record by record.

Reads every batch folder belonging to one experiment — ``batch-ablation-<tag>-*``,
or the folders named with ``--batch`` — and puts their numbers side by side:

* **Scores.** Each arm's Level 1/2/3 means over the records that scored, from
  the newest ``evaluation_*.json`` stage 3 wrote inside the batch. A mean
  is taken only over records where that metric was *available*, and how many that
  was is printed beside it: Level 3 is unavailable where fewer than two roles were
  detected by both sides or no interaction was derived, and a mean over three
  records is not the same evidence as a mean over ten.
* **Behaviour.** How the agent worked without the tool — steps per run and calls
  per tool, counted from each record's ``run.log`` (the ``[Agent] Step N action:``
  lines the loop prints). This is the half of an ablation that the score cannot
  show: whether losing ``grep`` made the agent read more files, search more, or
  simply give up earlier.

The control arm (``full``, or ``--control <arm>``) is subtracted from the others
when one is present, so each arm reports its own numbers *and* its delta. The
comparison is deliberately aggregate: per-record score tables belong in the
batch's own ``evaluation_*.json``, which is where a reader who wants one
record's story should go.

Usage (``--output-root`` is required):
    secpat ablation report --output-root <root> --tag 20260823-171500
    secpat ablation report --output-root <root> --batch a --batch b
    secpat ablation report --output-root <root> --tag pilot --control no-grep
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

#: The metrics, in report order, keyed as `secpat_scoring` emits them.
METRICS: list[tuple[str, str]] = [
    ("level1", "L1 detection"),
    ("level2", "L2 grounding"),
    ("level3", "L3 interaction order"),
]

_ACTION = re.compile(r"^\[Agent\] Step (\d+) action: (\w+)\(")

#: Arm names run.sh writes, longest first. A batch folder is
#: ``batch-ablation-<tag>-<arm>`` and both halves contain hyphens, so the tag can
#: only be recovered by stripping a *known* arm suffix — splitting on the last
#: hyphen would read "no-grep" as a tag ending in "-no".
ARM_SUFFIXES: list[str] = ["no-search", "no-grep", "no-read", "no-list", "full"]


def split_batch_name(name: str) -> tuple[str, str] | None:
    """``batch-ablation-<tag>-<arm>`` → ``(tag, arm)``; None if it is not one."""
    if not name.startswith("batch-ablation-"):
        return None
    rest = name[len("batch-ablation-"):]
    for arm in ARM_SUFFIXES:
        if rest.endswith(f"-{arm}"):
            return rest[: -len(arm) - 1], arm
    return rest, ""


# ---------------------------------------------------------------------------
# reading one arm
# ---------------------------------------------------------------------------


#: Withheld tool -> the arm name run.sh gives it.
TOOL_ARMS: dict[str, str] = {
    "grep": "no-grep",
    "search": "no-search",
    "read_file": "no-read",
    "list_dir": "no-list",
}


def arm_name(batch_dir: Path, tag: str | None, config: dict | None = None) -> str:
    """The arm's short name.

    Read off ``batch_config.json`` when the batch has one: a control is usually
    named for its model (``<model>``) and its arms ``ablation-<arm>-<model>``,
    so the folder name alone does not say which is which — the disabled tools do.
    No tool withheld is the ``full`` arm. Older batches without a config fall back
    to the folder name.
    """
    if config and "disabled_tools" in config:
        disabled = config["disabled_tools"]
        return "+".join(TOOL_ARMS.get(t, f"no-{t}") for t in disabled) or "full"
    parsed = split_batch_name(batch_dir.name)
    if parsed and parsed[1]:
        return parsed[1]
    name = batch_dir.name
    prefix = f"batch-ablation-{tag}-" if tag else ""
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def newest_evaluation(batch_dir: Path) -> Path | None:
    """The most recent stage-3 report in this batch.

    Stage 3 timestamps every report rather than overwriting, so a rescore after a
    metric change leaves both on disk; the newest is the one whose metrics match
    the other arms', which is the only version an arm-to-arm comparison may use.
    """
    # By the timestamp in the name, which a report may carry after a prefix, so
    # the name alone does not sort by age.
    def stamp(p: Path) -> tuple[str, str]:
        m = re.search(r"\d{8}-\d{6}", p.name)
        return (m.group(0) if m else "", p.name)

    reports = sorted(batch_dir.glob("evaluation_*.json"), key=stamp)
    return reports[-1] if reports else None


def tool_usage(batch_dir: Path) -> dict:
    """Per-record step and tool-call counts, read from the runs' own logs.

    The log is the record here rather than transcript.md because it is written
    line by line as the run goes: a run that died at step 40 still has 39 actions
    in it, and an ablation is exactly the setting where runs die.
    """
    per_record: dict[str, dict] = {}
    for log_path in sorted(batch_dir.glob("*/run.log")):
        calls: dict[str, int] = {}
        steps = 0
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if m := _ACTION.match(line):
                steps = max(steps, int(m.group(1)))
                calls[m.group(2)] = calls.get(m.group(2), 0) + 1
        submitted = (log_path.parent / "recovery.json").is_file()
        per_record[log_path.parent.name] = {
            # The final submit_mapping is a step too — the last Thought → Action —
            # but it has no observation, so it prints no "action:" line.
            "steps": steps + 1 if steps else 0,
            "calls": calls,
            "submitted": submitted,
        }
    return per_record


def read_arm(batch_dir: Path, tag: str | None) -> dict:
    """Everything one arm contributes to the comparison."""
    config_path = batch_dir / "batch_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

    report_path = newest_evaluation(batch_dir)
    records = json.loads(report_path.read_text(encoding="utf-8")) if report_path else []

    scores: dict[str, dict[str, float]] = {key: {} for key, _ in METRICS}
    errors: list[str] = []
    for rec in records:
        if rec.get("error"):
            errors.append(f"{rec['slug']}: {rec['error']}")
            continue
        for metric in rec.get("metrics", []):
            if metric.get("available") and metric.get("score") is not None:
                scores.setdefault(metric["key"], {})[rec["slug"]] = float(metric["score"])

    usage = tool_usage(batch_dir)
    runs = len(usage)
    completed = sum(1 for u in usage.values() if u["submitted"])
    calls: dict[str, int] = {}
    for u in usage.values():
        for name, n in u["calls"].items():
            calls[name] = calls.get(name, 0) + n

    return {
        "arm": arm_name(batch_dir, tag, config),
        "batch": batch_dir.name,
        "disabled_tools": config.get("disabled_tools", []),
        "provider": config.get("provider"),
        "model": config.get("model"),
        "report": report_path.name if report_path else None,
        "scores": scores,
        "errors": errors,
        "runs": runs,
        "completed_runs": completed,
        "mean_steps": round(sum(u["steps"] for u in usage.values()) / runs, 1) if runs else None,
        "median_steps": (
            statistics.median([u["steps"] for u in usage.values()]) if runs else None
        ),
        "max_steps_seen": max((u["steps"] for u in usage.values()), default=None),
        "total_steps": sum(u["steps"] for u in usage.values()),
        "calls": calls,
        "per_record_usage": usage,
    }


def mean(values: dict[str, float]) -> float | None:
    return sum(values.values()) / len(values) if values else None


def paired_delta(arm: dict[str, float], control: dict[str, float]) -> tuple[float | None, int]:
    """Arm minus control over the records **both** scored, and how many those are.

    Not the difference of the two full means: an arm whose hardest record failed
    to score would otherwise show an improvement built entirely out of the record
    it lost. Pairing on the shared records makes the delta a comparison of the
    same repos under two tool sets, which is the thing the ablation is asking.
    """
    shared = sorted(set(arm) & set(control))
    if not shared:
        return None, 0
    diff = sum(arm[s] - control[s] for s in shared) / len(shared)
    return diff, len(shared)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def fmt(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"  ({value:+.3f})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A plain fixed-width table — read in a terminal, pasted into a paper."""
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) for row in rows)
    return f"{line}\n{rule}\n{body}"


def render(arms: list[dict], control: dict | None, tag: str) -> str:
    out: list[str] = []
    out.append(f"# Tool ablation — {tag}")
    out.append("")
    out.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}.")
    out.append("")

    # --- what each arm was ---
    out.append("## Arms")
    out.append("")
    out.append("```")
    out.append(table(
        ["arm", "disabled", "provider", "model", "runs", "completed", "batch"],
        [[
            a["arm"],
            ", ".join(a["disabled_tools"]) or "— (control)",
            a["provider"] or "?",
            a["model"] or "default",
            str(a["runs"]),
            str(a["completed_runs"]),
            a["batch"],
        ] for a in arms],
    ))
    out.append("```")
    out.append("")

    # --- scores ---
    out.append("## Scores")
    out.append("")
    out.append(
        (
            "Mean over the records where the metric was available; `n` is how many those "
            f"were. The delta in parentheses is against the control arm (`{control['arm']}`), "
            "paired over the records both arms scored — not the difference of the two "
            "means, which an arm that lost its hardest record would flatter."
        )
        if control
        else (
            "Mean over the records where the metric was available; `n` is how many those "
            "were. No control arm in this experiment, so no deltas."
        )
    )
    out.append("")
    rows = []
    for a in arms:
        row = [a["arm"]]
        for key, _ in METRICS:
            values = a["scores"].get(key, {})
            m = mean(values)
            delta = None
            if control is not None and a is not control:
                delta, _shared = paired_delta(values, control["scores"].get(key, {}))
            row.append(f"{fmt(m)}{fmt_delta(delta)} (n={len(values)})")
        rows.append(row)
    out.append("```")
    out.append(table(["arm", *(label for _, label in METRICS)], rows))
    out.append("```")
    out.append("")

    # --- behaviour ---
    out.append("## Agent behaviour")
    out.append("")
    out.append("Tool calls across the whole arm, and steps per run. This is where an "
               "ablation shows its substitution effect — which tool the agent reached "
               "for instead, and whether it needed more steps to get there.")
    out.append("")
    tools = sorted({t for a in arms for t in a["calls"]})
    out.append("```")
    out.append(table(
        ["arm", "steps/run", "median", "max", "total", *tools],
        [[
            a["arm"],
            fmt(a["mean_steps"], 1),
            fmt(a["median_steps"], 1),
            str(a["max_steps_seen"]),
            str(a["total_steps"]),
            *(str(a["calls"].get(t, 0)) for t in tools),
        ] for a in arms],
    ))
    out.append("```")
    out.append("")
    out.append("A step is one Thought → Action → Observation cycle, `submit_mapping` "
               "included. `MAX_STEPS` is 50; a run that hits it writes no result at all, "
               "so it would show up as a missing record rather than a 50-step one.")
    out.append("")

    # --- what did not score ---
    problems = [(a["arm"], e) for a in arms for e in a["errors"]]
    if problems:
        out.append("## Records that did not score")
        out.append("")
        out.append("An unscored record is missing from its arm's mean, so the arms above "
                   "are not always over the same set — check these before comparing.")
        out.append("")
        for arm, err in problems:
            out.append(f"- **{arm}** — {err}")
        out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", default=None,
                    help="Experiment tag: compares every batch-ablation-<tag>-* folder")
    ap.add_argument("--batch", action="append", default=[],
                    help="Explicit batch folder to include (repeatable); "
                         "overrides --tag discovery")
    ap.add_argument("--output-root", required=True,
                    help="Folder holding the arm batches")
    ap.add_argument("--control", default="full",
                    help="Arm the deltas are taken against, by arm name "
                         "(default: %(default)s; absent means no deltas)")
    ap.add_argument("--out", default=None,
                    help="Markdown destination (default: <output-root>/ablation-<tag>.md)")
    args = ap.parse_args(argv)

    root = Path(args.output_root)
    if args.batch:
        batch_dirs = [root / b for b in args.batch]
    elif args.tag:
        batch_dirs = sorted(root.glob(f"batch-ablation-{args.tag}-*"))
    else:
        # Newest experiment on disk, so the script is runnable with no arguments
        # right after a run.
        tags = sorted(
            {parsed[0] for p in root.glob("batch-ablation-*")
             if (parsed := split_batch_name(p.name))}
        )
        if not tags:
            print("error: no ablation batches found; pass --tag or --batch", file=sys.stderr)
            return 2
        args.tag = tags[-1]
        batch_dirs = sorted(root.glob(f"batch-ablation-{args.tag}-*"))

    missing = [d for d in batch_dirs if not d.is_dir()]
    if missing:
        for d in missing:
            print(f"error: no such batch: {d}", file=sys.stderr)
        return 2
    if not batch_dirs:
        print("error: no batches matched", file=sys.stderr)
        return 2

    arms = [read_arm(d, args.tag) for d in batch_dirs]
    unscored = [a["arm"] for a in arms if a["report"] is None]
    if unscored:
        print(f"note: no evaluation_*.json in: {', '.join(unscored)} "
              f"— run secpat evaluate --batch-path <folder> to score them",
              file=sys.stderr)

    control = next((a for a in arms if a["arm"] == args.control), None)
    text = render(arms, control, args.tag or "ablation")
    print(text)

    out_path = Path(args.out) if args.out else root / f"ablation-{args.tag or 'latest'}.md"
    out_path.write_text(text + "\n", encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "control": args.control if control else None,
                "arms": [
                    {
                        **{k: v for k, v in a.items() if k != "per_record_usage"},
                        "means": {
                            key: mean(a["scores"].get(key, {})) for key, _ in METRICS
                        },
                        "paired_delta_vs_control": None if control is None or a is control else {
                            key: dict(
                                zip(
                                    ("delta", "n_shared"),
                                    paired_delta(
                                        a["scores"].get(key, {}),
                                        control["scores"].get(key, {}),
                                    ),
                                )
                            )
                            for key, _ in METRICS
                        },
                    }
                    for a in arms
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWritten: {out_path}")
    print(f"Written: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
