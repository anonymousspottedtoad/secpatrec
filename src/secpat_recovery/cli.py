from __future__ import annotations

import logging
from pathlib import Path

import click

# `.env` is loaded in this package's __init__, which has already run by the time
# this module's body does — it has to, see the note there.
from .baseline import recover as one_shot
from .baseline.recover import MAX_INPUT_TOKENS, MAX_TOKENS, PROVIDER
from .engine import EngineConfig, PatternRecoveryEngine
from .llm import PROVIDERS, make_agent
from .manifest import load_records
from .output.mapping_report import write_json, write_transcript
from .recovery_agent import ABLATABLE_TOOLS


@click.group()
def cli() -> None:
    """Security pattern recovery engine."""


def _split(ctx, param, values: tuple[str, ...]) -> tuple[str, ...]:
    """Repeatable options that also take comma-separated lists."""
    return tuple(v.strip() for value in values for v in value.split(",") if v.strip())


def _experiment_options(groundtruth: bool = True, recoveries: bool = True):
    """The options that pick the experiments and name each part of the dataset.

    Every part is a flag of its own, and the experiments require each one they
    use: nothing is read from a default location.
    """
    folder = click.Path(exists=True, file_okay=False, path_type=Path)
    options = [
        click.option("--models", multiple=True, callback=_split, metavar="NAME",
                     help="Experiments: a model from experiments/config.toml "
                          "(gpt-6.1-sol, deepseek-v4-flash-0731, glm-5.3-flash, "
                          "muse-spark-1.3-contributor). Repeat for several."),
        click.option("--experiments", multiple=True, callback=_split, metavar="NAME",
                     help="Experiments: pipeline, baseline, ablation_no_grep, "
                          "ablation_no_search, or all. Repeat for several. Default: pipeline."),
        click.option("--repo", "repos", multiple=True, callback=_split,
                     help="Only this repo (exact name). Repeat for several."),
        click.option("--pattern", "patterns", multiple=True, callback=_split,
                     help="Only this pattern (exact name). Repeat for several."),
        click.option("--manifests", multiple=True,
                     type=click.Path(exists=True, dir_okay=False, path_type=Path),
                     help="Experiments: a manifest of records to run (pattern, repo, "
                          "language). Repeat for several."),
        *([click.option("--groundtruth-path", "groundtruth_path", type=folder,
                        help="Experiments: the ground truth, "
                             "<language>/<pattern>__<repo>.json.")]
          if groundtruth else []),
        click.option("--cpg-path", type=click.Path(file_okay=False, path_type=Path),
                     help="Experiments: the CPGs, <repo>__<JAVASRC|PYTHONSRC>.cpg. A repo "
                          "with no CPG there gets one built there by joern-parse."),
        click.option("--repos-path", type=folder,
                     help="Experiments: the codebases, <language>/<repo>/."),
        click.option("--patterns-path", type=folder,
                     help="Experiments: the security patterns, <pattern>/04_pattern.json."),
        *([click.option("--recoveries-path", type=folder,
                        help="Experiments: the stored recoveries to score, "
                             "<experiment>/<model>/<language>/<pattern>__<repo>/"
                             "recovery.json.")]
          if recoveries else []),
    ]

    def decorate(f):
        for option in reversed(options):
            f = option(f)
        return f

    return decorate


def _run_experiments(mode: str, **kw) -> None:
    from types import SimpleNamespace

    from .experiments import runner

    required = {
        "--manifests": kw["manifests"],
        "--groundtruth-path": kw["groundtruth_path"],
        "--cpg-path": kw["cpg_path"],
        "--repos-path": kw["repos_path"],
        "--patterns-path": kw["patterns_path"],
    }
    if mode in ("evaluate", "interaction"):
        required["--recoveries-path"] = kw["recoveries_path"]
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise click.UsageError(
            f"name every part of the dataset the experiments use; missing: {', '.join(missing)}"
        )
    opts = SimpleNamespace(
        mode=mode,
        models=kw["models"],
        experiments=kw["experiments"] or ("pipeline",),
        repos=kw["repos"],
        patterns=kw["patterns"],
        manifests=list(kw["manifests"]),
        groundtruth_path=kw["groundtruth_path"],
        cpg_path=kw["cpg_path"],
        repos_path=kw["repos_path"],
        patterns_path=kw["patterns_path"],
        recoveries_path=kw["recoveries_path"],
        output=kw["output_dir"],
        dry_run=kw["dry_run"],
    )
    try:
        raise SystemExit(runner.run(opts))
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


@cli.command("recover")
@click.argument(
    "codebase_path",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--pattern-file",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the pattern: a 04_pattern.json, or a raw Mermaid .mmd/.md file.",
)
@click.option(
    "--output-dir",
    default="./output",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where output files are written.",
)
@click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch: run every repo × pattern record of this manifest instead of one codebase.",
)
@click.option(
    "--batch-path",
    type=click.Path(path_type=Path),
    help="Batch: the folder the records' runs are written into (required with --manifest-path).",
)
@click.option("--only", multiple=True, help="Batch: only records matching this substring.")
@click.option("--skip", multiple=True, help="Batch: drop records matching this substring.")
@click.option(
    "--kb-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Batch: parent of the per-repo KB caches (default: <engine>/.kb-batch).",
)
@click.option(
    "--sleep",
    type=float,
    default=0,
    show_default=True,
    help="Batch: seconds to pause after each run.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Batch: print the plan, run nothing.")
@click.option(
    "--provider",
    default="openai",
    type=click.Choice(PROVIDERS),
    show_default=True,
    help="Model to run the recovery agent on; the self-hosted ones are served locally.",
)
@click.option("--model", default=None, help="Override the default model for the chosen provider.")
@click.option("--max-files", default=150, show_default=True, help="Max codebase files to send.")
@click.option(
    "--max-file-size",
    default=65536,
    show_default=True,
    help="Max bytes per file before truncation.",
)
@click.option(
    "--kb-dir",
    default=".kb",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory for the ChromaDB knowledge base.",
)
@click.option(
    "--rebuild-kb",
    is_flag=True,
    default=False,
    help="Delete and rebuild the KB before recovering.",
)
@click.option(
    "--no-order",
    "no_order",
    is_flag=True,
    default=False,
    help=(
        "Skip stage 3 and emit the role mapping alone. Normally the interaction "
        "order is derived from the repo's call graph on Joern — a query service "
        "(JOERN_SERVICE_URL), or local Joern when that is unset."
    ),
)
@click.option(
    "--language",
    default=None,
    help="Override the detected source language (picks the Joern frontend).",
)
@click.option(
    "--disable-tool",
    "disabled_tools",
    multiple=True,
    type=click.Choice(ABLATABLE_TOOLS),
    help=(
        "Withhold a tool from the recovery agent (repeatable) — this is the "
        "tool-ablation switch. The tool is removed from the schemas sent to the "
        "model; pair it with --system-prompt-file so the prompt does not "
        "advertise a tool the run does not have. The KB is still built when "
        "'search' is withheld; only the agent's access to it is ablated."
    ),
)
@click.option(
    "--system-prompt-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Run the agent on this system prompt instead of the built-in one. The "
        "tool-ablation arms use it to supply their own copy of the prompt with "
        "the withheld tool edited out (src/secpat_recovery/ablation/prompts/), so the "
        "canonical prompt is never rewritten to run an experiment."
    ),
)
@_experiment_options(recoveries=False)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug-level logging.")
def recover(
    codebase_path: Path | None,
    pattern_file: Path | None,
    output_dir: Path,
    manifest_path: Path | None,
    batch_path: Path | None,
    only: tuple[str, ...],
    skip: tuple[str, ...],
    kb_root: Path | None,
    sleep: float,
    dry_run: bool,
    provider: str,
    model: str | None,
    max_files: int,
    max_file_size: int,
    kb_dir: Path,
    rebuild_kb: bool,
    no_order: bool,
    language: str | None,
    disabled_tools: tuple[str, ...],
    system_prompt_file: Path | None,
    models: tuple[str, ...],
    experiments: tuple[str, ...],
    repos: tuple[str, ...],
    patterns: tuple[str, ...],
    manifests: tuple[Path, ...],
    groundtruth_path: Path | None,
    cpg_path: Path | None,
    repos_path: Path | None,
    patterns_path: Path | None,
    verbose: bool,
) -> None:
    """Recover a security pattern implementation from a codebase.

    Give a codebase and --pattern-file for one repo, or --manifest-path and
    --batch-path to run every record of a manifest, each as its own run of this
    command, one after another.

    With --models, run the paper's experiments on the dataset instead: each
    selected experiment (pipeline, baseline, ablation arms) with each model, every
    batch scored and summarised in <output-dir>/index.html.
    """
    from .pattern_input import load_pattern, parse_pattern

    if models:
        if codebase_path or manifest_path:
            raise click.UsageError(
                "--models runs the experiments on the dataset; leave out the codebase "
                "and --manifest-path"
            )
        _run_experiments("recover", models=models, experiments=experiments,
                         repos=repos, patterns=patterns, manifests=manifests,
                         groundtruth_path=groundtruth_path, cpg_path=cpg_path,
                         repos_path=repos_path, patterns_path=patterns_path,
                         recoveries_path=None, output_dir=output_dir, dry_run=dry_run)

    if manifest_path:
        if not batch_path:
            raise click.UsageError("--batch-path is required with --manifest-path")
        from types import SimpleNamespace

        from .batch import recover_batch

        raise SystemExit(
            recover_batch.run(
                SimpleNamespace(
                    manifest_path=manifest_path,
                    batch_path=batch_path,
                    provider=provider,
                    model=model,
                    kb_root=kb_root,
                    rebuild_kb=rebuild_kb,
                    no_order=no_order,
                    disable_tool=list(dict.fromkeys(disabled_tools)),
                    system_prompt_file=system_prompt_file,
                    only=list(only),
                    skip=list(skip),
                    sleep=sleep,
                    dry_run=dry_run,
                )
            )
        )
    if not codebase_path or not pattern_file:
        raise click.UsageError("give a codebase and --pattern-file, --manifest-path, or --models")

    # A Joern export blocks for minutes with no output of its own; the runner logs
    # its progress at INFO, so the run is silent without this.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    pattern_text = load_pattern(pattern_file)
    pattern = parse_pattern(pattern_file)
    llm_provider = make_agent(provider, model)
    config = EngineConfig(
        max_files=max_files,
        max_file_bytes=max_file_size,
        kb_dir=kb_dir,
        rebuild_kb=rebuild_kb,
        detect_order=not no_order,
        language=language,
        disabled_tools=tuple(dict.fromkeys(disabled_tools)),
        system_prompt=(
            system_prompt_file.read_text(encoding="utf-8") if system_prompt_file else None
        ),
        system_prompt_source=str(system_prompt_file) if system_prompt_file else "",
    )
    engine = PatternRecoveryEngine(provider=llm_provider, config=config)

    result, transcript = engine.recover(codebase_path, pattern_text)

    # Saved before stage 3 runs: the role mapping cost a full agent run, and a
    # stage-3 raise (KotlinCitation) must not take it down with it. A
    # recovery.json with no `interaction_order` is one whose stage 3 did not finish.
    output_dir.mkdir(parents=True, exist_ok=True)

    json_dest = output_dir / "recovery.json"
    write_json(result, json_dest)
    click.echo(f"Written: {json_dest}")

    transcript_dest = output_dir / "transcript.md"
    write_transcript(transcript, transcript_dest)
    click.echo(f"Written: {transcript_dest}")

    result["interaction_order"] = engine.derive_order(codebase_path, result, pattern)
    write_json(result, json_dest)
    click.echo(f"Updated with interaction order: {json_dest}")

    statuses = {r.get("status", "missing") for r in result.get("role_mappings", [])}
    if not statuses:
        badge = "UNKNOWN"
    elif statuses == {"present"}:
        badge = "FULL"
    elif statuses == {"missing"}:
        badge = "NONE"
    else:
        badge = "MIXED"
    click.echo(f"\nOverall status: {badge}")
    _echo_interaction_order(result.get("interaction_order") or {})


def _echo_interaction_order(order: dict) -> None:
    """Print the derived order, or the one line saying why there is none.

    Stage 3 never fails the run, so its outcome would otherwise be visible only
    inside recovery.json — and whether an order was derived at all is exactly the
    thing a run must not leave the reader to guess at.
    """
    if not order:
        return
    if not order.get("available"):
        click.echo(f"Interaction order: unavailable — {order.get('reason', 'no reason recorded')}")
        return

    interactions = order.get("interactions", [])
    click.echo(
        f"Interaction order ({order.get('language', '?')}): "
        f"{len(interactions)} interaction(s), entry {order.get('entryRole') or 'none'} "
        f"({order.get('entryRule') or 'n/a'})"
    )
    for item in interactions:
        click.echo(
            f"  {item['rank']:>2}. {item['interaction']}"
            f"   {item.get('file', '')}:{item.get('line', '')}"
        )
    if reason := order.get("reason"):
        click.echo(f"  note: {reason}")


@cli.command("baseline")
@click.argument(
    "codebase_path",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--pattern-file",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="The pattern's 04_pattern.json (single-repo mode).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Where to write recovery.json (default: baseline_out/<repo>).",
)
@click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch: run every repo × pattern record of this manifest.",
)
@click.option(
    "--batch-path",
    type=click.Path(path_type=Path),
    help="Batch: write into this folder, skipping records already recovered "
    "(required with --manifest-path).",
)
@click.option("--only", multiple=True, help="Batch: keep matching records.")
@click.option("--skip", multiple=True, help="Batch: drop matching records.")
@click.option(
    "--smallest",
    type=int,
    metavar="N",
    help="Batch: keep only the N smallest repos (ranked without any API call).",
)
@click.option(
    "--provider",
    default=PROVIDER,
    type=click.Choice(PROVIDERS),
    show_default=True,
    help="Same providers as `secpat recover`.",
)
@click.option(
    "--model", default=None, help="Model tag within the provider (default: the provider's own)."
)
@click.option(
    "--max-tokens",
    type=int,
    default=MAX_TOKENS,
    show_default=True,
    help="Output cap, where the provider has one.",
)
@click.option(
    "--max-input-tokens",
    type=int,
    default=MAX_INPUT_TOKENS,
    show_default=True,
    help="Refuse a pack larger than this.",
)
@click.option("--all-text", is_flag=True, default=False, help="Pack every UTF-8 file.")
@click.option(
    "--max-file-bytes",
    type=int,
    default=0,
    metavar="N",
    help="Skip files larger than N bytes (0 = no cap).",
)
@click.option("--dry-run", is_flag=True, default=False, help="List what would run, send nothing.")
def baseline(
    codebase_path: Path | None,
    pattern_file: Path | None,
    output_dir: Path | None,
    manifest_path: Path | None,
    batch_path: Path | None,
    only: tuple[str, ...],
    skip: tuple[str, ...],
    smallest: int | None,
    provider: str,
    model: str | None,
    max_tokens: int,
    max_input_tokens: int,
    all_text: bool,
    max_file_bytes: int,
    dry_run: bool,
) -> None:
    """One-shot baseline: the whole repo in one prompt, one call, no agent.

    The comparison point for `secpat recover` — see `baseline/__init__.py`. Give
    a codebase and --pattern-file for one repo, or --manifest-path and
    --batch-path for a batch in the layout `secpat evaluate` scores.
    """
    options = dict(
        provider_name=provider,
        model=model,
        all_text=all_text,
        max_file_bytes=max_file_bytes,
        max_input_tokens=max_input_tokens,
        max_tokens=max_tokens,
    )

    if manifest_path:
        if not batch_path:
            raise click.UsageError("--batch-path is required with --manifest-path")
        records = load_records(manifest_path, only, skip)
        if not records:
            raise click.UsageError("no records matched")
        rows = one_shot.plan(records, all_text=all_text, max_file_bytes=max_file_bytes)
        if smallest:
            rows = rows[:smallest]
        if dry_run:
            one_shot.print_plan(rows)
            return
        raise SystemExit(one_shot.run_batch([rec for rec, _, _ in rows], batch_path, **options))

    if not codebase_path or not pattern_file:
        raise click.UsageError("give a codebase and --pattern-file, or --manifest-path")
    if dry_run:
        click.echo(f"{codebase_path} against {pattern_file}")
        return

    try:
        result = one_shot.recover(codebase_path, pattern_file, **options)
    except ValueError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from exc

    out_dir = output_dir or Path("baseline_out") / codebase_path.name
    one_shot.write_run(out_dir, result)
    meta = result["baseline"]
    click.echo(
        f"{len(result['role_mappings'])} roles ·"
        f" {meta['prompt_tokens']:,} prompt tokens ·"
        f" {meta['seconds']}s -> {out_dir / 'recovery.json'}"
    )


@cli.command("evaluate")
@click.argument(
    "codebase_path",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--prediction",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="The pipeline's recovery.json to score (single-repo mode).",
)
@click.option(
    "--groundtruth-path",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    help=(
        "The ground truth: a role-mapping file (same schema as recovery.json) for one "
        "repo, the snapshot folder (<pattern>__<repo>.json per record) for a batch, or "
        "for the experiments the folder holding <language>/<pattern>__<repo>.json."
    ),
)
@click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Batch: score every repo × pattern record of this manifest.",
)
@click.option(
    "--batch-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Batch: the batch folder to score (required with --manifest-path).",
)
@click.option(
    "--recovery-order",
    type=click.Choice(["derived", "stored"]),
    default="derived",
    show_default=True,
    help="Batch: Level 3's recovery-side order — walked on Joern, or read from the "
    "recovery.json's interaction_order (the baseline's claim).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Batch: HTML report path (default: <batch>/evaluation_<timestamp>.html).",
)
@click.option(
    "--kb-dir",
    default=".kb",
    show_default=True,
    type=click.Path(path_type=Path),
    help="KB directory holding struct.json (the AST containment index).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Where the report is written (default: ./eval/results for one repo, "
    "./output for the experiments).",
)
@click.option(
    "--pattern-file",
    "pattern_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Reference pattern JSON. Supplies the role vocabulary and Level 3's "
        "behaviour walkthrough; without it only Levels 1 and 2 can be scored."
    ),
)
@click.option(
    "--no-order",
    "no_order",
    is_flag=True,
    default=False,
    help="Skip Level 3 — the same switch `secpat recover` has for its own stage 3.",
)
@_experiment_options(groundtruth=False)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Experiments: print the plan, run nothing."
)
def evaluate(
    codebase_path: Path | None,
    prediction: Path | None,
    groundtruth_path: Path | None,
    manifest_path: Path | None,
    batch_path: Path | None,
    recovery_order: str,
    out: Path | None,
    kb_dir: Path,
    output_dir: Path | None,
    pattern_path: Path | None,
    no_order: bool,
    models: tuple[str, ...],
    experiments: tuple[str, ...],
    repos: tuple[str, ...],
    patterns: tuple[str, ...],
    manifests: tuple[Path, ...],
    cpg_path: Path | None,
    repos_path: Path | None,
    patterns_path: Path | None,
    recoveries_path: Path | None,
    dry_run: bool,
) -> None:
    """Score a recovery against a ground-truth mapping, at every level.

    Runs the shared ``secpat_scoring`` framework, so every consumer of it computes
    the same numbers: two scorers would make any divergence between them read as
    recovery drift when it was really implementation drift.

    Levels 1 and 2 need only ``struct.json``. Level 3 additionally needs the
    pattern's behaviour walkthrough (``--pattern-file``) and a call graph, both built
    from ``codebase_path`` by the same ``secpat_interaction_order.detect``
    ``secpat recover`` derives its own stage 3 with — so this scores an order
    against one produced by an identically configured walk. Without the pattern,
    or with ``--no-order``, Level 3 reports itself unavailable and is left out of
    the composite rather than failing the run.

    With --manifest-path and --batch-path it scores a whole batch instead, each
    record against its role mapping in the ground-truth snapshot --groundtruth-path
    names.

    With neither a codebase, --prediction nor --manifest-path, it scores the
    dataset's stored recoveries (the paper's experiment outputs) again, batch by
    batch, and summarises them in <output-dir>/index.html. No model is called.
    """
    import json as _json

    if not (codebase_path or prediction or manifest_path):
        _run_experiments("evaluate", models=models, experiments=experiments,
                         repos=repos, patterns=patterns, manifests=manifests,
                         groundtruth_path=groundtruth_path, cpg_path=cpg_path,
                         repos_path=repos_path, patterns_path=patterns_path,
                         recoveries_path=recoveries_path,
                         output_dir=output_dir or Path("./output"),
                         dry_run=dry_run)
    if models:
        raise click.UsageError(
            "--models scores the dataset's recoveries; leave out the codebase, "
            "--prediction and --manifest-path"
        )
    if not groundtruth_path:
        raise click.UsageError("--groundtruth-path is required to score a recovery or a batch")
    output_dir = output_dir or Path("./eval/results")

    if manifest_path:
        if not batch_path:
            raise click.UsageError("--batch-path is required with --manifest-path")
        from types import SimpleNamespace

        from .batch import evaluate_batch

        raise SystemExit(
            evaluate_batch.run(
                SimpleNamespace(
                    manifest_path=manifest_path,
                    groundtruth_path=groundtruth_path,
                    batch_path=batch_path,
                    recovery_order=recovery_order,
                    out=out,
                )
            )
        )
    if not codebase_path or not prediction:
        raise click.UsageError("give a codebase and --prediction, or --manifest-path")

    from secpat_scoring import EvaluationRequest, evaluate

    from .interaction import detect_language, repo_key

    struct_path = kb_dir / "struct.json"
    if not struct_path.exists():
        raise click.UsageError(
            f"No struct.json in {kb_dir}. Build the KB first "
            f"(secpat recover {codebase_path} ... --rebuild-kb)."
        )
    struct = _json.loads(struct_path.read_text(encoding="utf-8"))

    def _rows(path: Path) -> list[dict]:
        return _json.loads(path.read_text(encoding="utf-8")).get("role_mappings", [])

    pattern = _json.loads(pattern_path.read_text(encoding="utf-8")) if pattern_path else None

    # Level 3's backend is built by the shared package, exactly as `pattern
    # recover` stage 3 and `secpat evaluate --batch-path` build theirs —
    # wired for the invocations a call graph cannot contain. Scoring an order
    # against a walk configured differently from the one that produced it
    # measures the difference between two walkers, not between two recoveries.
    predicted_rows = _rows(prediction)
    language = detect_language(
        codebase_path, [row.get("file", "") for row in predicted_rows]
    )

    # Level 3 derives both sides on Joern, from the conditioned mapping the metric
    # hands it — which is why this is a callable rather than a value computed here.
    # Levels 1 and 2 are unaffected if Joern cannot answer; Level 3 reports itself
    # unavailable and says why.
    detect_order = None
    if not no_order:
        from secpat_interaction_order import detect as _detect

        repo = repo_key(codebase_path)
        click.echo(f"Level 3: deriving {repo} ({language}) on Joern")

        # The walkthrough is handed in by the scorer, without optional-role steps.
        def detect_order(mappings, behaviour):  # noqa: F811 — the None above is the disabled case
            return _detect(
                repo, language, mappings, behaviour=behaviour, repo_path=codebase_path
            )

    report = evaluate(
        EvaluationRequest(
            ground_truth_rows=_rows(groundtruth_path),
            recovery_result_rows=predicted_rows,
            pattern=pattern,
            struct=struct,
            detect_order=detect_order,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    text_path = output_dir / "evaluation.txt"
    json_path.write_text(_json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = [
        f"Roles scored: {report.ground_truth_roles} ground truth / "
        f"{report.recovery_result_roles} recovery result ({report.roles.scope})",
        "",
    ]
    for m in report.metrics:
        d = m.to_dict()
        score = d["score"]
        lines.append(
            f"{d['level']}: "
            + ("unavailable" if score is None else f"{score:.4f} (drift {d['drift']})")
        )
    lines += ["", f"Overall drift: {report.overall_drift}"]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    click.echo(f"Written: {json_path}")
    click.echo(f"Written: {text_path}")
    click.echo("")
    for line in lines:
        click.echo(line)


@cli.command("interaction")
@click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="One batch: derive the order for every record of this manifest (with --batch-path).",
)
@click.option(
    "--batch-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="One batch: the folder holding the records' recovery.json files.",
)
@_experiment_options()
@click.option(
    "--output-dir",
    default="./output",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Experiments: the folder every run writes into.",
)
@click.option(
    "--dry-run", is_flag=True, default=False, help="Experiments: print the plan, run nothing."
)
def interaction(
    manifest_path: Path | None,
    batch_path: Path | None,
    models: tuple[str, ...],
    experiments: tuple[str, ...],
    repos: tuple[str, ...],
    patterns: tuple[str, ...],
    manifests: tuple[Path, ...],
    groundtruth_path: Path | None,
    cpg_path: Path | None,
    repos_path: Path | None,
    patterns_path: Path | None,
    recoveries_path: Path | None,
    output_dir: Path,
    dry_run: bool,
) -> None:
    """Derive the interaction order from existing role mappings, and score it.

    By default, runs on the dataset's stored recoveries (the paper's experiment
    outputs): each recovery's interaction order is derived from its role mapping
    (stage 3 of the pipeline), then every batch is scored against the ground truth
    and summarised in <output-dir>/index.html. No model is called.

    With --manifest-path and --batch-path, derives the order for one batch of
    recovery.json files instead, and writes it into each of them.
    """
    if manifest_path or batch_path:
        if not (manifest_path and batch_path):
            raise click.UsageError("--manifest-path and --batch-path go together")
        from .experiments.dataset import derive_orders

        records = [
            r for r in load_records(manifest_path)
            if (not repos or r["repo"] in repos) and (not patterns or r["pattern"] in patterns)
        ]
        click.echo(f"{len(records)} record(s): interaction order in {batch_path}")
        raise SystemExit(derive_orders(records, batch_path))
    _run_experiments("interaction", models=models, experiments=experiments,
                     repos=repos, patterns=patterns, manifests=manifests,
                     groundtruth_path=groundtruth_path, cpg_path=cpg_path,
                     repos_path=repos_path, patterns_path=patterns_path,
                     recoveries_path=recoveries_path, output_dir=output_dir, dry_run=dry_run)


@cli.command("report")
@click.option(
    "--output-dir",
    default="./output",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
    help="The experiments' output folder.",
)
def report(output_dir: Path) -> None:
    """Rebuild <output-dir>/index.html from the experiments' scores."""
    from .experiments import report as experiments_report

    click.echo(f"Written: {experiments_report.build(output_dir)}")


# ── the pack ─────────────────────────────────────────────────────────
# `pack` wraps a tool that parses its own arguments (argparse). Everything after
# the command name, --help included, is handed to it untouched, so it keeps its
# own flags and its own help text.

_PASS_THROUGH = dict(ignore_unknown_options=True, allow_extra_args=True)


@cli.command("pack", context_settings=_PASS_THROUGH, add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def pack(args: tuple[str, ...]) -> None:
    """Pack a repo into one prompt as the baseline does, or --measure its tokens."""
    from .baseline import pack as repo_pack

    raise SystemExit(repo_pack.main(list(args), prog="secpat pack"))


if __name__ == "__main__":
    cli()
