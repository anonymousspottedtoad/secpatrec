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
    help="Path to the pattern: a crawler 04_pattern.json, or a raw Mermaid .mmd/.md file.",
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
@click.option(
    "--parallel",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="Batch: records run at once (default: this provider's entry in SECPAT_PARALLEL, else 1).",
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
    help="Batch: seconds between run launches.",
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
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug-level logging.")
def recover(
    codebase_path: Path | None,
    pattern_file: Path | None,
    output_dir: Path,
    manifest_path: Path | None,
    batch_path: Path | None,
    parallel: int | None,
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
    verbose: bool,
) -> None:
    """Recover a security pattern implementation from a codebase.

    Give a codebase and --pattern-file for one repo, or --manifest-path and
    --batch-path to run every record of a manifest, each as its own run of this
    command, --parallel at a time.
    """
    from .pattern_input import load_pattern, parse_pattern

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
                    parallel=parallel,
                    sleep=sleep,
                    dry_run=dry_run,
                )
            )
        )
    if not codebase_path or not pattern_file:
        raise click.UsageError("give a codebase and --pattern-file, or --manifest-path")

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
    "--thinking",
    is_flag=True,
    default=False,
    help="Claude only: adaptive thinking (note Opus 5 thinks by default).",
)
@click.option(
    "--effort",
    default="high",
    type=click.Choice(["low", "medium", "high", "xhigh", "max"]),
    show_default=True,
    help="Claude only, with --thinking.",
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
    thinking: bool,
    effort: str,
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
        thinking=thinking,
        effort=effort,
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
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "The ground truth: a role-mapping file (same schema as recovery.json) for one "
        "repo, or the snapshot folder (<pattern>__<repo>.json per record) for a batch."
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
    default="./eval/results",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where the evaluation report is written.",
)
@click.option(
    "--pattern",
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
def evaluate(
    codebase_path: Path | None,
    prediction: Path | None,
    groundtruth_path: Path,
    manifest_path: Path | None,
    batch_path: Path | None,
    recovery_order: str,
    out: Path | None,
    kb_dir: Path,
    output_dir: Path,
    pattern_path: Path | None,
    no_order: bool,
) -> None:
    """Score a recovery against a ground-truth mapping, at every level.

    Runs the shared ``secpat_scoring`` framework, so every consumer of it computes
    the same numbers: two scorers would make any divergence between them read as
    recovery drift when it was really implementation drift.

    Levels 1 and 2 need only ``struct.json``. Level 3 additionally needs the
    pattern's behaviour walkthrough (``--pattern``) and a call graph, both built
    from ``codebase_path`` by the same ``secpat_interaction_order.detect``
    ``secpat recover`` derives its own stage 3 with — so this scores an order
    against one produced by an identically configured walk. Without the pattern,
    or with ``--no-order``, Level 3 reports itself unavailable and is left out of
    the composite rather than failing the run.

    With --manifest-path and --batch-path it scores a whole batch instead, each
    record against its role mapping in the ground-truth snapshot --groundtruth-path
    names.
    """
    import json as _json

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


@cli.command("extract")
@click.option(
    "--url",
    "urls",
    multiple=True,
    help="Pattern page URL to crawl (repeatable).",
)
@click.option(
    "--slug",
    "slugs",
    multiple=True,
    help="Pattern name slug to crawl, e.g. obscure_token_based_access_control (repeatable).",
)
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="JSON config listing urls/slugs to crawl.",
)
@click.option(
    "--random",
    "pick_random",
    is_flag=True,
    default=False,
    help="Crawl one random pattern from the catalogue.",
)
@click.option(
    "--page-dir",
    default="./security_patterns",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory where each pattern's reference-page folder is written.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose logging (per-step timings at DEBUG level).",
)
def extract(
    urls: tuple[str, ...],
    slugs: tuple[str, ...],
    config_file: Path | None,
    pick_random: bool,
    page_dir: Path,
    verbose: bool,
) -> None:
    """Crawl DistriNet security-pattern pages into per-pattern reference folders.

    Each pattern is read once by gpt-6-sol vision (diagram → Mermaid) and
    synthesised by gpt-6-sol (reasoning) into 04_pattern.json — the structured
    pattern consumed by `secpat recover`.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The HTTP client logs every request at INFO — too noisy for step tracing.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    from .crawler import CrawlerConfig, PatternCrawler, resolve_urls
    from .crawler.config import random_slug

    all_urls: list[str] = []
    if config_file is not None:
        cfg = CrawlerConfig.from_file(config_file)
        all_urls.extend(cfg.resolve_urls())

    extra = resolve_urls(list(urls), list(slugs)) if (urls or slugs) else []
    all_urls.extend(extra)

    if pick_random:
        slug = random_slug()
        click.echo(f"Random pattern selected: {slug}")
        all_urls.extend(resolve_urls(slugs=[slug]))

    # de-dup preserving order
    seen: set[str] = set()
    all_urls = [u for u in all_urls if not (u in seen or seen.add(u))]

    if not all_urls:
        raise click.UsageError(
            "No patterns to crawl. Provide --url, --slug, --config, or --random."
        )

    crawler = PatternCrawler(page_dir=page_dir)

    for url in all_urls:
        click.echo(f"\nCrawling: {url}")
        result = crawler.crawl_url(url)
        has = "yes" if result.has_diagram else "no"
        subflows = len(result.diagram_subflows)
        click.echo(f"  diagram: {has}  sub-flows: {subflows}  links: {result.link_count}")
        click.echo(
            f"  pages:   {result.page_dir}/  (01_text, 02_diagram, 03_subflows, 04_pattern.json)"
        )


# ── the ablation, the pack ──────────────────────────────────────────
# These commands wrap tools that parse their own arguments (argparse). Everything
# after the command name, --help included, is handed to the tool untouched, so
# each keeps its own flags and its own help text.

_PASS_THROUGH = dict(ignore_unknown_options=True, allow_extra_args=True)


@cli.group("ablation")
def ablation() -> None:
    """The tool ablation: run the arms, (re)write their prompts, compare them."""


@ablation.command("run", context_settings=_PASS_THROUGH, add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def ablation_run(args: tuple[str, ...]) -> None:
    """Run one manifest once per arm, score the arms and the control, compare them."""
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parent / "ablation" / "run.sh"
    # The script runs its Python steps on this interpreter, the one with the
    # package installed, unless PYTHON says otherwise.
    env = {**os.environ, "PYTHON": os.environ.get("PYTHON", sys.executable)}
    raise SystemExit(subprocess.call(["bash", str(script), *args], env=env))


@ablation.command("prompts", context_settings=_PASS_THROUGH, add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def ablation_prompts(args: tuple[str, ...]) -> None:
    """Write each arm's copy of the system prompt; --check verifies them."""
    from .ablation import generate_prompts

    raise SystemExit(generate_prompts.main(list(args), prog="secpat ablation prompts"))


@ablation.command("report", context_settings=_PASS_THROUGH, add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def ablation_report(args: tuple[str, ...]) -> None:
    """Put an ablation's arms side by side: scores and agent behaviour."""
    from .ablation import report

    raise SystemExit(report.main(list(args), prog="secpat ablation report"))


@cli.command("pack", context_settings=_PASS_THROUGH, add_help_option=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def pack(args: tuple[str, ...]) -> None:
    """Pack a repo into one prompt as the baseline does, or --measure its tokens."""
    from .baseline import pack as repo_pack

    raise SystemExit(repo_pack.main(list(args), prog="secpat pack"))


if __name__ == "__main__":
    cli()
