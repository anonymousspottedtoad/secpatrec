"""The paper's experiments: `secpat evaluate`, `secpat interaction` and
`secpat recover --models`.

  evaluate     Score the stored recoveries (--recoveries-path) again, against the
               ground truth. No model is called.
  interaction  Derive each stored recovery's interaction order from its role
               mapping (stage 3) and save it in its recovery.json, then score the
               batch. No model is called. The baseline keeps the order its model
               stated, which is what it is scored on.
  recover      Run the pipeline with each model (knowledge base, agent,
               interaction order), or the one-shot baseline, then score. The
               models' keys and endpoints must be set (environment or .env); the
               run stops before it starts and names any that are missing.

Every part of the dataset is named by its own flag (see `dataset`), and the
records are the manifests' (--manifests), filtered by --repo / --pattern.

A batch is one experiment x model x language. Batches run one after another, and
each step is a `secpat` command run on one batch, its records one at a time.
Every run writes into one output folder:

    <mode>/<experiment>/<model>/<language>/   a batch: records + evaluation_<stamp>.json/.html
    index.html                                every batch's newest scores, rebuilt after each batch
    runs.json                                 each run's settings and batch statuses
    run.log                                   everything every run printed
    kb/                                       the knowledge-base cache (recover)

Joern always runs locally, on the CPGs in --cpg-path; a repo with no CPG there
gets one built there by joern-parse.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path

from . import report
from .dataset import cpg_file, load_runs, write_manifest

HERE = Path(__file__).resolve().parent
#: The repo root of an editable install, where `.env` is read from.
REPO = HERE.parents[2]
PROMPTS = HERE.parent / "ablation" / "prompts"
EXPERIMENTS = report.EXPERIMENTS
LANGUAGES = ["java", "python"]
#: What each ablation arm changes, as the paper's runs did it.
ABLATION_FLAGS = {
    "ablation_no_grep": [
        "--disable-tool", "grep", "--system-prompt-file", str(PROMPTS / "no-grep.md"),
    ],
    "ablation_no_search": [
        "--disable-tool", "search", "--system-prompt-file", str(PROMPTS / "no-search.md"),
    ],
}
QUIET = ("ok", "dry run", "no matching records")


def load_config() -> dict:
    """The models, as config.toml describes them."""
    return tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))["models"]


def secpat(*args: str) -> list[str]:
    return [sys.executable, "-m", "secpat_recovery.cli", *args]


class Run:
    """One invocation, writing into the shared output folder: run.log and runs.json."""

    def __init__(self, output: Path, settings: dict):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.history_path = output / "runs.json"
        self.history = (json.loads(self.history_path.read_text(encoding="utf-8"))
                        if self.history_path.is_file() else [])
        self.state = {**settings, "started_at": datetime.now().isoformat(timespec="seconds"),
                      "batches": []}
        self.history.append(self.state)
        self.save()
        self.log = (output / "run.log").open("a", encoding="utf-8")
        self.log.write(f"\n##### {self.state['started_at']} · {settings['mode']} #####\n")

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.log.write(text + "\n")
        self.log.flush()

    def call(self, cmd: list[str], env: dict) -> int:
        """Run a command, its output to the console and run.log."""
        self.say("$ secpat " + " ".join(cmd[3:]))
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            self.log.write(line)
        self.log.flush()
        return proc.wait()

    def record(self, batch: dict) -> None:
        self.state["batches"].append(batch)
        self.save()
        report.build(self.output)

    def save(self) -> None:
        self.history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def preflight(opts, models: dict, runs: list[dict], languages: list[str]) -> list[str]:
    """Everything that would stop the run part-way, found before it starts."""
    if not runs:
        return ["no record in --manifests matches --repo / --pattern"]
    problems = []
    if not opts.dry_run:
        for name, default in (("JOERN_BIN", "joern"), ("JOERN_PARSE_BIN", "joern-parse")):
            if not shutil.which(os.environ.get(name) or default):
                problems.append(f"{default} is not on PATH (or set {name}): install Joern 4.0.627,"
                                " see the README.")
    missing: dict[str, list[str]] = {}
    for r in runs:
        files = {
            "--repos-path": opts.repos_path / r["language"] / r["repo"],
            "--patterns-path": opts.patterns_path / r["pattern"] / "04_pattern.json",
            "--groundtruth-path": (opts.groundtruth_path / r["language"]
                                   / f"{r['pattern']}__{r['repo']}.json"),
        }
        for flag, path in files.items():
            if not path.exists():
                missing.setdefault(flag, []).append(str(path))
    for flag, paths in missing.items():
        problems.append(f"{len(paths)} file(s) missing under {flag}, e.g. {paths[0]}")
    if opts.mode in ("evaluate", "interaction"):
        for experiment in opts.experiments:
            for model in models:
                for lang in languages:
                    stored = opts.recoveries_path / experiment / model / lang
                    if not stored.is_dir():
                        problems.append(f"no stored recoveries at {stored}")
    if opts.mode == "recover" and not opts.dry_run:
        unset = {name: [v for v in cfg["needs"] if not os.environ.get(v)]
                 for name, cfg in models.items()}
        unset = {name: vs for name, vs in unset.items() if vs}
        if unset:
            ready = [m for m in models if m not in unset]
            lines = "\n".join(f"    {name}: {', '.join(vs)}" for name, vs in unset.items())
            skip = (f"    secpat recover {' '.join(f'--models {m}' for m in ready)} ..."
                    if ready else "    (no selected model is ready)")
            problems.append(
                f"These models have settings missing from the environment and {REPO / '.env'}:\n"
                f"{lines}\n"
                "Fill them in (see .env.example), or leave those models out, e.g.:\n" + skip
            )
    return problems


def batch_env(env: dict, cfg: dict) -> dict:
    """The child environment for one model: its own endpoint, nobody else's."""
    child = dict(env)
    child.pop("LLM_BASE_URL", None)
    if cfg.get("base_url"):
        child["LLM_BASE_URL"] = env.get(cfg["base_url"], "")
    return child


def run_batch(run: Run, opts, experiment: str, model: str, cfg: dict, lang: str,
              runs: list[dict], env: dict, tmp: Path) -> str:
    runs = [r for r in runs if r["language"] == lang]
    if not runs:
        return "no matching records"
    batch = run.output / opts.mode / experiment / model / lang
    manifest = write_manifest(runs, opts.repos_path, opts.patterns_path,
                              tmp / f"{experiment}-{model}-{lang}.json")
    on_batch = ["--manifest-path", str(manifest), "--batch-path", str(batch)]
    status = []

    if opts.mode in ("evaluate", "interaction"):
        source = opts.recoveries_path / experiment / model / lang
        stored = [source / f"{r['pattern']}__{r['repo']}" for r in runs]
        stored = [s for s in stored if (s / "recovery.json").is_file()]
        if opts.dry_run:
            run.say(f"would score {len(stored)} stored recoveries from {source}")
            return "dry run"
        for record in stored:
            (batch / record.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(record / "recovery.json", batch / record.name / "recovery.json")
        run.say(f"copied {len(stored)} stored recoveries from {source}")
        derive = opts.mode == "interaction" and experiment != "baseline"
        if derive and run.call(secpat("interaction", *on_batch), env):
            status.append("some orders failed")
    else:
        model_flags = ["--provider", cfg["provider"], "--model", cfg["model"]]
        dry = ["--dry-run"] if opts.dry_run else []
        if experiment == "baseline":
            cmd = secpat("baseline", *on_batch, *model_flags, *dry)
        else:
            cmd = secpat("recover", *on_batch, *model_flags, "--kb-root", str(run.output / "kb"),
                         *ABLATION_FLAGS.get(experiment, []), *dry)
        rc = run.call(cmd, batch_env(env, cfg))
        if opts.dry_run:
            return "dry run" if rc == 0 else f"dry run failed ({rc})"
        if rc:
            status.append("some recoveries failed")

    stored_order = ["--recovery-order", "stored"] if experiment == "baseline" else []
    groundtruth = ["--groundtruth-path", str(opts.groundtruth_path / lang)]
    if run.call(secpat("evaluate", *on_batch, *groundtruth, *stored_order), env):
        status.append("scoring failed")
    return "; ".join(status) or "ok"


def run(opts) -> int:
    """Run `opts.mode` ("evaluate", "interaction" or "recover") over the selected batches.

    `opts` carries models, experiments, repos, patterns, the dataset paths
    (manifests, groundtruth_path, cpg_path, repos_path, patterns_path,
    recoveries_path), output and dry_run. Raises ValueError for an unknown model
    or experiment.
    """
    config = load_config()
    models = list(opts.models) or list(config)
    experiments = EXPERIMENTS if "all" in opts.experiments else list(opts.experiments)
    unknown = [m for m in models if m not in config]
    unknown += [e for e in experiments if e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown model or experiment: {', '.join(unknown)} "
                         f"(models: {', '.join(config)}; experiments: {', '.join(EXPERIMENTS)})")
    opts.experiments = experiments
    for name in ("groundtruth_path", "cpg_path", "repos_path", "patterns_path", "recoveries_path"):
        if getattr(opts, name):
            setattr(opts, name, Path(getattr(opts, name)).resolve())
    selected = {m: config[m] for m in models}
    runs = load_runs(opts.manifests, opts.repos, opts.patterns)
    languages = [lang for lang in LANGUAGES if any(r["language"] == lang for r in runs)]

    problems = preflight(opts, selected, runs, languages)
    if problems:
        print("Cannot start:\n\n" + "\n\n".join(problems), file=sys.stderr)
        return 2

    unbuilt = [r for r in runs if not cpg_file(r, opts.cpg_path).exists()]
    if unbuilt and not opts.dry_run:
        print(f"{len(unbuilt)} of {len(runs)} records have no CPG in {opts.cpg_path}; "
              "joern-parse builds them there first.", flush=True)
    # Local Joern on these CPGs, whatever the environment says.
    env = {**os.environ, "JOERN_SERVICE_URL": "", "JOERN_CPG_ROOT": str(opts.cpg_path)}

    output = Path(opts.output).resolve()
    run = Run(output, {"mode": opts.mode, "models": models, "experiments": experiments,
                       "languages": languages, "repos": list(opts.repos),
                       "patterns": list(opts.patterns), "cpg": str(opts.cpg_path),
                       "manifests": [str(Path(m).resolve()) for m in opts.manifests],
                       "dry_run": opts.dry_run})
    batches = [(e, m, lang) for e in experiments for m in models for lang in languages]
    run.say(f"{opts.mode}: {len(batches)} batch(es) -> {output}")

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (experiment, model, lang) in enumerate(batches, 1):
            run.say(f"\n=== [{i}/{len(batches)}] {experiment} / {model} / {lang} ===")
            start = time.monotonic()
            status = run_batch(run, opts, experiment, model, selected[model], lang, runs, env,
                               Path(tmp))
            failed += status not in QUIET
            run.say(f"--- {status} ({time.monotonic() - start:.0f}s)")
            run.record({"experiment": experiment, "model": model, "language": lang,
                        "status": status, "seconds": round(time.monotonic() - start)})

    run.say(f"\nReport: {output / 'index.html'}")
    return 1 if failed else 0
