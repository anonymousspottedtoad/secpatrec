# Batch runs

The batch modes of `secpat recover` and `secpat evaluate` live here. A batch is a
manifest of repo × pattern records, run through the recovery pipeline, then scored
against a ground-truth snapshot:

```bash
secpat recover  --manifest-path <manifest.json> --batch-path <batch folder> [--parallel N]
secpat evaluate --manifest-path <manifest.json> --batch-path <batch folder> \
    --groundtruth-path <snapshot folder>
```

Paths are relative to where you run the command; a manifest's own `codebase_path`
and `pattern_file` are relative to the manifest's folder, so a manifest and the
files it lists can move together.

## Inputs

**The manifest** is a JSON object whose `runs` list holds one record per repo ×
pattern pair: `id`, `pattern`, `repo`, `pattern_file` (a `04_pattern.json`) and
`codebase_path`, both paths relative to the manifest's folder
(`secpat_recovery.manifest` reads it).

**The ground-truth snapshot** is a folder of `<pattern>__<repo>.json` files, one per
record, each holding that record's ground-truth role mapping under `role_mappings`
(same row schema as `recovery.json`). Scoring takes it as given — it neither knows
nor asks where the mapping came from — and reads nothing else, so a snapshot is
fixed and inspectable and two scoring runs on it agree. Other fields in the file
are ignored.

## Recover (`secpat recover --manifest-path` → `recover_batch.py`)

Runs `secpat recover` once per manifest record, each in its own process, as many
at once as the provider's parallel setting allows (below). Each record gets its
own subfolder in `--batch-path`:

```
<batch-path>/
├── <pattern>__<repo>/
│   ├── recovery.json     # the role mapping + the derived interaction order
│   ├── transcript.md     # the full agent run
│   └── run.log           # raw CLI output (where a failure is recorded)
├── batch_config.json     # what the batch was run with
└── evaluation_<timestamp>.html/.json   # `secpat evaluate` writes its report here
```

Useful flags: `--provider` (default `openai`), `--model`, `--no-order` (skip the
interaction order), `--parallel N`, `--dry-run`, `--only <substr>` / `--skip
<substr>` (repeatable), `--sleep N`. An existing `--batch-path` takes more records;
a record already in it is rerun, so resume with a manifest of the pending records.

### The interaction order

`secpat recover` ends by deriving the order its recovered roles interact in, so
each `recovery.json` carries an `interaction_order` block beside its
`role_mappings`. It is derived by `secpat-interaction-order` on the Joern query
service — the same detector scoring derives both sides' orders with, so an order
is only ever compared with one derived the same way.

The derivation cannot fail a run: a query service that cannot answer leaves
`available: false` with the reason in `recovery.json` and prints `no order` for
that record, rather than discarding a completed agent run. Scoring re-derives
both sides' orders itself — the ground truth has no order recorded — so the order
in `recovery.json` is the run's own end-to-end result, not an input to the score.

### Parallel runs (per provider)

```bash
# .env — one entry per provider; one not named runs a record at a time
SECPAT_PARALLEL=openai=4,claude=2,muse=8,qwen=4
```

`--parallel N` overrides the setting for one batch. For a hosted provider the
ceiling is a per-account token rate: past it, more runs in flight only means more
429s, which the client waits out and retries — slower, not lost — so set the
width to what the account sustains. A self-hosted server has no such quota, and
continuous batching (vLLM) means concurrent requests share the GPU rather than
queue behind each other.

Two things bound the width in practice:

- **The server's own capacity.** Past the point where the server runs out of KV
  cache it queues requests internally, so raising N further stops buying
  throughput. Match N to the endpoint, not to the record count.
- **One run at a time per repo.** Each record's KB lives in `.kb-batch/<repo>/`,
  and `build_kb` is not atomic, so two records on the *same* repo would race to
  create one ChromaDB index. Runs on a shared repo therefore serialize (the batch
  prints `waiting for <repo>'s KB`); distinct repos run at full width.

`--sleep` pauses between runs one at a time; with `--parallel` above 1 it becomes
the minimum spacing between run *launches*, which keeps N workers from hitting a
cold server (or starting N KB builds) at the same instant. Results are reported as
each run finishes, every line prefixed `[n/total] <record>`; each run's own output
stays in its `run.log`.

A KB left half-built by an interrupted run is detected and discarded — otherwise
it poisons every later run of that repo, because the engine only rebuilds when the
KB directory is absent or built from a different codebase.

### Tool ablation (`--disable-tool`, `--system-prompt-file`)

`--disable-tool NAME` (repeatable) withholds one of the agent's tools from every
run in the batch, and `--system-prompt-file` runs that batch on a prompt that does
not mention it. Both are needed: a tool lives in the schemas *and* in the prompt's
tool list, and a model told about a tool it cannot call spends steps finding that
out — steps the ablation would otherwise charge to the missing capability.

The KB is still built when `search` is withheld — only the agent's access to it is
ablated; `struct.json` is what scoring reads. Every batch folder carries a
`batch_config.json` (provider, model, `disabled_tools`, manifest, records) written
before the first run, because an arm and its control produce identically shaped
results.

## Evaluate (`secpat evaluate --batch-path` → `evaluate_batch.py`)

Scores each recovery result against its ground truth and writes
`<batch>/evaluation_<timestamp>.html` (+ `.json`) inside the batch folder, beside
the results it scores. The timestamp means a rescore never overwrites an earlier
report; `--out` sets a fixed path.

**The metrics are not reimplemented.** This calls `secpat_scoring.evaluate` once
per record; indexing the rows, resolving the role vocabulary, restricting both
sides to it and building the containment tree all happen inside that one call.
Roles the pattern flags `required: false` are not scored. Level 3 derives both
sides' orders with `secpat-interaction-order`'s `detect` — the same function
recovery uses — so the sides differ only in the mapping they walk from.
`--recovery-order stored` instead reads the recovery side's order from its
`recovery.json` (the one-shot baseline's claimed order).

- **Ground truth** — the record's role mapping in the snapshot (`--groundtruth-path`).
- **Recovery result** — the `recovery.json` the batch produced.
- `score = 1` at perfect agreement; `drift = 1 − score`. Levels are reported
  separately, never averaged into one number.

Two things to know:

- **Scoring needs Joern.** Either a query service (`JOERN_SERVICE_URL` /
  `JOERN_SERVICE_TOKEN`; prebuild the CPG for every repo the batch will touch), or,
  with `JOERN_SERVICE_URL` unset, a local Joern install, which builds a missing CPG
  into `JOERN_CPG_ROOT` on first use and reuses it after — see
  `secpat-interaction-order`'s README.
- **An empty struct index is refused**, not scored: a grammar that fails to load
  parses every file to nothing while Level 2 would still print plausible numbers.

## Ablation (`secpat ablation run`)

`secpat ablation run` runs one manifest once per **arm** — the control's provider,
model and manifest, changing only which tool the agent is given — then scores the
arms and the control against one snapshot and writes the comparison:

```bash
secpat ablation run --manifest-path <manifest.json> --groundtruth-path <snapshot> \
    --output-dir <dir> --control <control batch> --provider openai --model <model>
```

Its header (`secpat ablation run --help`) lists the flags and the layout. A rerun
resumes: each arm runs only the records without a `recovery.json` in its folder.

**The canonical prompt is never edited to run an experiment.** Each arm runs on its
own copy under `src/secpat_recovery/ablation/prompts/`, written by `secpat ablation
prompts` (`--check` verifies them, and the runner checks before spending an API
call). The generator is deliberately dumb: take `SYSTEM`, delete the withheld
tool's bullet, restate the sentences that prescribe its capability, write the
file — so an arm's edit is a `diff` away rather than a code path.

`secpat ablation report` puts the arms side by side — Level 1/2/3 means (with `n`,
the records each mean is over) and the agent's behaviour, steps per run and calls
per tool counted from each record's `run.log`. Deltas against the control are
paired: averaged over the records both arms scored, so an arm cannot show an
improvement built out of a record it failed to score.

## Files

| Path | Role |
| ---- | ---- |
| `recover_batch.py` | `secpat recover --manifest-path` — the batch runner |
| `evaluate_batch.py` | `secpat evaluate --batch-path` — scoring + HTML report |
| `../ablation/run.sh` | `secpat ablation run` — the tool-ablation driver |
| `../ablation/generate_prompts.py` | `secpat ablation prompts` — each arm's copy of the prompt |
| `../ablation/report.py` | `secpat ablation report` — arm-to-arm comparison |

Rebuildable caches, not results: `.kb-batch/` (KB caches) at the engine root.
