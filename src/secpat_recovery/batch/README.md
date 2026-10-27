# Batch runs

The batch modes of `secpat recover` and `secpat evaluate` live here. A batch is a
manifest of repo × pattern records, run through the recovery pipeline, then scored
against a ground-truth snapshot:

```bash
secpat recover  --manifest-path <manifest.json> --batch-path <batch folder>
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

Runs `secpat recover` once per manifest record, each in its own process, one
after another. Each record gets its own subfolder in `--batch-path`:

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
interaction order), `--dry-run`, `--only <substr>` / `--skip
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

### Running the records

Records run one at a time, in manifest order, each reported as it finishes with a
line prefixed `[n/total] <record>`; each run's own output stays in its `run.log`.
`--sleep N` pauses N seconds after each run. Each repo's KB is cached in
`.kb-batch/<repo>/` and reused by later records on that repo.

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

## Ablation prompts

An ablation arm is a `secpat recover` batch run with `--disable-tool` and
`--system-prompt-file` (above). **The canonical prompt is never edited to run an
experiment.** Each arm runs on its own copy under
`src/secpat_recovery/ablation/prompts/`: `no-grep.md` and `no-search.md` are
`SYSTEM` with the withheld tool's bullet deleted and the sentences that prescribe
its capability restated, so an arm's edit is a `diff` away rather than a code
path. `secpat recover --models <model> --experiments ablation_no_grep` (or
`ablation_no_search`) runs the arms on the dataset with these flags.

## Files

| Path | Role |
| ---- | ---- |
| `recover_batch.py` | `secpat recover --manifest-path` — the batch runner |
| `evaluate_batch.py` | `secpat evaluate --batch-path` — scoring + HTML report |
| `../ablation/prompts/` | the ablation arms' system prompts |

Rebuildable caches, not results: `.kb-batch/` (KB caches) at the engine root.
