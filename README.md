# secpatrec

Recovers a security pattern's roles from a codebase with an LLM agent, derives the
order those roles interact in, and scores recoveries against ground truth.

A run has three stages:

1. **Knowledge base** — the codebase is chunked and embedded into a ChromaDB index
   (`.kb/`, or `.kb-batch/<repo>/` in a batch), plus a tree-sitter struct index.
2. **Recovery agent** — a ReAct agent with `list`, `read`, `grep` and `search` tools
   maps each role in the pattern's `04_pattern.json` to a file, entity and lines.
3. **Interaction order** — the recovered mapping is walked on the repo's Joern CPG
   (`secpat-interaction-order`) and written into `recovery.json`.

Scoring (Levels 1–3) comes from `secpat-scoring`.

## Install

Python 3.11+. The two sibling packages are pinned by name, so install them first:

```bash
pip install -e ../secpat_interaction_recovery -e ../secpat_evaluation_metrics -e .
cp .env.example .env   # then fill in the keys you need
```

Level 3 scoring and stage 3 need Joern: either a Joern query service
(`JOERN_SERVICE_URL`), or a local Joern install with a full JDK — see
`../secpat_interaction_recovery/README.md`.

## Use

One repo, one pattern:

```bash
secpat recover <codebase> --pattern-file <pattern>/04_pattern.json --provider openai
```

A batch over a manifest, then scoring against a ground-truth snapshot:

```bash
secpat recover  --manifest-path <manifest.json> --batch-path <batch folder>
secpat evaluate --manifest-path <manifest.json> --batch-path <batch folder> \
    --groundtruth-path <ground-truth folder>
```

With the dataset in `../dataset`, for example:

```bash
secpat recover  --manifest-path ../dataset/manifest_python_full_20261005.json --batch-path runs/python-openai
secpat evaluate --manifest-path ../dataset/manifest_python_full_20261005.json --batch-path runs/python-openai \
    --groundtruth-path ../dataset/ground_truth/python
```

Other commands:

| Command | Does |
| ------- | ---- |
| `secpat baseline` | one-shot recovery: the whole repo packed into one prompt |
| `secpat ablation run / prompts / report` | the tool-ablation arms and their comparison |
| `secpat extract` | crawl a pattern page into `01_text.md` … `04_pattern.json` |
| `secpat pack` | pack a repo the way the baseline does |

`secpat <command> --help` lists each command's flags. Batches, scoring and ablation
are described in `src/secpat_recovery/batch/README.md`.

## Providers

`--provider` is one of `openai`, `claude`, `muse`, or a self-hosted family
(`deepseek`, `glm`, `qwen`, `devstral`, `gpt-oss`) served on any OpenAI-compatible
endpoint set by `LLM_BASE_URL`. `--model` overrides the provider's default model.
