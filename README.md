# secpatrec

**secpatrec recovers how a security pattern is implemented in a codebase.** Give
it a repository and a pattern from a security-pattern catalogue (password-based
authentication, verifiable-token authentication, session-based access control,
rate limiting, …). It tells you:

- **where each role of the pattern lives** — for every role (Enforcer,
  Verification manager, Password store, …) the file, entity and line that
  implements it, or that the role is missing or handled by an external library;
- **how those roles interact** — the order in which they call each other,
  derived from the code itself, not from the model's guess;
- **how good the recovery is** — scores against a ground truth at three levels:
  Level 1 (do the roles agree), Level 2 (do they sit in the same place in the
  code's structure) and Level 3 (do the interactions and their order agree).

It works on Java and Python codebases. A run has three stages:

1. **Knowledge base** — the codebase is chunked and embedded into a ChromaDB
   index, plus a tree-sitter index of its structure.
2. **Recovery agent** — an LLM agent with four tools (`list_dir`, `read_file`,
   `grep`, `search`) maps each role of the pattern's `04_pattern.json` to code.
3. **Interaction order** — the role mapping is walked on the repository's code
   property graph (CPG) with [Joern](https://joern.io), which yields the order
   the roles interact in.

The interaction walk (`secpat-interaction-order`) and the scoring
(`secpat-scoring`) are separate packages that install with secpatrec. The dataset of the
paper (100 records: 5 patterns × 10 repositories × 2 languages, with ground
truth, prebuilt CPGs and every experiment's outputs) is on Hugging Face, in the
bucket [`AnonymousSpottedToad/securityPatternsRecovery`](https://huggingface.co/buckets/AnonymousSpottedToad/securityPatternsRecovery).

## The paper

secpatrec is the code of the paper *A Hybrid Approach for Automated Security
Pattern Recovery*: SecPatRec, the approach of its Section III-B, and every
experiment of its evaluation (Section V, RQ1–RQ4). The paper's appendix is at
<https://doi.org/10.5281/zenodo.23664769>.

The paper releases five artifacts, listed in its Appendix G:

| Artifact | What it holds | In the paper |
| -------- | ------------- | ------------ |
| [secpatrec](https://github.com/anonymousspottedtoad/secpatrec) (this repository) | the role detection and mapping agent, the one-shot baseline, the two ablation settings, and the `secpat` command that runs and scores them | Sections III-B and V |
| [secpat_interaction_recovery](https://github.com/anonymousspottedtoad/secpat_interaction_recovery) | the walk of the Joern CPG that recovers the interaction sequence; installs with secpatrec | Section III-B2, Appendix C |
| [secpat_evaluation_metrics](https://github.com/anonymousspottedtoad/secpat_evaluation_metrics) | RDA, MLS and IRC; installs with secpatrec | Section IV-C, Appendix A-D |
| [the dataset](https://huggingface.co/buckets/AnonymousSpottedToad/securityPatternsRecovery) | the groundtruth, the codebases and their CPGs, the five patterns, and every experiment's outputs and scores | Sections IV-B and V, Appendices A-A and D |
| [secpat_annotation_portal](https://github.com/anonymousspottedtoad/secpat_annotation_portal) | the web interface the annotators used | Appendix D-B |

The code and the dataset name some of the paper's terms differently:

| Paper | Here |
| ----- | ---- |
| RDA, MLS, IRC | Level 1, Level 2, Level 3; `evaluation.json` also gives `drift = 1 - score` |
| recovery instance | record: one pattern in one repository |
| the specification *P* of a pattern | the pattern's `04_pattern.json` |
| P1, P2, P3, P4, P5 | `authentication_pwd`, `verifiable_token_based_authentication`, `obscure_token_based_access_control`, `session_based_access_control`, `limit_request_rate` |
| GPT-6.1, Muse Spark, DeepSeek, GLM | `gpt-6.1-sol`, `muse-spark-1.3-contributor`, `deepseek-v4-flash-0731`, `glm-5.3-flash` |
| semantic search, lexical (keyword) search | the agent's `search` and `grep` tools |
| SecPatRec, the one-shot baseline, and RQ4's settings without lexical or semantic search | the experiments `pipeline`, `baseline`, `ablation_no_grep`, `ablation_no_search` |

[What reproduces which result](#what-reproduces-which-result) maps the
experiments below to the paper's tables and figures.

## Prerequisites

| | Version | Check |
| - | ------- | ----- |
| Python | 3.11 or newer | `python3 --version` |
| Java | a full JDK 19 or newer, as Joern asks (not a JRE; the dataset's CPGs were built with JDK 25) | `java -version` and `javac -version` |
| Joern | 4.0.627 exactly: the prebuilt CPGs are in its format | installed below |
| git, curl, unzip | any | |
| Disk | about 5 GB: Joern 1.9 GB, the dataset 1.2 GB, the Python packages | |

On macOS, Joern's launchers also need GNU `readlink`: `brew install coreutils`.
A JDK if you have none: `brew install --cask temurin` (macOS) or
`sudo apt-get install -y openjdk-21-jdk` (Debian, Ubuntu).

Running the full pipeline also needs access to the models you run (an OpenAI or
Muse API key; for DeepSeek and GLM, their Modal endpoints and a proxy auth token;
see [Choosing models](#choosing-models)); scoring the paper's recoveries needs
none.

## Installation

### 1. secpatrec

Clone the repository:

```bash
git clone https://github.com/anonymousspottedtoad/secpatrec.git
```

Go into it:

```bash
cd secpatrec
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Update pip:

```bash
pip install --upgrade pip
```

Install secpatrec. This also installs `secpat-interaction-order` and
`secpat-scoring` from their GitHub release tags (listed in `pyproject.toml`):

```bash
pip install -e .
```

Check that the `secpat` command works:

```bash
secpat --help
```

### 2. Joern 4.0.627

Joern's own installer, as in [its installation guide](https://docs.joern.io/installation/),
but taken from the 4.0.627 release instead of the latest one. It downloads the
build for your system (about 1.8 GB) and installs it in `~/bin/joern/joern-cli`.

Make a folder for the installer:

```bash
mkdir -p ~/joern-installer
```

Go into it:

```bash
cd ~/joern-installer
```

Download the installer from the 4.0.627 release:

```bash
curl -L "https://github.com/joernio/joern/releases/download/v4.0.627/joern-install.sh" -o joern-install.sh
```

Make it executable:

```bash
chmod u+x joern-install.sh
```

Install Joern 4.0.627. `--version` is needed even with the script from that
release (without it, the script downloads the latest Joern), and
`--without-plugins` skips Joern's query database, which secpatrec does not use.
If `~/bin/joern/joern-cli` already exists, add `--reinstall`:

```bash
./joern-install.sh --version=v4.0.627 --without-plugins
```

Go back to the repo:

```bash
cd -
```

Delete the installer's folder, which still holds its 1.8 GB download:

```bash
rm -rf ~/joern-installer
```

Put Joern on your `PATH` (add this line to your `~/.zshrc` or `~/.bashrc` to keep
it in new shells):

```bash
export PATH="$HOME/bin/joern/joern-cli:$PATH"
```

Check that both commands are found:

```bash
which joern joern-parse
```

### 3. The dataset, from Hugging Face

The dataset goes in `dataset/` inside this repo (git ignores it).

Install the Hugging Face command-line tool:

```bash
pip install --upgrade huggingface_hub
```

Download the dataset, which is the Hugging Face bucket
[`AnonymousSpottedToad/securityPatternsRecovery`](https://huggingface.co/buckets/AnonymousSpottedToad/securityPatternsRecovery):

```bash
hf buckets sync hf://buckets/AnonymousSpottedToad/securityPatternsRecovery dataset
```

Check it arrived:

```bash
ls dataset
```

`ls` should show `codebase/`, `experiment_outputs/`, `ground_truth/`,
`security_patterns/`, the two manifests, `Recovery_Conversation_Session.txt`,
`LICENSE` and the dataset's own `README.md`, which describes every file.

## Run experiments

Every experiment runs through the `secpat` command, from the repo folder with
the virtual environment active:

- `secpat evaluate` scores the paper's recoveries (stored in the dataset) again,
  against the ground truth. No model is called.
- `secpat interaction` runs the interaction detector on the paper's recoveries —
  it derives each one's interaction order from its role mapping — then scores
  them. No model is called.
- `secpat recover --models ...` runs the pipeline (or the baseline, or an
  ablation arm) with each model, then scores the result.

All three write into `output/`, and `output/index.html` shows the scores.

### Dataset paths

Every experiment command names each part of the dataset with its own flag, and
none of them has a default. In the dataset downloaded in step 3 of the
installation, they are:

| Flag | What | In the dataset |
| ---- | ---- | -------------- |
| `--manifests` | the records to run: each one's pattern, repo and language (repeat for several) | `dataset/manifest_java_full_20261005.json`, `dataset/manifest_python_full_20261005.json` |
| `--groundtruth-path` | the ground truth, `<language>/<pattern>__<repo>.json` | `dataset/ground_truth` |
| `--cpg-path` | the CPGs, `<repo>__<JAVASRC\|PYTHONSRC>.cpg` | `dataset/codebase/cpg` |
| `--repos-path` | the codebases, `<language>/<repo>/` | `dataset/codebase/repos` |
| `--patterns-path` | the security patterns, `<pattern>/04_pattern.json` | `dataset/security_patterns` |
| `--recoveries-path` | the paper's recoveries to score (`evaluate`, `interaction` only) | `dataset/experiment_outputs` |

Point any flag elsewhere to use your own copy of that part. A repo with no CPG in
`--cpg-path` gets one built there by `joern-parse`, so an empty folder (e.g.
`--cpg-path output/cpg`) builds every CPG again.

### Choosing models

Give `--models` once for each model; the models run one after the other. For
example, gpt-6.1-sol and then Muse:
`secpat recover --models gpt-6.1-sol --models muse-spark-1.3-contributor ...`

| `--models` | Model | Settings it needs (in `.env` or the environment) |
| ---------- | ----- | ------------------------------------------------ |
| `gpt-6.1-sol` | OpenAI gpt-6.1-sol | `OPENAI_API_KEY` |
| `muse-spark-1.3-contributor` | Meta Muse | `MUSE_API_KEY` |
| `deepseek-v4-flash-0731` | DeepSeek V4 Flash, on Modal | `DEEPSEEK_BASE_URL`, `MODAL_KEY`, `MODAL_SECRET` |
| `glm-5.3-flash` | GLM 5.3 Flash, on Modal | `GLM_BASE_URL`, `MODAL_KEY`, `MODAL_SECRET` |

`secpat recover` needs `--models`, and checks every picked model's settings
before it starts. `secpat evaluate` and `secpat interaction` take the same option
to pick whose recoveries to score; they need no settings, and without `--models`
they score all four.

Each model's settings go in `.env`, in the repo folder; exported environment
variables work too and take precedence. Create it from the template:

```bash
cp -n .env.example .env
```

#### gpt-6.1-sol

Set `OPENAI_API_KEY` in `.env` to your OpenAI API key.

#### muse-spark-1.3-contributor

Set `MUSE_API_KEY` in `.env` to your Muse API key.

#### deepseek-v4-flash-0731 and glm-5.3-flash

DeepSeek and GLM run on Modal, each behind an OpenAI-compatible API (vLLM). Set
these in `.env`:

| Setting | Value |
| ------- | ----- |
| `DEEPSEEK_BASE_URL` | the DeepSeek endpoint's URL, ending in `/v1` |
| `GLM_BASE_URL` | the GLM endpoint's URL, ending in `/v1` |
| `MODAL_KEY`, `MODAL_SECRET` | the id and secret of a Modal proxy auth token (Modal dashboard → Settings → Proxy Auth Tokens) |

Each endpoint must serve its model under the name `secpat` asks for:
`deepseek-ai/DeepSeek-V4-Flash-0731` and `zai-org/GLM-5.3-Flash`.

**The experiments below use GPT** (`--models gpt-6.1-sol`) in every command. To
run them with other models, give `--models` once for each model, as above — for
example, add `--models muse-spark-1.3-contributor` to any of the commands.

### Model settings

Each model's request settings are fixed in the code (`src/secpat_recovery/llm/`),
not in `.env`:

| `--models` | Reasoning effort | Sampling | Output cap per agent step |
| ---------- | ---------------- | -------- | ------------------------- |
| `gpt-6.1-sol` | medium | the API's defaults | 8,192 tokens |
| `muse-spark-1.3-contributor` | low | temperature 1, top_p 1 | 32,000 tokens |
| `deepseek-v4-flash-0731` | `LLM_REASONING_EFFORT` if set (it is empty in `.env.example`), otherwise the endpoint's default | the endpoint's defaults | 8,192 tokens |
| `glm-5.3-flash` | low | the endpoint's defaults | 8,192 tokens |

- The agent has at most 50 steps to submit its mapping (`MAX_STEPS` in
  `src/secpat_recovery/recovery_agent/agent.py`).
- Semantic search embeds the code with `BAAI/bge-small-en-v1.5`.
- The baseline makes one call, with an output cap of 32,000 tokens, and only for
  a prompt within 900,000 tokens of the `o200k_base` tokenizer.

### Output

The scores are in `output/index.html`, in the repo folder: one report for every
experiment you run, rebuilt after each batch. Each batch also has its own
report, record by record, at
`output/<command>/<experiment>/<model>/<language>/evaluation_<stamp>.html` — for
example `output/evaluate/pipeline/gpt-6.1-sol/java/evaluation_<stamp>.html`.

### Check the setup (one record, about 15 seconds)

Score one stored recovery:

```bash
secpat evaluate --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --recoveries-path dataset/experiment_outputs \
    --repo scality_osis
```

The results are written to `output/` (the last line of the run names the report):

- `output/index.html` — the report: every batch's scores, by batch and by pattern.
  For this record: Level 1 `0.857`, Level 2 `0.986`, Level 3 `1.000`, the paper's
  scores.
- `output/evaluate/pipeline/gpt-6.1-sol/java/evaluation_<stamp>.html` (and
  `.json`) — the batch's own report, record by record, with each level's details.

### Experiment 1: evaluation — score the paper's recoveries again

Takes each recovery the paper's models produced (stored in
`dataset/experiment_outputs/`) and scores it against the ground truth, on Levels
1–3.

The pipeline experiment with GPT, both languages (100 records, about 20 minutes):

```bash
secpat evaluate --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --recoveries-path dataset/experiment_outputs
```

Every experiment with GPT — pipeline, baseline and both ablations (about 1 hour):

```bash
secpat evaluate --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --recoveries-path dataset/experiment_outputs \
    --experiments all
```

The scores equal the ones stored in `dataset/experiment_outputs/<experiment>/<model>/<language>/evaluation.json`,
except for the baseline's 16 records whose prompt was over the input limit: they
were never run, so a fresh score leaves them out where the paper counts them as 0.
The baseline is scored on the interaction order its model stated, as in the paper.

### Experiment 2: interaction — derive the interaction order, then score

Runs the interaction detector on each of the paper's recoveries: it walks the
recovery's role mapping on the repository's CPG and saves the interaction order
it finds in the recovery's `recovery.json`. Then it scores the recovery as in
Experiment 1.

The pipeline experiment with GPT, both languages (100 records, about 25 minutes):

```bash
secpat interaction --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --recoveries-path dataset/experiment_outputs
```

Every experiment with GPT — pipeline, baseline and both ablations (about 1½ hours):

```bash
secpat interaction --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --recoveries-path dataset/experiment_outputs \
    --experiments all
```

The scores are the same as in Experiment 1. The derived orders are in
`output/interaction/<experiment>/<model>/<language>/<pattern>__<repo>/recovery.json`.
The baseline keeps the order its model stated, so it is only scored.

### Experiment 3: run the full pipeline

Runs the knowledge base, the agent and the interaction order with GPT, then
scores the result.

One repository — the pipeline on `scality_osis` (Java, `authentication_pwd`; 1
record, about 2 minutes and 70K input tokens):

```bash
secpat recover --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --repo scality_osis
```

The full dataset — the pipeline on all 100 records, both languages:

```bash
secpat recover --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns
```

Every experiment on the full dataset — pipeline, baseline and both ablations:

```bash
secpat recover --models gpt-6.1-sol \
    --manifests dataset/manifest_java_full_20261005.json \
    --manifests dataset/manifest_python_full_20261005.json \
    --groundtruth-path dataset/ground_truth \
    --cpg-path dataset/codebase/cpg \
    --repos-path dataset/codebase/repos \
    --patterns-path dataset/security_patterns \
    --experiments all
```

The baseline runs on the same two manifests. It counts each record's prompt
first (`o200k_base`) and skips any over the 900K-token input limit without
calling the model: the same 16 records the paper did not run. They get an
`error.txt` in place of a `recovery.json`.

To run other models, give `--models` for each as in
[Choosing models](#choosing-models), and set their keys. Before a run starts,
`secpat recover` checks every selected model's settings. If any are missing it
stops, names them, and suggests a `--models` list without those models. Model
outputs vary from run to run, so expect scores near the paper's, not equal to
them.

| Experiment | What runs |
| ---------- | --------- |
| `pipeline` | the full pipeline |
| `baseline` | one-shot: the whole repository packed into one prompt, one call, no agent |
| `ablation_no_grep` | the pipeline without the `grep` tool (keyword search) |
| `ablation_no_search` | the pipeline without the `search` tool (semantic search) |

### What reproduces which result

Experiments 1 and 2 recompute, without calling any model, the score of every
record: the scores stored in each `evaluation.json` of the dataset. The paper's
results are built from these scores:

| In the paper | Result | Built from |
| ------------ | ------ | ---------- |
| Table III | the groundtruth: 500 roles (334 present, 75 external, 91 missing) and 280 interactions | `dataset/ground_truth/`; the dataset's README says how the roles are counted |
| RQ1: Fig. 35, Fig. 4, Table VIII | RDA, MLS and IRC per LLM and language (over the four LLMs: 0.84, 0.84 and 0.66), the presence statuses against the groundtruth, and the instances with the highest MLS and with MLS = 1 | the `pipeline` scores |
| RQ3: Fig. 37, Table IX, Fig. 5, Fig. 40 | SecPatRec against the one-shot baseline per pattern, over the 84 instances both can run and by codebase size; IRC by MLS band | the `pipeline` and `baseline` scores |
| RQ4: the RDA, MLS and IRC columns of Table X | each setting without one search tool, against the full pipeline | the `pipeline`, `ablation_no_grep` and `ablation_no_search` scores |

The scripts that turn these scores into the paper's tables and figures are not
included. The dataset holds no token counts, steps, costs or memory use, so the
artifacts do not reproduce RQ2 (Fig. 36), the cost side of RQ3 (Figs. 38 and
39), or the token, step and memory columns of Table X. Experiment 3 measures them
again: each new `recovery.json` records the provider's token counts, with the
agent's steps under `run_stats` and the baseline's single call under `baseline`.
Model outputs vary from run to run, so its scores come out near the paper's, not
equal to them.

### Options

`secpat evaluate`, `secpat interaction` and `secpat recover --models` take the
same options. Give an option several times for several values, e.g.
`--models gpt-6.1-sol --models muse-spark-1.3-contributor`.

| Option | Default | |
| ------ | ------- | - |
| `--models` | `evaluate`, `interaction`: all four | `gpt-6.1-sol`, `deepseek-v4-flash-0731`, `glm-5.3-flash`, `muse-spark-1.3-contributor` (listed in `src/secpat_recovery/experiments/config.toml`) |
| `--experiments` | `pipeline` | `pipeline`, `baseline`, `ablation_no_grep`, `ablation_no_search`, or `all` |
| `--repo`, `--pattern` | every record | exact repo / pattern names |
| `--manifests`, `--groundtruth-path`, `--cpg-path`, `--repos-path`, `--patterns-path`, `--recoveries-path` | required | the dataset's parts, see [Dataset paths](#dataset-paths) |
| `--output-dir` | `output` | the folder every run writes into |

Everything runs in sequence: one batch (experiment × model × language) at a
time, and one record at a time within a batch.

## Other usage

### The `secpat` command

| Command | Does |
| ------- | ---- |
| `secpat interaction` | the paper's recoveries, their interaction order derived and scored (above); or, with `--manifest-path --batch-path`, the interaction order for one batch of recoveries |
| `secpat recover` | the experiments with `--models` (above); one repo (`<codebase> --pattern-file <04_pattern.json>`); or a batch (`--manifest-path --batch-path`) |
| `secpat evaluate` | the paper's recoveries, scored again (above); one recovery (`--prediction --groundtruth-path`); or a batch (`--manifest-path --batch-path --groundtruth-path`) |
| `secpat baseline` | the one-shot baseline, for one repo or a batch |
| `secpat report` | rebuild `output/index.html` |
| `secpat pack` | pack a repo the way the baseline does, or `--measure` its tokens (needs `ANTHROPIC_API_KEY`) |

`secpat <command> --help` lists every flag. `--provider` is `openai`, `muse`,
`deepseek` or `glm`; `--model` overrides the provider's default model;
`--disable-tool grep|search` with the arm's prompt from
`src/secpat_recovery/ablation/prompts/` gives an ablation arm. Batches and scoring
are described in `src/secpat_recovery/batch/README.md`.

### One repository or batch, by hand

Outside the experiments (`secpat evaluate`, `secpat interaction` and
`secpat recover --models`), `secpat` uses the Joern settings in your environment. Point it at the dataset's prebuilt CPGs:

```bash
export JOERN_CPG_ROOT="$PWD/dataset/codebase/cpg"
```

Recover one repository:

```bash
secpat recover dataset/codebase/repos/java/scality_osis \
    --pattern-file dataset/security_patterns/authentication_pwd/04_pattern.json \
    --provider openai --output-dir runs/scality_osis
```

The dataset's manifests list only each record's pattern, repo and language;
`secpat` finds the codebase and pattern from the dataset's layout. Recover a
batch:

```bash
secpat recover --manifest-path dataset/manifest_java_full_20261005.json --batch-path runs/java-gpt --provider openai --only scality_osis
```

Derive each recovery's interaction order from its role mapping:

```bash
secpat interaction --manifest-path dataset/manifest_java_full_20261005.json --batch-path runs/java-gpt --repo scality_osis
```

Score the batch (writes `evaluation_<stamp>.json` and `.html` into it):

```bash
secpat evaluate --manifest-path dataset/manifest_java_full_20261005.json --batch-path runs/java-gpt --groundtruth-path dataset/ground_truth/java
```

### Joern

The experiments (`secpat evaluate`, `secpat interaction` and
`secpat recover --models`) always use local Joern, on the CPGs in `--cpg-path`,
building any that are missing there. Otherwise `secpat` uses local Joern unless `JOERN_SERVICE_URL` points it at a
Joern query service; locally, it builds a missing CPG into `JOERN_CPG_ROOT`
(default `~/.cache/secpat/joern_cpg`). See the `secpat-interaction-order`
repository for details.

## License

MIT — see [LICENSE](LICENSE).
