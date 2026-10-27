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
truth, prebuilt CPGs and every experiment's outputs) is on Hugging Face.

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

Download the dataset (replace `HF_REPO_ID` with the dataset's id on Hugging Face):

```bash
hf download HF_REPO_ID --repo-type dataset --local-dir dataset
```

Check it arrived:

```bash
ls dataset
```

`ls` should show `codebase/`, `experiment_outputs/`, `ground_truth/`,
`security_patterns/`, the two manifests and the dataset's own `README.md`, which
describes every file.

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
