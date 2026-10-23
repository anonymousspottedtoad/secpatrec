"""One-shot baseline: pack a whole repo into one prompt, recover the pattern in one API call.

    secpat baseline codebase/CentOS_duffy \
        -p security_patterns/obscure_token_based_access_control/04_pattern.json \
        -o /tmp/duffy-baseline

    secpat baseline --manifest-path <manifest.json> --batch-path <batch folder>
    secpat baseline --manifest-path ... --batch-path ... --smallest 5 --dry-run
    secpat baseline --manifest-path ... --batch-path ... --smallest 5

Independent of `secpat recover` in *method* — no knowledge base, no embeddings,
no ReAct loop, no tools, no interaction-order stage; one pack, one message, one
response. It shares the pipeline's transport on purpose: the call goes through
`llm.make_agent`, the same `--provider` names on the same model tags, because a
baseline run on a different model measures the model, not the method.

It lives beside `recovery_agent/` and imports none of it: `pack.py` builds the
prompt's repository half from `codebase_loader`'s file set, `recover.py` makes
the call through `llm`, and nothing else in the package is reached.

There is exactly one prompt (`SYSTEM`): the instruction and the output schema,
written out as the shape of the JSON object to return. The model answers in
plain text through `LLMProvider.response`, and `parse_answer` reads the JSON back
out. No tool is declared and no provider-specific structured-output parameter is
set — the baseline is then the simplest thing the study can compare against, and
it is the *same* simplest thing on every provider (Anthropic, OpenAI and the
self-hosted runtimes each constrain output differently, or not at all).

The field wording matches the agent's `submit_mapping` so both pipelines are
scored against the same vocabulary — a bare entity name, a single declaration
line — but the schema is written out here rather than imported, because a
baseline that imports the pipeline it is a baseline for stops being independent
the first time the pipeline's contract changes.

Batch mode writes `<batch-path>/<pattern>__<repo>/`, the layout `secpat evaluate`
already scores:

    secpat evaluate --batch-path <batch folder> --manifest-path <manifest.json> \
        --groundtruth-path <snapshot folder>

The same answer carries the interaction order, in the envelope `secpat recover`
writes, so the two runs diff field for field. It is the model's *claim* about
the order, not a call-graph walk — `backend: "model"`, `mechanism: "model-claim"`
— and by default Level 3 does not read it: `secpat evaluate` derives the
order it scores from `role_mappings` itself, for the baseline exactly as for the
pipeline. `secpat evaluate --recovery-order stored` scores the claimed order
instead.

A model that answers with something other than the schema fails the record; it is
never retried on another model, since a row silently answered by a different
model would not be this baseline's result.
"""
