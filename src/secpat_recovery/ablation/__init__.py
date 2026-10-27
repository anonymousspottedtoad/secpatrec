"""The tool-ablation experiment: the full pipeline, run with one tool withheld.

``prompts/<arm>.md`` is each arm's copy of the canonical system prompt with the
withheld tool edited out: ``no-grep.md`` (no exact text search over the repo)
and ``no-search.md`` (no semantic search over the knowledge base).

The mechanism the arms use stays with the agent, in ``recovery_agent``:
``build_tools(disabled)`` and ``ABLATABLE_TOOLS``, reached through
``secpat recover --disable-tool`` and ``--system-prompt-file``. A run is an
ablation only because of the flags it is given:

    secpat recover --manifest-path <m.json> --batch-path <dir> --provider openai \\
        --disable-tool grep --system-prompt-file src/secpat_recovery/ablation/prompts/no-grep.md
"""
