"""The tool-ablation experiment: the full pipeline, run with one tool withheld.

Everything an ablation needs beyond the agent itself lives here:

- ``run.sh`` (``secpat ablation run``) runs one manifest once per arm, scores the
  arms and the control, and writes the comparison (see its header for flags and
  layout).
- ``prompts/<arm>.md`` is each arm's copy of the canonical system prompt with the
  withheld tool edited out; ``generate_prompts.py`` writes them and ``--check``
  verifies they are current.
- ``report.py`` puts the arms' scores and step counts side by side.

The mechanism the arms use stays with the agent, in ``recovery_agent``:
``build_tools(disabled)`` and ``ABLATABLE_TOOLS``, reached through
``secpat recover --disable-tool`` and ``--system-prompt-file``. A run is an
ablation only because of the flags it is given, so the agent has to accept them
whether or not this folder exists.

    secpat ablation run --manifest-path <m.json> --groundtruth-path <snapshot> \
        --output-dir <dir> --control <batch> --dry-run
    secpat ablation prompts --check
    secpat ablation report --output-root <dir> --tag <tag>
"""
