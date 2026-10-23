from __future__ import annotations

import json
from pathlib import Path


def write_json(result: dict, path: Path) -> None:
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _fmt_tool_call(tool: str, tool_input: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
    return f"{tool}({args})"


def _fmt_usage(usage: dict | None) -> list[str]:
    """The step's billed tokens as one line, or nothing when none were reported."""
    if not usage:
        return []
    return [
        f"**Tokens:** {usage.get('input_tokens', 0):,} in "
        f"({usage.get('cached_input_tokens', 0):,} cached) · "
        f"{usage.get('output_tokens', 0):,} out "
        f"({usage.get('reasoning_tokens', 0):,} reasoning)",
        "",
    ]


def write_transcript(transcript: list[dict], path: Path) -> None:
    """Render the recovery agent's ReAct loop (per-step thought, tool call,
    observation, and the final mapping) as a human-readable Markdown transcript.

    The system prompt and task are deliberately omitted — only the loop detail
    is recorded."""
    lines: list[str] = ["# Recovery Agent Transcript", ""]

    for entry in transcript:
        etype = entry.get("type")

        if etype == "prompt":
            lines += [
                f"- **Model:** {entry.get('model', '')}",
                f"- **KB size:** {entry.get('kb_size', '')} chunks",
            ]
            # Only an ablated run says anything about tools: with the full set
            # the line is noise, and without it an ablated transcript reads as an
            # agent that inexplicably never used the tool it did not have.
            if disabled := entry.get("disabled_tools"):
                lines += [
                    f"- **Tools:** {', '.join(entry.get('tools', []))}",
                    f"- **Ablation — disabled:** {', '.join(disabled)}",
                ]
            lines += [""]

        elif etype == "step":
            # One ReAct cycle: Thought → Action → Observation, always all three.
            lines += [
                f"## Step {entry.get('step')}",
                "",
                "**Thought:**",
                "",
                entry.get("thought", "").rstrip() or "_(none recorded)_",
                "",
                f"**Action:** `{_fmt_tool_call(entry.get('tool', ''), entry.get('input', {}))}`",
                "",
                *_fmt_usage(entry.get("usage")),
                "**Observation:**",
                "",
                "```",
                str(entry.get("observation", "")).rstrip(),
                "```",
                "",
            ]

        elif etype == "submit":
            lines += [
                f"## Step {entry.get('step')} — submit_mapping (final)",
                "",
                "**Thought:**",
                "",
                entry.get("thought", "").rstrip() or "_(none recorded)_",
                "",
                "**Action:** `submit_mapping(...)`",
                "",
                *_fmt_usage(entry.get("usage")),
                "**Result:**",
                "",
                "```json",
                json.dumps(entry.get("input", {}), indent=2),
                "```",
                "",
            ]

    path.write_text("\n".join(lines), encoding="utf-8")
