from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..llm.base import LLMProvider
from ..llm.openai import OpenAIProvider
from .config import base_name
from .extractor import (
    diagram_to_full_mermaid,
    select_main_subflow,
    split_subflows,
    synthesize_pattern,
)
from .fetcher import (
    DiagramSubflow,
    PatternPage,
    clean_page_text,
    fetch_pattern,
    render_full_diagram,
    render_subflows,
    sanitize_mermaid,
)

# The markdown/JSON files written into each pattern's reference-page folder.
TEXT_FILE = "01_text.md"  # the pattern's prose (site chrome stripped)
FULL_DIAGRAM_FILE = "02_diagram.md"  # the full diagram, transcribed from the image
SUBFLOWS_FILE = "03_subflows.md"  # the diagram split into named sub-flows
PATTERN_FILE = "04_pattern.json"  # final structured pattern (roles + directed interactions)

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    url: str
    name: str  # bare name slug (no id prefix) — also the output folder name
    page_dir: Path | None  # per-pattern folder holding the markdown files + JSON
    has_diagram: bool
    full_diagram: str | None  # complete Mermaid transcription of the whole image
    diagram_subflows: list[DiagramSubflow]  # one faithful Mermaid diagram per sub-flow
    pattern_json: dict | None  # final structured pattern (the `secpat recover` input)
    link_count: int


class PatternCrawler:
    """Fetch DistriNet security-pattern pages into per-pattern reference folders.

    Each pattern folder holds 01_text/02_diagram/03_subflows markdown plus
    04_pattern.json — the structured pattern consumed by ``secpat recover``.
    """

    # Every crawler stage (vision, split, select, synthesis) runs on this model.
    MODEL = "gpt-6-sol"
    # Reasoning effort for the final synthesis agent.
    SYNTHESIS_EFFORT = "medium"

    def __init__(
        self,
        page_dir: Path = Path("./security_patterns"),
        diagram_provider: LLMProvider | None = None,
        synthesis_provider: LLMProvider | None = None,
    ):
        self.page_dir = Path(page_dir)
        # OpenAI reads the diagram image and transcribes it to Mermaid. Built
        # lazily (only when a diagram is present) so no key is needed otherwise.
        self._diagram_provider = diagram_provider
        # The final synthesis agent runs on the same model with reasoning.
        self._synthesis_provider = synthesis_provider

    def _diagram_agent(self) -> LLMProvider:
        """The LLM agent that reads the diagram image → Mermaid (OpenAI vision)."""
        if self._diagram_provider is None:
            # Pinned: this provider also backs the two text-only stages (split /
            # select), independently of the recovery agent's model.
            self._diagram_provider = OpenAIProvider(model=self.MODEL, vision_model=self.MODEL)
        return self._diagram_provider

    def _synthesis_agent(self) -> LLMProvider:
        """The LLM agent that synthesizes the final pattern JSON (reasoning)."""
        if self._synthesis_provider is None:
            # 16k output cap, as before: reasoning tokens bill against it.
            self._synthesis_provider = OpenAIProvider(
                model=self.MODEL,
                max_tokens=16000,
                reasoning_effort=self.SYNTHESIS_EFFORT,
            )
        return self._synthesis_provider

    def crawl_url(self, url: str, write: bool = True) -> PatternResult:
        logger.info("[1/6] Fetching page: %s", url)
        page: PatternPage = fetch_pattern(url)
        name = base_name(page.slug)  # id-prefix stripped, used as the folder name
        logger.info(
            "      fetched '%s' — %d chars, %d links, diagram=%s",
            page.slug,
            len(page.text),
            len(page.links),
            "yes" if page.diagram is not None else "no",
        )

        # ONE vision (image) call turns the diagram into the full Mermaid
        # flowchart; sub-flow splitting and main-flow selection are text-only
        # tasks over that flowchart. Best-effort — a failure must not abort.
        full_diagram: str | None = None
        diagram_subflows: list[DiagramSubflow] = []
        if page.diagram is None:
            logger.info("[2-4/6] No diagram on page — skipping transcription/split/select")
        else:
            reader = self._diagram_agent()
            logger.info("[2/6] Transcribing diagram image → flowchart (%s)", reader.vision_model_id)
            try:
                full_diagram = diagram_to_full_mermaid(reader, page)
                logger.info("      flowchart: %d lines", len((full_diagram or "").splitlines()))
            except Exception:  # noqa: BLE001 — diagram transcription is best-effort
                logger.exception("      full-diagram transcription failed")
                full_diagram = None

            logger.info("[3/6] Splitting flowchart into sub-flows (text)")
            try:
                diagram_subflows = split_subflows(reader, page, full_diagram)
                logger.info(
                    "      %d sub-flow(s): %s",
                    len(diagram_subflows),
                    ", ".join(sf.name for sf in diagram_subflows) or "(none)",
                )
            except Exception:  # noqa: BLE001 — sub-flow split is best-effort
                logger.exception("      sub-flow split failed")
                diagram_subflows = []

            # With several sub-flows, a selector agent keeps only the MAIN flow
            # described in the pattern text; a single sub-flow is kept as-is.
            if len(diagram_subflows) > 1:
                logger.info("[4/6] Selecting main flow from %d candidates", len(diagram_subflows))
                try:
                    main = select_main_subflow(reader, page, diagram_subflows)
                    if main is not None:
                        logger.info("      main flow: '%s' (prefix `%s`)", main.name, main.prefix)
                        diagram_subflows = [main]
                except Exception:  # noqa: BLE001 — selection is best-effort
                    logger.exception("      main-flow selection failed")
            elif diagram_subflows:
                logger.info(
                    "[4/6] Single sub-flow — kept as main flow: '%s'", diagram_subflows[0].name
                )

        # Final synthesis: derive the canonical structured pattern (roles and
        # the directed request+response interaction walkthrough) as JSON — the
        # pattern input consumed by `secpat recover`. The main-flow Mermaid is
        # fed in only to recover each step's direction + concrete labels; it is
        # not part of the output. Text-only; best-effort so a failure never aborts.
        main_flow = diagram_subflows[0].mermaid if diagram_subflows else full_diagram
        main_flow = sanitize_mermaid(main_flow.strip()) if main_flow else ""
        synthesizer = self._synthesis_agent()
        logger.info("[5/6] Synthesizing final pattern JSON (%s, thinking)", synthesizer.model_id)
        pattern_json: dict | None = None
        try:
            pattern_json = synthesize_pattern(synthesizer, page, main_flow)
            logger.info(
                "      %d roles, %d interactions",
                len(pattern_json.get("roles", [])),
                len(pattern_json.get("behaviour", [])),
            )
        except Exception:  # noqa: BLE001 — synthesis is best-effort
            logger.exception("      pattern synthesis failed")
            pattern_json = None

        page_dir: Path | None = None
        if write:
            logger.info("[6/6] Writing outputs")
            # One folder per pattern, holding the markdown files + final JSON.
            page_dir = self.page_dir / name
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / TEXT_FILE).write_text(clean_page_text(page), encoding="utf-8")
            (page_dir / FULL_DIAGRAM_FILE).write_text(
                render_full_diagram(page, full_diagram), encoding="utf-8"
            )
            (page_dir / SUBFLOWS_FILE).write_text(
                render_subflows(page, diagram_subflows), encoding="utf-8"
            )
            if pattern_json is not None:
                (page_dir / PATTERN_FILE).write_text(
                    json.dumps(pattern_json, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            logger.info(
                "      wrote %s/ (01_text, 02_diagram, 03_subflows, 04_pattern.json)", page_dir
            )
        else:
            logger.info("[6/6] write=False — skipping file output")

        return PatternResult(
            url=url,
            name=name,
            page_dir=page_dir,
            has_diagram=page.diagram is not None,
            full_diagram=full_diagram,
            diagram_subflows=diagram_subflows,
            pattern_json=pattern_json,
            link_count=len(page.links),
        )

    def crawl(self, urls: list[str], write: bool = True) -> list[PatternResult]:
        results: list[PatternResult] = []
        total = len(urls)
        for i, u in enumerate(urls, 1):
            logger.info("=== Pattern %d/%d ===", i, total)
            results.append(self.crawl_url(u, write=write))
        return results
