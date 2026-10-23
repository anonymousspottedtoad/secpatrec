from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

_USER_AGENT = "secpat-recovery-crawler/0.1 (+https://securitypatterns.distrinet-research.be)"
_SKIP_TAGS = {"script", "style", "head", "noscript"}
_IMAGE_MEDIA_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
}


@dataclass
class PatternImage:
    url: str
    media_type: str
    data_b64: str


@dataclass
class PatternPage:
    url: str
    slug: str
    text: str
    links: list[str] = field(default_factory=list)  # hyperlinks: "anchor text -> href"
    diagram: PatternImage | None = None


@dataclass
class DiagramSubflow:
    """One sub-flow of a pattern diagram, identified by the shared prefix on its
    edge labels and named from the pattern description (see
    ``extractor.diagram_to_mermaid``)."""

    name: str
    prefix: str
    mermaid: str


_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


class _PageParser(HTMLParser):
    """Renders the page as Markdown, preserving the heading/list hierarchy.

    Also collects the page's hyperlinks and the pattern diagram image src.
    """

    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._page_url = page_url
        self._skip = 0
        self._drop = 0  # depth inside a dropped element (e.g. heading permalink)
        self._pre = 0  # depth inside <pre>/<code> — preserve whitespace, no wrapping
        self._href: str | None = None
        self._anchor: list[str] = []
        self._lists: list[list] = []  # stack of ["ul"|"ol", counter]
        self._buf: list[str] = []  # inline text for the current block
        self._kind = "p"  # block kind: "p" | "h" | "li" | "pre"
        self._prefix = ""  # markdown prefix for the current block (##, -, 1., …)
        self._blocks: list[tuple[str, str]] = []  # finalized (kind, text)
        self.links: list[str] = []
        self.diagram_src: str | None = None

    # -- block assembly -------------------------------------------------
    def _flush(self) -> None:
        text = "".join(self._buf)
        if self._pre:
            text = text.strip("\n")
        else:
            text = re.sub(r"\s+", " ", text).strip()
        if text:
            self._blocks.append((self._kind, self._prefix + text))
        self._buf = []
        self._prefix = ""
        self._kind = "p"

    def _list_prefix(self) -> str:
        depth = len(self._lists)
        indent = "  " * (depth - 1)
        marker = "- "
        if self._lists and self._lists[-1][0] == "ol":
            self._lists[-1][1] += 1
            marker = f"{self._lists[-1][1]}. "
        return indent + marker

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "img" and self.diagram_src is None:
            src = a.get("src") or ""
            # the pattern diagram lives under the site's own /img/ folder
            if "/img/" in src and "creativecommons" not in src:
                self.diagram_src = src
        elif tag == "a" and a.get("href"):
            # drop MkDocs "¶" heading permalinks — they add no content
            if "headerlink" in (a.get("class") or ""):
                self._drop += 1
            else:
                self._href = a["href"]
                self._anchor = []
        elif tag in _HEADINGS:
            self._flush()
            self._kind = "h"
            self._prefix = "#" * _HEADINGS[tag] + " "
        elif tag in ("ul", "ol"):
            self._flush()
            self._lists.append([tag, 0])
        elif tag == "li":
            self._flush()
            self._kind = "li"
            self._prefix = self._list_prefix()
        elif tag in ("p", "blockquote", "tr", "div", "section"):
            self._flush()
        elif tag == "pre":
            self._flush()
            self._pre += 1
            self._kind = "pre"
        elif tag == "code" and not self._pre:
            self._buf.append("`")  # inline code
        elif tag == "br":
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag == "a" and self._drop:
            self._drop -= 1
        elif tag == "a" and self._href is not None:
            text = re.sub(r"\s+", " ", "".join(self._anchor)).strip()
            href = urljoin(self._page_url, self._href)
            self.links.append(f"{text} -> {href}")
            self._buf.append(f"[{text}]({href})" if text else f"<{href}>")
            self._href = None
        elif tag in ("ul", "ol"):
            self._flush()
            if self._lists:
                self._lists.pop()
        elif tag in _HEADINGS or tag in ("li", "p", "blockquote", "tr", "div", "section"):
            self._flush()
        elif tag == "pre" and self._pre:
            self._flush()
            self._pre -= 1
        elif tag == "code" and not self._pre:
            self._buf.append("`")  # close inline code

    def handle_data(self, data: str) -> None:
        if self._skip or self._drop:
            return
        if self._href is not None:
            self._anchor.append(data)
        else:
            self._buf.append(data)

    def markdown(self) -> str:
        self._flush()
        lines: list[str] = []
        for kind, text in self._blocks:
            if kind == "pre":
                if lines:
                    lines.append("")
                lines.extend(["```", text, "```"])
            elif kind in ("h", "p"):
                if lines:
                    lines.append("")  # blank line before headings/paragraphs
                lines.append(text)
            else:  # list items stay tight
                lines.append(text)
        return "\n".join(lines).strip()


_FOOTER_RE = re.compile(
    r"^(#+\s+References\b|Last update:|Back to top\b|© Copyright\b|Made with \[)",
    re.IGNORECASE,
)


def clean_page_text(page: PatternPage) -> str:
    """The pattern's text, stripped of site chrome.

    Drops the navigation sidebar + per-page table of contents by starting at the
    page's H1 title, and drops the trailing references/citations block and site
    footer. No diagram — that lives in its own files (see ``render_full_diagram``
    and ``render_subflows``).
    """
    lines = page.text.splitlines()

    # Everything before the H1 title is nav chrome + the table of contents.
    start = next((i for i, line in enumerate(lines) if line.startswith("# ")), 0)

    # Everything from the References section / footer onward is irrelevant.
    end = next(
        (i for i in range(start, len(lines)) if _FOOTER_RE.match(lines[i])),
        len(lines),
    )

    return "\n".join(lines[start:end]).rstrip() + "\n"


def _page_title(page: PatternPage) -> str:
    """The page H1 text, falling back to the slug."""
    for line in page.text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return page.slug


def _safe_participant_id(display: str) -> str:
    """A Mermaid-valid participant id for a (possibly multi-word) display name."""
    parts = re.split(r"\W+", display)
    ident = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return ident or "P"


_DECL_RE = re.compile(r"^(\s*)(participant|actor)\s+(.+?)\s*$")


def sanitize_mermaid(src: str) -> str:
    """Make a Mermaid ``sequenceDiagram`` renderable when the vision model emits
    multi-word participant names (e.g. ``participant Password policy``), which
    Mermaid rejects. Each such name is aliased — ``participant PasswordPolicy as
    Password policy`` — and every reference to it (in arrows, notes, activations)
    is rewritten to the alias id. Non-sequence diagrams pass through unchanged.
    """
    lines = src.splitlines()
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if not first.startswith("sequenceDiagram"):
        return src

    # 1. Collect declared display names that aren't already valid identifiers.
    id_map: dict[str, str] = {}
    for line in lines:
        m = _DECL_RE.match(line)
        if not m or " as " in m.group(3):
            continue
        display = m.group(3).strip().strip('"')
        if not re.fullmatch(r"[A-Za-z0-9_]+", display):
            id_map[display] = _safe_participant_id(display)
    if not id_map:
        return src

    displays = sorted(id_map, key=len, reverse=True)  # longest first, avoid partials

    def _replace_refs(segment: str) -> str:
        for d in displays:
            segment = re.sub(rf"(?<!\w){re.escape(d)}(?!\w)", id_map[d], segment)
        return segment

    out: list[str] = []
    for line in lines:
        m = _DECL_RE.match(line)
        if m and " as " not in m.group(3):
            display = m.group(3).strip().strip('"')
            if display in id_map:
                out.append(f"{m.group(1)}{m.group(2)} {id_map[display]} as {display}")
                continue
        # Rewrite actor references, which sit before the first ':' on arrow/note
        # lines (the message text after ':' is left untouched).
        if ":" in line:
            head, tail = line.split(":", 1)
            out.append(_replace_refs(head) + ":" + tail)
        else:
            out.append(_replace_refs(line))
    return "\n".join(out)


def render_full_diagram(page: PatternPage, mermaid: str | None) -> str:
    """The full diagram as one Mermaid block, transcribed exhaustively from the
    source image (see ``extractor.diagram_to_full_mermaid``)."""
    md = f"# {_page_title(page)} — full diagram\n"
    if page.diagram is not None:
        md += f"\n![Pattern structure and behaviour diagram]({page.diagram.url})\n"
    if mermaid and mermaid.strip():
        md += f"\n```mermaid\n{sanitize_mermaid(mermaid.strip())}\n```\n"
    return md


def render_subflows(page: PatternPage, subflows: list[DiagramSubflow] | None) -> str:
    """The diagram split into one named Mermaid block per sub-flow, grouped by
    the shared edge prefix (see ``extractor.diagram_to_subflows``)."""
    md = f"# {_page_title(page)} — sub-flows\n"
    for sf in subflows or []:
        md += f"\n## {sf.name}\n"
        if sf.prefix:
            md += f"\n_Edge prefix: `{sf.prefix}`_\n"
        md += f"\n```mermaid\n{sanitize_mermaid(sf.mermaid.strip())}\n```\n"
    return md


def _http_get(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted catalogue host)
        return resp.read()


def _slug_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[-1] if parts else urlparse(url).netloc


def fetch_pattern(url: str, timeout: float = 20.0) -> PatternPage:
    """Fetch a pattern page: its text, its hyperlinks, and its diagram image."""
    html = _http_get(url, timeout).decode("utf-8", errors="replace")
    parser = _PageParser(url)
    parser.feed(html)

    diagram: PatternImage | None = None
    if parser.diagram_src:
        img_url = urljoin(url, parser.diagram_src)
        ext = urlparse(img_url).path.rsplit(".", 1)[-1].lower()
        media_type = _IMAGE_MEDIA_TYPES.get(ext)
        if media_type:
            try:
                raw = _http_get(img_url, timeout)
                diagram = PatternImage(
                    url=img_url,
                    media_type=media_type,
                    data_b64=base64.standard_b64encode(raw).decode("ascii"),
                )
            except Exception:  # noqa: BLE001 — a missing diagram must not abort the crawl
                diagram = None

    return PatternPage(
        url=url,
        slug=_slug_from_url(url),
        text=parser.markdown(),
        links=parser.links,
        diagram=diagram,
    )
