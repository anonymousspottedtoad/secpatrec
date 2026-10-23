from __future__ import annotations

from .config import CATALOGUE, SLUGS, CrawlerConfig, base_name, resolve_urls
from .fetcher import PatternImage, PatternPage, fetch_pattern
from .pipeline import PatternCrawler, PatternResult

__all__ = [
    "CATALOGUE",
    "SLUGS",
    "CrawlerConfig",
    "base_name",
    "resolve_urls",
    "PatternImage",
    "PatternPage",
    "fetch_pattern",
    "PatternCrawler",
    "PatternResult",
]
