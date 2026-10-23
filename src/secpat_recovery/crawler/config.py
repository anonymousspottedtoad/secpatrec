from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# The KU Leuven / DistriNet security-pattern catalogue: (name slug, human-readable
# name, page URL). The name slug is the identifier callers use everywhere (--slug,
# output file name). Each URL is stored in full so it can be opened directly; the
# numeric id it contains is required by the site (the id-less path returns 404) and
# is never part of the identifier.
CATALOGUE: list[tuple[str, str, str]] = [
    (
        "authentication",
        "Authentication",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_001__authentication/",
    ),
    (
        "authentication_pwd",
        "Password-based authentication",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_002__authentication_pwd/",
    ),
    (
        "verifiable_token_based_authentication",
        "Verifiable token-based authentication",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_003__verifiable_token_based_authentication/",
    ),
    (
        "opaque_token_based_authentication",
        "Opaque token-based authentication",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_004__opaque_token_based_authentication/",
    ),
    (
        "obscure_token_based_access_control",
        "Obscure token-based access control",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_005__obscure_token_based_access_control/",
    ),
    (
        "session_based_access_control",
        "Session-based access control",
        "https://securitypatterns.distrinet-research.be/patterns/01_01_006__session_based_access_control/",
    ),
    (
        "authorisation",
        "Authorisation",
        "https://securitypatterns.distrinet-research.be/patterns/02_01_001__authorisation/",
    ),
    (
        "log_entity_actions",
        "Log entity actions",
        "https://securitypatterns.distrinet-research.be/patterns/02_02_001__log_entity_actions/",
    ),
    (
        "limit_request_rate",
        "Limit request rate",
        "https://securitypatterns.distrinet-research.be/patterns/02_03_001__limit_request_rate/",
    ),
    (
        "data_validation",
        "Data validation",
        "https://securitypatterns.distrinet-research.be/patterns/04_01_001__data_validation/",
    ),
    (
        "output_filter",
        "Output filter",
        "https://securitypatterns.distrinet-research.be/patterns/05_01_001__output_filter/",
    ),
    (
        "selective_encrypted_transmission",
        "Selective encrypted transmission",
        "https://securitypatterns.distrinet-research.be/patterns/06_01_001__selective_encrypted_transmission/",
    ),
    (
        "encrypted_tunnel",
        "Encrypted tunnel",
        "https://securitypatterns.distrinet-research.be/patterns/06_01_002__encrypted_tunnel/",
    ),
    (
        "verifiable_transmission",
        "Verifiable transmission",
        "https://securitypatterns.distrinet-research.be/patterns/06_02_001__verifiable_transmission/",
    ),
    (
        "selective_encrypted_storage",
        "Selective encrypted storage",
        "https://securitypatterns.distrinet-research.be/patterns/07_01_001__selective_encrypted_storage/",
    ),
    (
        "transparent_encrypted_storage",
        "Transparent encrypted storage",
        "https://securitypatterns.distrinet-research.be/patterns/07_01_002__transparent_encrypted_storage/",
    ),
    (
        "cryptographic_action",
        "Cryptographic action",
        "https://securitypatterns.distrinet-research.be/patterns/99_01_001__cryptographic_action/",
    ),
    (
        "encryption",
        "Encryption",
        "https://securitypatterns.distrinet-research.be/patterns/99_01_002__encryption/",
    ),
    (
        "digital_signature",
        "Digital signature",
        "https://securitypatterns.distrinet-research.be/patterns/99_01_003__digital_signature/",
    ),
    (
        "mac",
        "Message authentication code (MAC)",
        "https://securitypatterns.distrinet-research.be/patterns/99_01_004__mac/",
    ),
    (
        "crypto_key_management",
        "Cryptographic key management",
        "https://securitypatterns.distrinet-research.be/patterns/99_02_001__crypto_key_management/",
    ),
    (
        "crypto_as_a_service",
        "Cryptography as a service",
        "https://securitypatterns.distrinet-research.be/patterns/99_02_002__crypto_as_a_service/",
    ),
    (
        "crypto_self_managed",
        "Self-managed cryptography",
        "https://securitypatterns.distrinet-research.be/patterns/99_02_003__crypto_self_managed/",
    ),
]

SLUGS: list[str] = [name for name, _, _ in CATALOGUE]

# name slug -> page URL, e.g. "authentication_pwd" ->
# "https://.../patterns/01_01_002__authentication_pwd/"
NAME_TO_URL: dict[str, str] = {name: url for name, _, url in CATALOGUE}


def base_name(full_slug: str) -> str:
    """Drop the numeric id prefix: ``01_01_005__obscure_token`` -> ``obscure_token``."""
    return full_slug.split("__", 1)[-1]


def pattern_url(slug: str) -> str:
    """Resolve a catalogue name slug to its full page URL."""
    try:
        return NAME_TO_URL[slug]
    except KeyError:
        raise ValueError(
            f"Unknown pattern slug: {slug!r}. Known slugs: {', '.join(SLUGS)}"
        ) from None


@dataclass
class CrawlerConfig:
    """Which patterns to crawl and where to write the Mermaid output.

    The JSON config lists the whole catalogue under ``patterns``; each entry has
    an ``enabled`` flag showing which pattern(s) are currently "on". Explicit
    ``urls`` may also be supplied.
    """

    urls: list[str] = field(default_factory=list)
    slugs: list[str] = field(default_factory=list)
    output_dir: Path = Path("./reference_patterns")

    @classmethod
    def from_file(cls, path: Path) -> CrawlerConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # enabled catalogue entries — identified by their bare name slug
        slugs = [p["slug"] for p in data.get("patterns", []) if p.get("enabled")]
        # or a plain list of slugs
        slugs.extend(data.get("slugs", []))
        return cls(
            urls=list(data.get("urls", [])),
            slugs=slugs,
            output_dir=Path(data.get("output_dir", "./reference_patterns")),
        )

    def resolve_urls(self) -> list[str]:
        return resolve_urls(self.urls, self.slugs)


def resolve_urls(
    urls: list[str] | None = None,
    slugs: list[str] | None = None,
) -> list[str]:
    """Merge explicit URLs and catalogue slugs into a de-duplicated URL list."""
    resolved: list[str] = list(urls or [])
    resolved.extend(pattern_url(s) for s in (slugs or []))
    seen: set[str] = set()
    return [u for u in resolved if not (u in seen or seen.add(u))]


def random_slug(rng: random.Random | None = None) -> str:
    """Pick one random pattern slug from the catalogue."""
    return (rng or random).choice(SLUGS)
