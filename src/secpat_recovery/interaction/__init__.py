"""Interaction-order derivation — stage 3 of the recovery pipeline.

``detect_action_sequence`` turns the agent's role mapping into the ordering the
recovered roles run in, by walking the codebase's call graph from each role's
cited entity. The walk itself comes from the shared ``secpat_interaction_order``
package, not reimplemented here: scoring derives orders with the same detector,
and a second walker would make implementation drift read as recovery drift. This
package supplies the local facts a derivation needs — which language, under which
repo name, and over which codebase on disk.
"""

from __future__ import annotations

from .backends import detect_language, repo_key
from .order import KotlinCitation, detect_action_sequence

__all__ = [
    "KotlinCitation",
    "detect_action_sequence",
    "detect_language",
    "repo_key",
]
