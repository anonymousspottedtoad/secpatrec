"""Stage 3 of a recovery: the order in which the recovered roles interact.

The agent answers *what* implements each role. That is a set, not a sequence —
and the pattern it was given is a numbered walkthrough, so half of what a
recovery claims (this role runs before that one) is left underivable at the point
where every location needed to derive it has just been found. This stage closes
that: it walks the codebase's call graph from each role's cited entity and reads
the ordering out of the code, giving the run an end-to-end result rather than a
mapping a later evaluation has to complete.

It is deliberately *not* asked of the model. The agent's own claim about ordering
would be a claim about code it saw in fragments; a call graph is evidence, and
every interaction carries the file and line that evidence it.

The envelope is one fixed document shape, so every derived order — whatever
derived it — can be compared field for field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from secpat_interaction_order import detect as detect_interactions
from secpat_scoring.evaluation import scored_mappings

from .backends import detect_language, repo_key

logger = logging.getLogger(__name__)

#: Statuses that count as "the role was found in code". Mirrors
#: ``secpat_scoring.base.is_detected``: an `external` role is detected too — it
#: is fulfilled by a library the repo calls, and the call site is a location a
#: walk can reach.
_DETECTED = frozenset({"present", "external"})

#: Kotlin sources. `.kt` is indexed by a regex scan, not parsed, and the query
#: service has no Kotlin profile — a role cited there cannot be grounded or walked.
_KOTLIN_SUFFIXES = frozenset({".kt", ".kts"})


class KotlinCitation(Exception):
    """A recovered role cites a Kotlin file, which this pipeline does not support."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        cited = "; ".join(
            f"{r.get('patternRole')} -> {r.get('entity')!r} at {r.get('file')}:{r.get('lines')}"
            for r in rows
        )
        super().__init__(
            f"{len(rows)} role(s) cite a Kotlin file: {cited}. Kotlin is not a supported "
            "language — the struct index scans it by regex and the walk has no profile for it."
        )


def detect_action_sequence(
    codebase: Path,
    role_mappings: list[dict],
    pattern: dict | None,
    *,
    detect: bool = True,
    language: str | None = None,
) -> dict:
    """Detect the recovered roles' action sequence — their interaction order.

    Every failure *of the derivation* — no call graph, too few grounded roles, an
    unreachable service — comes back as an envelope with ``available: false`` and
    a ``reason`` naming it. The role mapping this runs behind cost a full agent
    run, so a stage-3 problem must not discard it; it is recorded beside it
    instead, where the run's own output says what stopped the order.

    A row the graph cannot locate is not a failure: the walk reads it as a library
    call, whatever its ``status`` says, and a ``present`` row that is really a
    library call is a status disagreement for Level 1 to score.

    Several roles cited on one library call are not a failure either: each is
    reached at that call, and no interaction is read between them.

    :class:`KotlinCitation` raises before anything else, ``--no-order`` included:
    a role cited in a ``.kt`` file is not a case this pipeline expects.

    ``detect=False`` skips the stage and says so in ``reason``. It used to be
    ``backend="none"``, one value of a flag that named which of three analyses to
    use. There is one analysis, so what is left is a yes/no, and it is spelled as
    one.
    """
    kotlin = [
        row for row in role_mappings or []
        if Path((row.get("file") or "").strip()).suffix.lower() in _KOTLIN_SUFFIXES
    ]
    if kotlin:
        raise KotlinCitation(kotlin)

    # Resolved once, here: `repo_key` reads the directory's own name off it, and
    # that name is what the query service resolves its checkout by.
    codebase = codebase.resolve()

    envelope: dict[str, Any] = {
        "available": False,
        "language": language or "",
        "repo": repo_key(codebase),
        "reason": "",
        "entryRole": "",
        "entryRule": "",
        "interactions": [],
        "anchoredRoles": [],
        "skippedRoles": [],
    }

    if not detect:
        envelope["reason"] = "action sequence detection was disabled (--no-order)."
        return envelope

    pattern = pattern or {}
    mappings = _index_roles(role_mappings)
    # Only the roles Level 3 scores are walked — see `scored_mappings`.
    scored = scored_mappings(pattern, mappings)
    walked = scored.mappings
    grounded = [role for role, row in walked.items() if _is_grounded(row)]
    if len(grounded) < 2:
        # Checked before anything is built: a Joern export takes minutes, and no
        # call graph can produce an interaction between fewer than two endpoints.
        envelope["skippedRoles"] = sorted(set(mappings) - set(grounded))
        envelope["reason"] = (
            f"{len(grounded)} role is grounded in code. An interaction runs between "
            "two roles, so at least two need a file and a location."
            if len(grounded) == 1
            else "No role is grounded in code — there is no location to walk from."
        )
        return envelope

    if not language:
        # Read off the grounded citations first — see detect_language: in a repo
        # with a front-end in another language, the roles say which one the walk
        # actually has to reach.
        language = detect_language(codebase, [mappings[role].get("file", "") for role in grounded])
        envelope["language"] = language
    if not language:
        envelope["reason"] = (
            f"no source language could be detected in {codebase} — the order is read "
            f"out of a call graph, and building one needs to know what to parse."
        )
        return envelope

    detection = detect_interactions(
        repo_key(codebase), language, walked, behaviour=scored.behaviour, repo_path=codebase
    )
    if not detection.available:
        envelope["reason"] = detection.reason
        logger.warning("interaction order unavailable: %s", detection.reason)
        return envelope

    anchored = dict(detection.anchored_roles)
    envelope.update(
        {
            "available": True,
            "entryRole": detection.entry_role,
            "entryRule": detection.entry_rule,
            "interactions": [
                {"rank": rank, **interaction.to_dict()}
                for rank, interaction in enumerate(detection.interactions, start=1)
            ],
            "anchoredRoles": [
                {"role": role, "kind": kind} for role, kind in sorted(anchored.items())
            ],
            "skippedRoles": sorted(set(mappings) - set(anchored)),
        }
    )

    if not detection.interactions:
        envelope["reason"] = (
            "The roles are grounded, but the call graph never runs from one of them "
            "into another — no interaction to order. Check that each role cites the "
            "function that does the work rather than the module or class around it."
        )

    logger.info(
        "interaction order: %d interaction(s) over %d anchored role(s) "
        "(entry: %s, %s)",
        len(detection.interactions), len(anchored),
        detection.entry_role or "none", detection.entry_rule or "n/a",
    )
    return envelope


def _index_roles(rows: list[dict]) -> dict[str, dict]:
    """Index role rows by ``patternRole``, first row winning on a duplicate.

    The same indexing ``secpat_scoring.evaluation.index_roles`` does, so the
    detector sees the mapping in the shape both sides of an evaluation see it.
    """
    indexed: dict[str, dict] = {}
    for row in rows or []:
        role = (row.get("patternRole") or "").strip()
        if role and role not in indexed:
            indexed[role] = row
    return indexed


def _is_grounded(row: dict) -> bool:
    """Whether a role row was found *and* says where — both are needed to walk."""
    status = (row.get("status") or "present").strip().lower()
    if status not in _DETECTED:
        return False
    return bool((row.get("file") or "").strip()) and bool(
        (row.get("entity") or "").strip() or (row.get("lines") or "").strip()
    )
