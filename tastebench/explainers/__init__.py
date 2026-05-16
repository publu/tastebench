"""tastebench.explainers — the explainer dictionary + accessor.

The dictionary (`explainers.json`) is a first-class deliverable: one entry
per craft feature, brain network, key brain-ROI group, and edit type. Every
`compare` / `optimize` line attaches its matching entry, and
`report --llm` embeds the whole thing so a CLI user can paste it into any
LLM for a deeper explanation.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

_PATH = os.path.join(os.path.dirname(__file__), "explainers.json")


@lru_cache(maxsize=1)
def load_all() -> dict:
    """The full explainer dictionary (including the `_meta` block)."""
    with open(_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def entries() -> dict:
    """All explainer entries (without the `_meta` block)."""
    return {k: v for k, v in load_all().items() if k != "_meta"}


def meta() -> dict:
    return load_all().get("_meta", {})


def _candidates(term: str):
    """Yield lookup keys to try, tolerating edit/network naming variants."""
    yield term
    yield f"edit.{term}"
    yield f"net.{term}"
    # strip a trailing ".mean" / ".reliability" suffix from flat vector keys
    if "." in term:
        head = term.split(".")
        if head[0] in ("net", "edit") and len(head) > 1:
            yield head[1]
        yield head[0]


def get_explainer(term: str) -> Optional[dict]:
    """Return the explainer entry for a term, or None if unknown.

    Tolerant of the prefixed keys used internally (e.g. ``net.Auditory.mean``
    -> ``Auditory``; ``shorten_intro`` -> ``edit.shorten_intro``).
    """
    table = entries()
    for key in _candidates(term):
        if key in table:
            return table[key]
    return None


def explain_line(term: str, fallback: str = "") -> str:
    """One-line plain-language explainer for a term (for report rows)."""
    e = get_explainer(term)
    if e:
        return e.get("plain", fallback)
    return fallback


def by_kind(kind: str) -> dict:
    """All entries of a given kind: craft | brain_network | brain_roi | edit."""
    return {k: v for k, v in entries().items() if v.get("kind") == kind}


def glossary_text(term: Optional[str] = None) -> str:
    """Human-readable glossary. Full dictionary, or one detailed entry."""
    if term:
        e = get_explainer(term)
        if not e:
            known = ", ".join(sorted(entries()))
            return f"Unknown term: {term!r}\nKnown terms: {known}"
        lines = [
            f"{e.get('term', term)}  [{e.get('kind', '?')}]",
            "=" * 60,
            e.get("plain", ""),
            "",
            e.get("detail", ""),
            "",
            f"How computed : {e.get('how_computed', '-')}",
            f"Units / range: {e.get('units_range', '-')}",
        ]
        if e.get("how_to_act"):
            lines.append(f"How to act   : {e['how_to_act']}")
        if e.get("refs"):
            lines.append(f"Refs         : {e['refs']}")
        return "\n".join(lines)

    out = []
    order = ["craft", "brain_network", "brain_roi", "edit"]
    titles = {
        "craft": "CRAFT FEATURES",
        "brain_network": "BRAIN NETWORKS (12-network signature)",
        "brain_roi": "BRAIN ROI / DERIVED",
        "edit": "EDIT TYPES",
    }
    for kind in order:
        group = by_kind(kind)
        if not group:
            continue
        out.append(f"\n{titles[kind]}\n" + "-" * 60)
        for k in sorted(group):
            e = group[k]
            out.append(f"  {e.get('term', k):<22} {e.get('plain', '')}")
    return "\n".join(out).strip()
