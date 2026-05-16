"""tribe_taste.report — emit JSON, Markdown, and an LLM-ready bundle.

`--llm` produces a self-contained bundle: the raw numbers + the FULL
explainer dictionary + a framing question, so a CLI user can paste it into
any LLM and get a deeper, grounded explanation without the model having to
guess what the features mean.
"""

from __future__ import annotations

import json
from typing import Any

from . import explainers

_FRAMING = (
    "You are helping a musician understand how their demo diverges from a "
    "taste they admire (the 'reference set'), and what to change. Below is "
    "the raw analysis from tribe-taste plus the full glossary of every "
    "term used (craft features, brain networks, ROI groups, edit types). "
    "Use ONLY these numbers and definitions. Be concrete and musical. Do "
    "not invent outcomes or streams; tribe-taste predicts taste-distance, "
    "not hits. Explain: (1) the single biggest divergence and why it "
    "matters, (2) the 2-3 highest-leverage edits and the trade-offs, "
    "(3) where the analysis is least confident and why."
)


def to_json(payload: dict, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, default=str)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def to_markdown(payload: dict) -> str:
    kind = payload.get("_kind", "report")
    out: list[str] = []

    if kind == "profile":
        p = payload
        out.append("# tribe-taste — taste profile\n")
        out.append(f"- references: **{p['n_refs']}**")
        out.append(f"- consistency: **{_fmt(p.get('consistency'))}** "
                   "(higher = tighter taste)")
        out.append(f"- layers: craft={p['layers']['craft']} "
                   f"brain={p['layers']['brain']}\n")
        out.append("## Reference set\n")
        for r in p["refs"]:
            out.append(f"- {r['name']} "
                       f"(craft={'ok' if r['craft_ok'] else '-'}, "
                       f"brain={'ok' if r['brain_ok'] else '-'})")
        out.append("\n## Taste centroid (craft)\n")
        out.append("| feature | centroid | spread | n |")
        out.append("|---|---:|---:|---:|")
        for k, v in p["centroid"].items():
            if k.startswith("net."):
                continue
            out.append(f"| {k} | {_fmt(v)} | {_fmt(p['spread'].get(k))} "
                       f"| {p['n'].get(k, 0)} |")
        netk = [k for k in p["centroid"] if k.startswith("net.")]
        if netk:
            out.append("\n## Taste centroid (brain, 12-network)\n")
            out.append("| signal | centroid | spread |")
            out.append("|---|---:|---:|")
            for k in netk:
                out.append(f"| {k} | {_fmt(p['centroid'][k])} "
                           f"| {_fmt(p['spread'].get(k))} |")

    elif kind == "compare":
        c = payload
        out.append("# tribe-taste — demo vs your taste\n")
        out.append(f"**{c['demo']['name']}**\n")
        out.append(f"- overall distance to taste: "
                   f"**{_fmt(c['overall_distance'])}** "
                   f"-> _{c['verdict']}_")
        nr = c.get("nearest_reference")
        if nr:
            out.append(f"- nearest reference: **{nr['name']}** "
                       f"(distance {_fmt(nr['distance'])})")
        out.append(f"- compared on {c['n_terms_compared']} signals "
                   f"({c['profile']['n_refs']} references)")
        if c["demo"].get("brain_note"):
            out.append(f"\n> brain layer skipped: {c['demo']['brain_note']}")
        out.append("\n## Craft divergence (biggest first)\n")
        out.append("| feature | demo | taste | delta | norm | tag | note |")
        out.append("|---|---:|---:|---:|---:|---|---|")
        for r in c["craft_deltas"]:
            out.append(
                f"| {r['term']} | {_fmt(r['demo'])} | {_fmt(r['taste'])} "
                f"| {_fmt(r['delta'])} | {_fmt(r['spread_norm'])} "
                f"| {r['kind']} | {r['plain']} |"
            )
        if c["brain_deltas"]:
            out.append("\n## Neural divergence (12-network, hypothesis view)\n")
            out.append("| signal | demo | taste | delta | norm | note |")
            out.append("|---|---:|---:|---:|---:|---|")
            for r in c["brain_deltas"]:
                out.append(
                    f"| {r['term']} | {_fmt(r['demo'])} | {_fmt(r['taste'])} "
                    f"| {_fmt(r['delta'])} | {_fmt(r['spread_norm'])} "
                    f"| {r['plain']} |"
                )

    elif kind == "optimize":
        o = payload
        out.append("# tribe-taste — ranked edits toward your taste\n")
        out.append(f"**{o['demo']['name']}**\n")
        out.append(f"- base craft distance: "
                   f"**{_fmt(o.get('base_craft_distance'))}**")
        out.append(f"- references: {o['profile']['n_refs']} "
                   f"(consistency {_fmt(o['profile']['consistency'])})\n")
        if not o["edits"]:
            out.append(f"_{o.get('note', 'no edits found')}_")
        else:
            out.append("| # | edit | feature | from | to | pred. gain "
                       "| confidence |")
            out.append("|--:|---|---|---:|---:|---:|---|")
            for i, e in enumerate(o["edits"], 1):
                out.append(
                    f"| {i} | {e['edit']} | {e['feature']} "
                    f"| {_fmt(e['from'])} | {_fmt(e['to'])} "
                    f"| {_fmt(e['predicted_gain'])} | {e['confidence']} |"
                )
            out.append("\n### Edits explained\n")
            for i, e in enumerate(o["edits"], 1):
                out.append(f"**{i}. {e['edit']}** — {e['plain']}")
                if e.get("how_to_act"):
                    out.append(f"  - how: {e['how_to_act']}")
                out.append(f"  - {e['caveat']}\n")
        out.append(f"\n> {o.get('note', '')}")

    else:
        out.append("```json")
        out.append(to_json(payload))
        out.append("```")

    return "\n".join(out) + "\n"


def to_llm_bundle(payload: dict) -> str:
    """Self-contained bundle: framing + raw numbers + FULL glossary."""
    bundle = {
        "framing_question": _FRAMING,
        "analysis": payload,
        "glossary": explainers.entries(),
        "glossary_meta": explainers.meta(),
    }
    return to_json(bundle)


def render(payload: dict, fmt: str = "markdown", llm: bool = False) -> str:
    """fmt in {markdown, json}; llm=True wraps the LLM bundle (JSON)."""
    if llm:
        return to_llm_bundle(payload)
    if fmt == "json":
        return to_json(payload)
    return to_markdown(payload)
