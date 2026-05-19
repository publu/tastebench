"""tastebench.report — emit Markdown, JSON, the shareable card, and the
LLM-ready bundle.

Everything a reader sees is in the producer voice (see `identity.py`); the
numbers are evidence under the sentence, never the lede. `--llm` produces a
perfectly self-describing bundle (raw numbers + the FULL glossary + the
framing question + a schema note) so a CLI user can paste it into any model
and get an expert tutor with zero guessing.
"""

from __future__ import annotations

import json
from typing import Any

from . import explainers
from . import fingerprint as fp
from . import identity as idy

_FRAMING = (
    "You are a sharp, working music producer mentoring a peer. They gave "
    "tastebench a set of REFERENCE tracks they admire; the tool learned the "
    "shared 'taste signature' of that set and measured how their own DEMO "
    "diverges from it. Below is the complete raw analysis plus the full "
    "glossary defining every term (craft features, the 12 brain networks, "
    "ROI groups, edit types). Ground every claim ONLY in these numbers and "
    "definitions. Speak plainly and musically, like a friend at the desk — "
    "never a stat dump. Do NOT predict hits, streams, or chart outcomes: "
    "tastebench measures distance to a taste the user chose, not success "
    "(hit outcomes are irreducibly noisy — Salganik et al., Science 2006). "
    "Deliver, in this order: (1) the headline — is this on the taste or "
    "not, in one human sentence; (2) the single biggest lever and why it "
    "matters musically; (3) the 2-3 next highest-leverage edits with their "
    "trade-offs; (4) where the read is least trustworthy and why "
    "(low-confidence edits / loose reference set / brain layer absent)."
)

_SCHEMA_NOTE = {
    "taste_match": "headline 0-100; 100 = the demo already sounds like the "
    "reference set on every measured signal. Derived from overall_distance.",
    "overall_distance": "RMS of spread-normalized feature deltas; 0 = "
    "identical taste, grows unbounded. Lower is closer.",
    "spread_norm": "a single signal's deviation in units of the reference "
    "set's own consistency on that signal (so a signal the taste is tight "
    "on counts more than one it is loose on). |value| > ~1.5 is a real gap.",
    "voice": "the producer-language read of that row (the point; the numbers "
    "are the evidence under it).",
    "predicted_gain": "modeled drop in craft-distance if this one edit is "
    "applied — a hypothesis to A/B, not a guaranteed outcome.",
    "confidence": "high = song-bones edit on a tight reference set; "
    "downgraded for production-state features or a loose reference set.",
}


_BRAIN_CTA = (
    "brain layer off — run `python scripts/download_models.py` (~20 GB, "
    "one-time) to unlock your taste's neural fingerprint"
)

_BANDED_WARN = (
    "⚠ BRAIN ROI FALLBACK — upstream HCP labels were unavailable, so the "
    "360 ROIs are equal-width vertex bands and the 12-network split is "
    "geometrically approximate. Read the neural numbers as indicative "
    "only, not as a precise per-network map."
)


def _banded(payload: dict) -> bool:
    return payload.get("brain_roi_banded") is True


def to_json(payload: dict, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, default=str)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _matchbar(match) -> str:
    """A flat-text TASTE MATCH bar for markdown (no rich)."""
    if match is None:
        return "`TASTE MATCH  —`"
    fill = round(match / 100 * 24)
    return f"`TASTE MATCH  [{'█' * fill}{'·' * (24 - fill)}]  {match}%`"


# --------------------------------------------------------------------------
# Markdown — producer voice, fingerprint up top, numbers as evidence.
# --------------------------------------------------------------------------

def to_markdown(payload: dict) -> str:
    kind = payload.get("_kind", "report")
    out: list[str] = []

    if kind == "profile":
        p = payload
        out.append("# your taste signature\n")
        out.append(f"Learned from **{p['n_refs']}** references. "
                   f"Consistency **{_fmt(p.get('consistency'))}** — "
                   + ("a tight, opinionated taste; prescriptions will be "
                      "sharp." if (p.get("consistency") or 0) > 0.6 else
                      "a loose taste; treat the levers as directional, not "
                      "exact.") + "\n")
        out.append("**The set**\n")
        for r in p["refs"]:
            out.append(f"- `{r['name']}` "
                       f"(craft {'✓' if r['craft_ok'] else '—'} · "
                       f"brain {'✓' if r['brain_ok'] else '—'})")
        out.append("\n## What this taste is made of (craft centroid)\n")
        out.append("| signal | center | spread | refs |")
        out.append("|---|---:|---:|---:|")
        for k, v in p["centroid"].items():
            if k.startswith("net."):
                continue
            out.append(f"| {k} | {_fmt(v)} | {_fmt(p['spread'].get(k))} "
                       f"| {p['n'].get(k, 0)} |")
        netk = [k for k in p["centroid"] if k.startswith("net.")]
        if netk:
            out.append("\n## The neural side (12-network, hypothesis view)\n")
            if _banded(p):
                out.append(f"> {_BANDED_WARN}\n")
            out.append("| signal | center | spread |")
            out.append("|---|---:|---:|")
            for k in netk:
                out.append(f"| {k} | {_fmt(p['centroid'][k])} "
                           f"| {_fmt(p['spread'].get(k))} |")

    elif kind == "compare":
        c = payload
        m = c.get("taste_match")
        out.append(f"# {c['demo']['name']} vs your taste\n")
        out.append(f"## {idy.verdict_word(m).upper()} — {m}% taste match\n")
        out.append(f"> {c.get('headline', c['verdict'])}\n")
        out.append(_matchbar(m) + "\n")
        bl = c.get("biggest_lever")
        if bl:
            out.append(f"**The one thing:** {bl['line']}.\n")
        nr = c.get("nearest_reference")
        if nr:
            out.append(f"Closest to `{nr['name']}` of your set. "
                       f"Scored across {c['n_terms_compared']} signals.\n")
        if c["demo"].get("brain_note"):
            out.append(f"> _{_BRAIN_CTA}. This read is craft-only._\n")

        out.append("## Where it diverges (worst first)\n")
        for r in c["craft_deltas"]:
            if abs(r["spread_norm"]) < 0.4:
                continue
            out.append(
                f"- **{r['term'].replace('_', ' ')}** — {r['voice']}  "
                f"<sub>demo {_fmt(r['demo'])} · taste {_fmt(r['taste'])} · "
                f"{r['spread_norm']:+.1f}σ</sub>"
            )
        tight = [r for r in c["craft_deltas"] if abs(r["spread_norm"]) < 0.4]
        if tight:
            out.append("\n**Already on taste:** "
                       + ", ".join(r["term"].replace("_", " ")
                                   for r in tight) + ".")
        if c["brain_deltas"]:
            out.append("\n## The neural read (hypothesis view)\n")
            if _banded(c):
                out.append(f"> {_BANDED_WARN}\n")
            for r in c["brain_deltas"]:
                if not r["term"].endswith(".mean"):
                    continue
                out.append(
                    f"- **{r['term'].split('.')[1]}** — {r['voice']}  "
                    f"<sub>{r['spread_norm']:+.1f}σ</sub>"
                )

    elif kind == "optimize":
        o = payload
        m = o.get("taste_match")
        out.append(f"# push {o['demo']['name']} toward your taste\n")
        out.append(f"## {m}% taste match → here's how to move it\n")
        out.append(f"> {o.get('headline', '')}\n")
        out.append(_matchbar(m) + "\n")
        if o.get("the_one_move"):
            out.append(f"**Do this first:** {o['the_one_move']}\n")
        if not o["edits"]:
            out.append(f"_{o.get('note', 'no edits found')}_")
        else:
            out.append("## The edit list (ranked by what moves the dial)\n")
            for i, e in enumerate(o["edits"], 1):
                ma, mb = e.get("match_now"), e.get("match_after")
                swing = (f"  `{ma}% → {mb}%`"
                         if ma is not None and mb is not None else "")
                out.append(
                    f"**{i}. {e['voice']}** "
                    f"<sub>[{e['confidence']} confidence]</sub>{swing}"
                )
                if e.get("how_to_act"):
                    out.append(f"   - how: {e['how_to_act']}")
                out.append(
                    f"   - move `{e['feature']}` "
                    f"{_fmt(e['from'])} → {_fmt(e['to'])} "
                    f"(taste sits at {_fmt(e['toward_taste'])})")
                out.append(f"   - _{e['caveat']}_\n")
        out.append(f"\n> _{o.get('note', '')}_")

    else:
        out.append("```json")
        out.append(to_json(payload))
        out.append("```")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# The shareable taste-print card — a tasteful Unicode artifact people post.
# --------------------------------------------------------------------------

def to_card(payload: dict) -> str:
    """A clean, postable taste-print card (pure Unicode, paste-safe)."""
    kind = payload.get("_kind")
    if kind == "compare":
        name = payload["demo"]["name"]
        match = payload.get("taste_match")
        head = payload.get("headline", "")
        rows = fp.ascii_rows(payload)
        bl = payload.get("biggest_lever") or {}
    elif kind == "optimize":
        name = payload["demo"]["name"]
        match = payload.get("taste_match")
        head = payload.get("headline", "")
        rows = []
        bl = {"line": payload.get("the_one_move", "")}
    else:
        return "taste-print card is available for compare / vibe results."

    W = 60
    bar_fill = round((match or 0) / 100 * 30)
    bar = "█" * bar_fill + "·" * (30 - bar_fill)

    def line(s: str = "") -> str:
        return "│ " + s[:W].ljust(W) + " │"

    L = []
    L.append("╭" + "─" * (W + 2) + "╮")
    L.append(line("tastebench · an MRI for your taste"))
    L.append("├" + "─" * (W + 2) + "┤")
    L.append(line(f"{name}"))
    L.append(line(""))
    L.append(line(f"TASTE MATCH   {match}%" if match is not None
                  else "TASTE MATCH   —"))
    L.append(line(f"[{bar}]"))
    L.append(line(""))
    for seg in _wrap(head, W):
        L.append(line(seg))
    if rows:
        L.append(line(""))
        L.append(line("— taste fingerprint —"))
        for r in rows:
            L.append(line(r))
    if bl.get("line"):
        L.append(line(""))
        L.append(line("the one move:"))
        for seg in _wrap(bl["line"], W):
            L.append(line("  " + seg))
    L.append("├" + "─" * (W + 2) + "┤")
    L.append(line("◆ = your demo   █ on the taste   · off it"))
    L.append("╰" + "─" * (W + 2) + "╯")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# --------------------------------------------------------------------------
# The LLM bundle — self-describing, glossary-complete, one paste = a tutor.
# --------------------------------------------------------------------------

def to_llm_bundle(payload: dict) -> str:
    bundle = {
        "_about": (
            "This is a self-contained tastebench analysis bundle. It "
            "includes the raw analysis, a schema note explaining every key, "
            "and the FULL term glossary. You can answer the framing question "
            "from this alone — do not request anything else."
        ),
        "role_and_task": _FRAMING,
        "how_to_read_the_numbers": _SCHEMA_NOTE,
        "honesty_constraints": [
            "Measures distance to a user-chosen taste, not quality or success.",
            "Never predicts hits, streams, or chart outcomes.",
            "Edits are hypotheses to A/B, not guarantees.",
            "The brain layer is an unvalidated hypothesis view; flag it as "
            "such if it is present.",
        ],
        "analysis": payload,
        "glossary": explainers.entries(),
        "glossary_meta": explainers.meta(),
    }
    return to_json(bundle)


def to_verdict(payload: dict) -> str:
    """The one-screen `vibe` verdict: dial + headline + the single lever.

    Instant read for the fast path; `--deep` / `--fix` go deeper.
    """
    c = payload
    m = c.get("taste_match")
    name = c.get("demo", {}).get("name", "your demo")
    L = ["", f"  {idy.verdict_word(m).upper()} — {m}% taste match"
         f"   ({name})", ""]
    L.append("  " + _matchbar(m).strip("`"))
    head = c.get("headline") or c.get("verdict") or ""
    if head:
        L += ["", "  " + "\n  ".join(_wrap(head, 70))]
    bl = c.get("biggest_lever")
    if bl and bl.get("line"):
        L.append(f"\n  ▶ the one thing: {bl['line']}.")
    nr = c.get("nearest_reference")
    if nr:
        L.append(f"  · closest to {nr['name']} of your set")
    if c.get("demo", {}).get("brain_note"):
        L.append(f"  · {_BRAIN_CTA}")
    if _banded(c):
        L.append("  ⚠ brain ROI fallback active — neural read is "
                 "geometrically approximate (see --deep)")
    L += ["", "  → --deep for the full read   ·   --fix for the ranked "
          "edits", ""]
    return "\n".join(L)


def render(payload: dict, fmt: str = "markdown", llm: bool = False) -> str:
    """fmt in {markdown, json, card}; llm=True wraps the LLM bundle (JSON)."""
    if llm:
        return to_llm_bundle(payload)
    if fmt == "json":
        return to_json(payload)
    if fmt == "card":
        return to_card(payload)
    return to_markdown(payload)
