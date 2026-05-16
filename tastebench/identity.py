"""tastebench.identity — the brand: one palette, one motif, one voice.

Everything a user sees routes through here so the CLI, the TUI, the report,
and the shareable card feel like one designed product instead of a stack of
scripts. Three things live here:

  * THE SPECTRUM  — a single far->close colour ramp + named accents. The
    only colours allowed anywhere user-facing.
  * THE MOTIF     — the "taste fingerprint": a labelled horizontal heat-bar
    per signal with the demo overlaid as a divergent ghost. This is the
    signature visual; it shows up identically in the TUI, the report, and
    the share card.
  * THE VOICE     — a sharp-producer-friend phrasebook that turns raw
    spread-normalized deltas into one evocative line, leaning on the
    explainer dictionary for the musical "why".

No analysis logic here — pure presentation. Import-cheap (no rich at module
load) so the package still imports clean with nothing installed.
"""

from __future__ import annotations

import math
from typing import Optional

# --------------------------------------------------------------------------
# THE SPECTRUM — the only palette. Far (you're off the taste) = molten red;
# close (you're in it) = cool mint. Named so call sites read like intent.
# --------------------------------------------------------------------------

SPECTRUM = [
    "#ff3b5c",  # 0  far / molten
    "#ff5a3c",
    "#ff7a30",
    "#ff9e1f",
    "#ffc400",
    "#e8d400",
    "#a9dd1e",
    "#5fd17a",
    "#22c4a6",  # 8  close / mint
]

INK = "#e7e2d8"        # primary text on dark
DIM = "#8a8577"        # secondary text
FAINT = "#4d4a42"      # rails / unfilled track
GOLD = "#ffc400"       # the brand accent (headline, the dial needle)
GHOST = "#9b8cff"       # the demo "ghost" overlay — always this violet
RULE = "grey37"

# semantic colours, all sampled from THE SPECTRUM so nothing clashes
HOT = SPECTRUM[0]
WARM = SPECTRUM[3]
COOL = SPECTRUM[-1]

WORDMARK = "𝚝𝚛𝚒𝚋𝚎-𝚝𝚊𝚜𝚝𝚎"
TAGLINE = "an MRI for your taste"

# the motif glyphs — kept in one place so every surface draws it the same
BLOCK = "█"
HALF = "▌"
TRACK = "─"
GHOST_MARK = "◆"
PEAK = "▲"


def ramp(frac: float) -> str:
    """frac in [0,1]; 0 = far (hot), 1 = close (cool). Hex from THE SPECTRUM."""
    if frac != frac:  # NaN
        frac = 0.0
    frac = max(0.0, min(1.0, frac))
    return SPECTRUM[round(frac * (len(SPECTRUM) - 1))]


def closeness(norm: float) -> float:
    """A spread-normalized deviation -> a 0..1 closeness (1 = dead on taste).

    Smooth, saturating: tiny deviations stay near 1, ~1.5 sigma is the
    knee, far-off saturates near 0. Used by every gauge/bar/score so the
    feel is consistent.
    """
    a = abs(norm or 0.0)
    return 1.0 / (1.0 + (a / 1.5) ** 1.35)


def resonance(distance: Optional[float]) -> Optional[int]:
    """Overall taste-distance -> the headline TASTE MATCH %, 0..100.

    Distance is an RMS of spread-normalized deltas (0 = identical taste).
    This is the one number the whole product orbits, so it gets its own
    deliberately forgiving curve: being *in* a taste shouldn't require
    being a clone of it.
    """
    if distance is None:
        return None
    c = 1.0 / (1.0 + (max(0.0, distance) / 2.6) ** 1.15)
    return int(round(100 * c))


# --------------------------------------------------------------------------
# THE VOICE — sharp producer friend. Plain, musical, a little opinionated.
# Never a stat dump; the number is evidence, the sentence is the point.
# --------------------------------------------------------------------------

# headline read on the whole demo, keyed by TASTE MATCH %
def verdict_line(match: Optional[int]) -> str:
    if match is None:
        return "Nothing comparable to score yet — feed it audio."
    if match >= 88:
        return "This is in the pocket. It already sounds like the refs you love."
    if match >= 72:
        return "You're in the taste. A couple of moves and it's undeniable."
    if match >= 55:
        return "The bones are right but it's drifting — there's real signal to chase."
    if match >= 35:
        return "It's off your taste in ways you can feel. Good news: they're fixable."
    return "This is a different record than the ones you picked. Big levers below."


def verdict_word(match: Optional[int]) -> str:
    if match is None:
        return "no read"
    if match >= 88:
        return "in the pocket"
    if match >= 72:
        return "on taste"
    if match >= 55:
        return "drifting"
    if match >= 35:
        return "off taste"
    return "different record"


# direction-aware phrasing per craft feature. {0} = the demo's own value,
# rendered for humans. "hi" = demo is above the taste, "lo" = below.
_FEATURE_VOICE = {
    "time_to_hook": {
        "hi": "your hook shows up at {v} — the refs land theirs way sooner; "
              "most listeners are gone before your best moment",
        "lo": "you hit the hook fast ({v}) — even quicker than the refs; "
              "that's their instinct, keep it",
    },
    "intro_length": {
        "hi": "you idle for {v} before anything happens; the refs don't make "
              "people wait",
        "lo": "you get moving in {v} — tighter than the refs, that's their DNA",
    },
    "chorus_lift_db": {
        "hi": "your sections swing {v} — even more contrast than the refs; "
              "huge dynamic punch",
        "lo": "only {v} between your loudest and quietest — the refs lift "
              "harder; the chorus doesn't pay off",
    },
    "loopability": {
        "hi": "you loop tighter than the refs ({v}) — it'd live forever in a feed",
        "lo": "the harmony wanders ({v}) — it doesn't loop clean like the refs do",
    },
    "key_stability": {
        "hi": "you stay home key harder than the refs ({v}) — rock solid",
        "lo": "the tonal center keeps slipping ({v}); the refs commit and "
              "you don't",
    },
    "tempo": {
        "hi": "you're running {v} BPM — sitting above the refs' pocket",
        "lo": "you're at {v} BPM — under the refs' tempo lane",
    },
    "tempo_stability": {
        "hi": "your pocket is steadier than the refs ({v}) — metronomic",
        "lo": "the tempo drifts ({v}); the refs hold a tighter pocket",
    },
    "brightness": {
        "hi": "you're brighter than the refs ({v} Hz) — more air up top",
        "lo": "you're darker than the refs ({v} Hz) — missing their top-end air",
    },
    "dynamic_range_db": {
        "hi": "your loudness breathes more than the refs ({v}) — lots of room",
        "lo": "you're squashed flat ({v}); the refs let it breathe",
    },
    "hook_density_per_min": {
        "hi": "you throw hooks faster than the refs ({v}/min) — busy",
        "lo": "not much grabs the ear ({v}/min); the refs keep more moments live",
    },
    "loop": {"hi": "", "lo": ""},
}

# how to render each feature's value in human units for the voice line
_UNIT = {
    "time_to_hook": lambda v: f"{v:.1f}s",
    "intro_length": lambda v: f"{v:.1f}s",
    "chorus_lift_db": lambda v: f"{v:.0f} dB",
    "loopability": lambda v: f"{v:.2f}",
    "key_stability": lambda v: f"{v:.2f}",
    "tempo": lambda v: f"{v:.0f}",
    "tempo_stability": lambda v: f"{v:.2f}",
    "brightness": lambda v: f"{v:.0f}",
    "dynamic_range_db": lambda v: f"{v:.0f} dB",
    "hook_density_per_min": lambda v: f"{v:.1f}",
    "voiced_fraction": lambda v: f"{v:.0%}",
    "f0_range_octaves": lambda v: f"{v:.2f} oct",
    "flatness": lambda v: f"{v:.3f}",
}


def humanize(feature: str, value) -> str:
    fn = _UNIT.get(feature)
    if fn is None or value is None:
        return f"{value}"
    try:
        return fn(float(value))
    except (TypeError, ValueError):
        return f"{value}"


def feature_line(feature: str, demo_value, delta: float, explainer: dict | None) -> str:
    """One producer sentence for a single craft divergence."""
    side = "hi" if (delta or 0.0) > 0 else "lo"
    table = _FEATURE_VOICE.get(feature)
    if table and table.get(side):
        return table[side].format(v=humanize(feature, demo_value))
    # fall back to the explainer's plain line, still conversational
    plain = (explainer or {}).get("plain", feature.replace("_", " "))
    return plain.rstrip(".").lower()


def brain_line(network: str, delta: float, explainer: dict | None) -> str:
    """One producer sentence for a brain-network divergence."""
    tag = (explainer or {}).get("plain", network).rstrip(".").lower()
    if abs(delta or 0.0) < 1e-6:
        return f"{tag} — right on the refs"
    more = "stronger than" if delta > 0 else "weaker than"
    return f"{tag} — {more} the refs"


def edit_line(edit: dict) -> str:
    """One imperative producer sentence for a ranked edit."""
    ex = edit.get("explainer") or {}
    plain = ex.get("plain") or edit.get("plain") or edit["edit"].replace("_", " ")
    return plain.rstrip(".")


def take(match: Optional[int], top_edit: Optional[dict]) -> str:
    """The single-screen 'here's the one move' line for `vibe`."""
    if not top_edit:
        if match is None:
            return "Feed it audio refs and a demo to get a read."
        return "No single edit jumps out — it's already close. Trust your ears."
    ex = top_edit.get("explainer") or {}
    act = ex.get("how_to_act") or top_edit.get("how_to_act") or ""
    lever = edit_line(top_edit)
    if act:
        return f"{lever}. {act.rstrip('.')}."
    return f"{lever}."
