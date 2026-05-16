"""tribe_taste.fingerprint — THE signature visual.

The "taste fingerprint": a stack of labelled horizontal heat-bars, one per
signal. Each bar's length + colour encodes how close the demo sits to the
taste on that signal (full + mint = dead on; short + molten = far off). The
demo is overlaid as a violet ghost diamond exactly where it lands, so the
gap reads at a glance — an MRI of the taste, not a table.

This identical motif renders in three places:
  * the TUI hero panel,
  * the `vibe` one-screen verdict,
  * the shareable taste-print card (a pure-Unicode variant, no rich).

It works on whichever layer is present: the 12 brain networks when the
model is installed, the craft "song-bones" when it isn't. Same motif, so
the product looks like one thing.
"""

from __future__ import annotations

from typing import Optional

from . import identity as idy

# fixed display order + short labels so the fingerprint looks the same every
# run (the brain layer comes back unordered).
_NET_ORDER = [
    ("Auditory", "auditory"),
    ("Orbito-Affective", "reward"),
    ("Default-Mode", "memory / stick"),
    ("Language", "lyric pull"),
    ("Somatomotor", "makes you move"),
    ("Cingulo-Opercular", "grabs attention"),
    ("Dorsal-Attention", "steers focus"),
    ("Frontoparietal", "mental effort"),
    ("Visual1", "visual (low)"),
    ("Visual2", "visual (scene)"),
    ("Posterior-Multimodal", "binds senses"),
    ("Ventral-Multimodal", "meaning"),
]

_CRAFT_ORDER = [
    ("time_to_hook", "time to hook"),
    ("intro_length", "intro drag"),
    ("chorus_lift_db", "chorus lift"),
    ("hook_density_per_min", "hook density"),
    ("loopability", "loopability"),
    ("key_stability", "key commit"),
    ("tempo", "tempo"),
    ("tempo_stability", "tempo lock"),
    ("brightness", "brightness"),
    ("dynamic_range_db", "dynamics"),
]

BAR_W = 34


def _rows_from_compare(cmp: dict):
    """(label, norm, demo, taste) rows in the fixed motif order.

    Prefers the brain layer (the real MRI); falls back to craft so the
    motif is always populated.
    """
    brain = {r["term"]: r for r in cmp.get("brain_deltas", [])
             if r["term"].endswith(".mean")}
    if brain:
        rows, layer = [], "brain"
        for net, label in _NET_ORDER:
            r = brain.get(f"net.{net}.mean")
            if r is None:
                continue
            rows.append((label, r["spread_norm"], r["demo"], r["taste"]))
        if rows:
            return rows, layer
    craft = {r["term"]: r for r in cmp.get("craft_deltas", [])}
    rows = []
    for feat, label in _CRAFT_ORDER:
        r = craft.get(feat)
        if r is None:
            continue
        rows.append((label, r["spread_norm"], r["demo"], r["taste"]))
    return rows, "craft"


def _bar_text(norm: float):
    """A single heat-bar with the demo ghost overlaid, as a rich Text."""
    from rich.text import Text

    close = idy.closeness(norm)
    fill = max(1, round(close * BAR_W))
    col = idy.ramp(close)
    # the ghost sits where the DEMO lands: closeness already encodes the
    # gap, so the diamond marks the head of the filled region.
    ghost_at = min(BAR_W - 1, max(0, fill - 1))

    t = Text()
    for i in range(BAR_W):
        if i == ghost_at:
            t.append(idy.GHOST_MARK, style=f"bold {idy.GHOST}")
        elif i < fill:
            t.append(idy.BLOCK, style=col)
        else:
            t.append(idy.TRACK, style=idy.FAINT)
    return t, col, close


def panel(cmp: dict, *, title_extra: str = "") -> object:
    """The hero fingerprint as a rich Panel."""
    from rich.box import HEAVY
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    rows, layer = _rows_from_compare(cmp)
    if not rows:
        return Panel(
            Text("no comparable signals — feed it audio", style=idy.DIM),
            box=HEAVY, border_style=idy.RULE, padding=(1, 2),
            title="[bold]taste fingerprint[/]",
        )

    grid = Table.grid(padding=(0, 1), expand=False)
    grid.add_column(justify="right", no_wrap=True, style=idy.DIM, min_width=15)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="left", no_wrap=True)

    for label, norm, _demo, _taste in rows:
        bar, col, close = _bar_text(norm)
        read = ("locked" if close > 0.82 else
                "close" if close > 0.6 else
                "drifting" if close > 0.38 else "off")
        grid.add_row(
            Text(label, style=idy.INK),
            bar,
            Text(f" {read}", style=col),
        )

    layer_tag = ("12-network brain MRI" if layer == "brain"
                 else "craft signature · model-free")
    legend = Text.from_markup(
        f"[{idy.GHOST}]{idy.GHOST_MARK}[/] your demo   "
        f"[{idy.COOL}]{idy.BLOCK}{idy.BLOCK}[/] on the taste   "
        f"[{idy.HOT}]{idy.BLOCK}{idy.BLOCK}[/] off it"
    )
    from rich.console import Group

    body = Group(grid, Text(), legend)
    ttl = f"[bold {idy.GOLD}]taste fingerprint[/] [{idy.DIM}]· {layer_tag}"
    if title_extra:
        ttl += f" · {title_extra}"
    ttl += "[/]"
    return Panel(body, box=HEAVY, border_style=idy.RULE, padding=(1, 2),
                 title=ttl, title_align="left")


# --------------------------------------------------------------------------
# Pure-Unicode variant for the shareable card / markdown (no rich, no colour
# — just the shape, so it survives a paste into Discord/Notion/a gist).
# --------------------------------------------------------------------------

_SHADE = " ░▒▓█"


def ascii_rows(cmp: dict, width: int = 22) -> list[str]:
    rows, _layer = _rows_from_compare(cmp)
    out = []
    for label, norm, _d, _t in rows:
        close = idy.closeness(norm)
        fill = max(1, round(close * width))
        ch = _SHADE[min(len(_SHADE) - 1, 1 + int(close * (len(_SHADE) - 2)))]
        bar = ch * fill + "·" * (width - fill)
        # ghost marker
        gi = min(width - 1, max(0, fill - 1))
        bar = bar[:gi] + "◆" + bar[gi + 1:]
        out.append(f"{label:>15} │{bar}│")
    return out
