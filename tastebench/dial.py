"""tastebench.dial — the TASTE MATCH gauge.

One satisfying headline number: TASTE MATCH %, how close the demo sits to
the taste you defined (100 = it already sounds like the records you love).
Rendered as a chunky gradient arc-bar that you watch MOVE when an optimize
edit is applied and re-scored — close the loop visibly, make it feel like
an arcade you want to keep playing.
"""

from __future__ import annotations

from typing import Optional

from . import identity as idy

_W = 30  # gauge cells


def _gauge_text(match: int, *, ghost: Optional[int] = None):
    """The gradient gauge as a rich Text. `ghost` draws a 'before' tick."""
    from rich.text import Text

    fill = round(match / 100 * _W)
    t = Text()
    for i in range(_W):
        frac = (i + 1) / _W
        if ghost is not None and i == max(0, round(ghost / 100 * _W) - 1) and i >= fill:
            t.append("╎", style=idy.DIM)
        elif i < fill:
            t.append("█", style=idy.ramp(frac))
        else:
            t.append("─", style=idy.FAINT)
    return t


def panel(distance: Optional[float], *, was: Optional[float] = None) -> object:
    """The TASTE MATCH dial as a rich Panel.

    `was`: a previous distance — draws the old needle + the live delta so a
    re-score visibly moves the dial.
    """
    from rich.box import HEAVY
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    match = idy.resonance(distance)
    if match is None:
        return Panel(
            Text("no read yet", style=idy.DIM, justify="center"),
            box=HEAVY, border_style=idy.RULE, padding=(1, 3),
            title=f"[bold {idy.GOLD}]TASTE MATCH[/]", title_align="left",
        )

    col = idy.ramp(match / 100)
    prev = idy.resonance(was) if was is not None else None

    big = Text(justify="center")
    big.append(f"{match}", style=f"bold {col}")
    big.append("%", style=idy.DIM)
    if prev is not None and prev != match:
        d = match - prev
        arrow = "▲" if d > 0 else "▼"
        big.append(f"   {arrow}{d:+d}", style=idy.COOL if d > 0 else idy.HOT)

    word = Text(idy.verdict_word(match), style=col, justify="center")
    gauge = _gauge_text(match, ghost=prev)

    body = Group(Text(), big, Text(), gauge, Text(), word)
    return Panel(body, box=HEAVY, border_style=col, padding=(1, 3),
                 title=f"[bold {idy.GOLD}]TASTE MATCH[/]", title_align="left",
                 subtitle=f"[{idy.DIM}]how close you are to the taste you picked[/]")


def inline(distance: Optional[float], *, was: Optional[float] = None) -> str:
    """A one-line gauge as a rich-markup string (for compact contexts)."""
    match = idy.resonance(distance)
    if match is None:
        return f"[{idy.DIM}]TASTE MATCH —[/]"
    col = idy.ramp(match / 100)
    fill = round(match / 100 * 18)
    bar = f"[{col}]" + "█" * fill + f"[/][{idy.FAINT}]" + "─" * (18 - fill) + "[/]"
    out = f"[bold {idy.GOLD}]TASTE MATCH[/] {bar} [bold {col}]{match}%[/]"
    if was is not None:
        prev = idy.resonance(was)
        if prev is not None and prev != match:
            d = match - prev
            out += f" [{idy.COOL if d > 0 else idy.HOT}]{d:+d}[/]"
    return out
