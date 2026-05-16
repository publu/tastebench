"""tribe_taste.tui — the product TUI.

A polished one-screen read-out of a taste comparison: a distance-to-your-
taste dial, a brain-network heatmap (references vs demo), a craft-delta
table, and the ranked edit list. Built on `rich`. (Style/layout patterns
are reused from an internal rich dashboard; this is a product view, not a
pipeline monitor.)

    tribe-taste tui REF [REF ...] [--demo DEMO] [--no-brain]
"""

from __future__ import annotations

from typing import Optional


def _need_rich():
    try:
        import rich  # noqa: F401
    except Exception:
        return False
    return True


# perceptual far(red) -> close(green) ramp for the dial / heatmap
RAMP = [
    "#ff4d4d", "#ff7a45", "#ffa600", "#ffd000", "#cfe000",
    "#9be600", "#5cd65c", "#2ecc71", "#16c6a4",
]


def _ramp(frac: float) -> str:
    frac = max(0.0, min(1.0, frac))
    return RAMP[int(frac * (len(RAMP) - 1))]


def _dial(distance: Optional[float]) -> "object":
    from rich.align import Align
    from rich.box import HEAVY
    from rich.panel import Panel
    from rich.text import Text

    if distance is None:
        body = Text("no comparable features", style="grey50")
        return Panel(Align.center(body), box=HEAVY, border_style="grey35",
                     title="distance to your taste", padding=(1, 2))
    # map 0 (perfect) .. 4 (far) onto a 0..1 closeness
    close = max(0.0, min(1.0, 1.0 - distance / 4.0))
    width = 40
    fill = int(close * width)
    col = _ramp(close)
    bar = Text()
    bar.append("█" * fill, style=col)
    bar.append("░" * (width - fill), style="grey23")
    label = (
        "very close" if distance < 0.75 else
        "in this taste" if distance < 1.5 else
        "noticeably off" if distance < 3.0 else "far"
    )
    body = Text()
    body.append(f"  {distance:.2f}  ", style=f"bold {col}")
    body.append(f"{label}\n\n", style="grey74")
    grp = _group(bar, Text(), body)
    return Panel(_center(grp), box=HEAVY, border_style=col,
                 title="[bold]distance to your taste[/]", padding=(1, 3))


def _group(*items):
    from rich.console import Group

    return Group(*items)


def _center(renderable):
    from rich.align import Align

    return Align.center(renderable)


def _heatmap(compare_result: dict) -> "object":
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    rows = [r for r in compare_result.get("brain_deltas", [])
            if r["term"].endswith(".mean")]
    t = Table(box=None, expand=True, padding=(0, 1))
    t.add_column("network", style="bold", no_wrap=True)
    t.add_column("demo", justify="right")
    t.add_column("taste", justify="right")
    t.add_column("", no_wrap=True)
    t.add_column("read", style="grey62")
    if not rows:
        return Panel(
            Text("brain layer not available (craft-only run).\n"
                 "Install the TRIBE model for the neural heatmap "
                 "(scripts/download_models.py).", style="grey50"),
            title="[bold]brain-network signature[/] [grey50](refs vs demo)[/]",
            box=ROUNDED, border_style="grey35", padding=(1, 2),
        )
    maxz = max((abs(r["spread_norm"]) for r in rows), default=1.0) or 1.0
    for r in rows:
        name = r["term"].split(".")[1]
        frac = 1.0 - min(1.0, abs(r["spread_norm"]) / maxz)
        col = _ramp(frac)
        mag = min(18, int(abs(r["spread_norm"]) / maxz * 18))
        bar = Text("█" * max(1, mag), style=col)
        t.add_row(
            name,
            f"{r['demo']:.2f}",
            f"{r['taste']:.2f}",
            bar,
            r["plain"],
        )
    return Panel(t, title="[bold]brain-network signature[/] "
                          "[grey50](demo vs your taste — hypothesis view)[/]",
                 box=ROUNDED, border_style="magenta", padding=(1, 1))


def _craft_table(compare_result: dict) -> "object":
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table

    t = Table(box=None, expand=True, padding=(0, 1))
    t.add_column("feature", style="bold cyan", no_wrap=True)
    t.add_column("demo", justify="right")
    t.add_column("taste", justify="right")
    t.add_column("Δ", justify="right")
    t.add_column("tag", no_wrap=True)
    t.add_column("note", style="grey62")
    for r in compare_result.get("craft_deltas", [])[:12]:
        d = r["spread_norm"]
        col = "green" if abs(d) < 0.75 else "yellow" if abs(d) < 1.5 else "red"
        tag_c = {"song-bones": "cyan", "production": "grey62",
                 "neural": "magenta"}.get(r["kind"], "grey62")
        t.add_row(
            r["term"],
            f"{r['demo']:.2f}",
            f"{r['taste']:.2f}",
            f"[{col}]{r['delta']:+.2f}[/]",
            f"[{tag_c}]{r['kind']}[/]",
            r["plain"],
        )
    return Panel(t, title="[bold cyan]craft divergence[/] "
                          "[grey50](biggest first)[/]",
                 box=ROUNDED, border_style="cyan", padding=(1, 1))


def _edits_panel(opt_result: dict) -> "object":
    from rich.box import ROUNDED
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    edits = opt_result.get("edits", [])
    if not edits:
        return Panel(Text(opt_result.get("note", "no edits"), style="grey50"),
                     title="[bold green]ranked edits[/]",
                     box=ROUNDED, border_style="green", padding=(1, 2))
    t = Table(box=None, expand=True, padding=(0, 1))
    t.add_column("#", justify="right", style="grey50")
    t.add_column("edit", style="bold green", no_wrap=True)
    t.add_column("from→to", justify="right")
    t.add_column("gain", justify="right")
    t.add_column("conf", no_wrap=True)
    t.add_column("note", style="grey62")
    for i, e in enumerate(edits, 1):
        cc = {"high": "green", "medium": "yellow", "low": "grey50"}.get(
            e["confidence"], "grey50")
        t.add_row(
            str(i),
            e["edit"],
            f"{e['from']:.2f}→{e['to']:.2f}",
            f"[green]+{e['predicted_gain']:.3f}[/]",
            f"[{cc}]{e['confidence']}[/]",
            e["plain"],
        )
    return Panel(t, title="[bold green]ranked edits[/] "
                          "[grey50](toward your taste — A/B, not guarantees)[/]",
                 box=ROUNDED, border_style="green", padding=(1, 1))


def _header(compare_result: dict, n_refs: int) -> "object":
    from rich.box import HEAVY
    from rich.panel import Panel
    from rich.text import Text

    demo = compare_result["demo"]["name"]
    nr = compare_result.get("nearest_reference")
    near = f" · nearest ref [bold]{nr['name']}[/]" if nr else ""
    txt = Text.from_markup(
        f"[bold white]tribe-taste[/]  [grey42]·[/]  demo [bold]{demo}[/]  "
        f"[grey42]vs[/]  [bold]{n_refs}[/] references{near}"
    )
    return Panel(txt, box=HEAVY, border_style="grey35", padding=(0, 2))


def run(refs, demo: Optional[str] = None, use_brain: bool = True) -> int:
    if not _need_rich():
        print("The TUI needs `rich`. Install with: pip install rich")
        return 1

    from rich.console import Console
    from rich.layout import Layout

    from .compare import compare
    from .optimize import optimize
    from .profile import build_profile

    console = Console()
    if not demo:
        # no demo: show the profile as a glossary-anchored summary
        from .profile import profile_summary
        from .report import to_markdown

        prof = build_profile(refs, use_brain=use_brain)
        payload = profile_summary(prof)
        payload["_kind"] = "profile"
        from rich.markdown import Markdown

        console.print(Markdown(to_markdown(payload)))
        console.print(
            "[grey50]Pass --demo to see the dial / heatmap / edit view.[/]"
        )
        return 0

    with console.status("[grey50]building taste profile + analyzing demo..."):
        prof = build_profile(refs, use_brain=use_brain)
        cmp = compare(demo, prof, use_brain=use_brain)
        opt = optimize(demo, prof, use_brain=False)

    lay = Layout()
    lay.split_column(
        Layout(_header(cmp, prof["n_refs"]), name="h", size=3),
        Layout(name="top", size=9),
        Layout(name="mid", ratio=1),
        Layout(_edits_panel(opt), name="edits", size=14),
    )
    lay["top"].split_row(
        Layout(_dial(cmp.get("overall_distance")), name="dial"),
        Layout(_heatmap(cmp), name="heat", ratio=2),
    )
    lay["mid"].update(_craft_table(cmp))
    console.print(lay)
    return 0
