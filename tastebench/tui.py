"""tastebench.tui — the product TUI.

A polished one-screen read-out of a taste comparison: a distance-to-your-
taste dial, a brain-network heatmap (references vs demo), a craft-delta
table, and the ranked edit list. Built on `rich`. (Style/layout patterns
are reused from an internal rich dashboard; this is a product view, not a
pipeline monitor.)

    tastebench tui REF [REF ...] [--demo DEMO] [--no-brain]
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
        f"[bold white]tastebench[/]  [grey42]·[/]  demo [bold]{demo}[/]  "
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


# --------------------------------------------------------------------------
# Interactive shell — `tastebench` with no args. Pick files + action in
# the app; no need to know the CLI. rich-only (rich.prompt), no raw-tty.
# --------------------------------------------------------------------------

_MEDIA_EXTS = None


def _media_exts():
    global _MEDIA_EXTS
    if _MEDIA_EXTS is None:
        try:
            from .engine import SUPPORTED_EXTS

            _MEDIA_EXTS = set(SUPPORTED_EXTS)
        except Exception:
            _MEDIA_EXTS = {
                ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus",
                ".png", ".jpg", ".jpeg", ".webp", ".bmp",
                ".mp4", ".mov", ".mkv", ".avi",
            }
    return _MEDIA_EXTS


def _pick(console, multi: bool, what: str):
    """Numbered directory browser. Returns list[str] of paths, or None."""
    import glob as _glob
    import os

    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table

    exts = _media_exts()
    cur = os.getcwd()
    chosen: list[str] = []
    while True:
        try:
            entries = sorted(os.listdir(cur))
        except OSError:
            cur = os.path.expanduser("~")
            continue
        dirs = [e for e in entries
                if os.path.isdir(os.path.join(cur, e)) and not e.startswith(".")]
        files = [e for e in entries
                 if os.path.splitext(e)[1].lower() in exts]
        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style="grey50")
        t.add_column()
        idx = []
        for d in dirs:
            idx.append(("d", d))
            t.add_row(str(len(idx)), f"[cyan]{d}/[/]")
        for f in files:
            idx.append(("f", f))
            mark = ("[green]✓[/] "
                    if os.path.join(cur, f) in chosen else "  ")
            t.add_row(str(len(idx)), f"{mark}{f}")
        seln = (f"  [grey50]·[/] [green]{len(chosen)}[/] selected"
                if multi and chosen else "")
        console.print(Panel(
            t if idx else "[grey50](no folders or media files here)[/]",
            title=f"[bold]pick {what}[/]  [grey50]{cur}[/]{seln}",
            subtitle="[grey50]#=open/select · u=up · ~=home · /path · "
            "g <glob>" + (" · done" if multi else "") + " · q=cancel[/]",
            border_style="cyan", padding=(1, 1)))
        ans = Prompt.ask("[bold cyan]>[/]", console=console,
                         default="" if multi else " ").strip()
        low = ans.lower()
        if low == "q":
            return None
        if low in ("u", ".."):
            cur = os.path.dirname(cur) or "/"
            continue
        if ans == "~":
            cur = os.path.expanduser("~")
            continue
        if multi and low in ("done", ""):
            return chosen or None
        if low.startswith("g "):
            for m in _glob.glob(os.path.expanduser(ans[2:].strip())):
                if (os.path.isfile(m)
                        and os.path.splitext(m)[1].lower() in exts):
                    m = os.path.abspath(m)
                    if m not in chosen:
                        chosen.append(m)
            continue
        cand = os.path.abspath(os.path.expanduser(ans)) if ans.strip() else ""
        if cand and os.path.isdir(cand):
            cur = cand
            continue
        if cand and os.path.isfile(cand):
            if not multi:
                return [cand]
            chosen.remove(cand) if cand in chosen else chosen.append(cand)
            continue
        if ans.isdigit() and 1 <= int(ans) <= len(idx):
            kind, name = idx[int(ans) - 1]
            full = os.path.abspath(os.path.join(cur, name))
            if kind == "d":
                cur = full
            elif not multi:
                return [full]
            else:
                chosen.remove(full) if full in chosen else chosen.append(full)


def interactive() -> int:
    """Guided, no-args TUI: pick references + demo + action, see results,
    iterate. Launched by bare `tastebench` (or `tastebench tui`)."""
    import os
    import sys

    if not _need_rich():
        print("Interactive TUI needs rich:  pip install rich")
        return 1

    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    if not sys.stdin.isatty():
        console.print(Panel(
            "[yellow]Not a TTY — the interactive TUI needs a real "
            "terminal.[/]\nUse the CLI instead, e.g.:\n"
            "  [cyan]tastebench vibe demo.wav --like ref1.wav ref2.wav[/]\n"
            "  [cyan]tastebench compare ref1.wav ref2.wav --to demo.wav[/]",
            title="tastebench", border_style="grey35"))
        return 0

    console.print(Panel.fit(
        "[bold]tastebench[/]  [grey50]· an MRI for your taste[/]\n"
        "[grey50]pick a few references you admire, then your demo — "
        "see exactly how yours diverges.[/]", border_style="cyan"))

    refs: list[str] = []
    demo = None
    use_brain = False
    while True:
        rs = (f"[green]{len(refs)}[/] refs" if refs else "[grey50]no refs[/]")
        ds = (f"demo [green]{os.path.basename(demo)}[/]" if demo
              else "[grey50]no demo[/]")
        bs = ("[magenta]brain ON[/]" if use_brain
              else "[grey50]brain off · fast craft-only[/]")
        console.print(Panel(
            "[bold]1[/] pick references    [bold]2[/] pick demo\n"
            "[bold]3[/] vibe [grey50](quick verdict)[/]    "
            "[bold]4[/] compare [grey50](full visual)[/]\n"
            "[bold]5[/] optimize [grey50](edit list)[/]    "
            "[bold]6[/] profile [grey50](taste only)[/]\n"
            "[bold]b[/] toggle brain    [bold]q[/] quit",
            title=f"[bold]{rs}  ·  {ds}  ·  {bs}[/]", border_style="grey35"))
        c = Prompt.ask("[bold cyan]choose[/]", console=console,
                       default="").strip().lower()

        if c == "q":
            console.print("[grey50]bye.[/]")
            return 0
        if c == "1":
            p = _pick(console, True, "references (work you admire)")
            if p:
                refs = p
            continue
        if c == "2":
            p = _pick(console, False, "your demo")
            if p:
                demo = p[0]
            continue
        if c == "b":
            use_brain = not use_brain
            if use_brain:
                console.print("[yellow]brain ON — needs the ~20 GB model; "
                              "first run loads it (minutes). Falls back to "
                              "craft cleanly if it's absent.[/]")
            continue
        if c not in ("3", "4", "5", "6"):
            console.print("[grey50]enter 1-6, b, or q.[/]")
            continue
        if not refs:
            console.print("[red]pick references first ([bold]1[/]).[/]")
            continue
        if c in ("3", "4", "5") and not demo:
            console.print("[red]pick a demo first ([bold]2[/]).[/]")
            continue
        try:
            if c == "4":
                run(refs, demo, use_brain=use_brain)
            elif c == "6":
                run(refs, None, use_brain=use_brain)
            else:
                with console.status("[grey50]analyzing…[/]"):
                    if c == "3":
                        from .compare import compare
                        from .report import to_verdict
                        pay = compare(demo, refs, use_brain=use_brain)
                        pay["_kind"] = "compare"
                        out = to_verdict(pay)
                    else:  # 5 optimize
                        from rich.markdown import Markdown
                        from .optimize import optimize
                        from .report import to_markdown
                        pay = optimize(demo, refs, use_brain=use_brain)
                        pay["_kind"] = "optimize"
                        out = Markdown(to_markdown(pay))
                console.print(out)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]run failed:[/] {e}")
        Prompt.ask("[grey50]Enter for the menu[/]", console=console,
                   default="")
