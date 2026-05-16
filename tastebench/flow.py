"""tastebench.flow — the default experience: a prompt you drop files into.

This is a *line prompt*, not a full-screen app — and that's the whole
point. Dragging a file from Finder onto a terminal pastes its path **at a
prompt**. A full-screen (alt-screen) TUI never receives that, which is why
the Textual version's drag-drop didn't work. Here it works because it's
the same mechanism every other terminal tool uses.

Flow: welcome → drop the work you admire (it's auto-processed through
TRIBE as it lands) → type `grade`, drop your draft, it's graded against
the board. Brain is ON automatically when the weights are present; if
they're not, that's surfaced up front with a one-word `download`. `auto`
toggles the automatic processing off/on.

M-series-faithful env is set on import (mps / bf16 text / whisperx f32).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from glob import glob
from pathlib import Path

for _k, _v in {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "TRIBE_DEVICE": "auto",
    "TRIBE_FAST_TEXT": "1",
    "TRIBE_TEXT_DTYPE": "bf16",
    "TRIBE_FAST_TEXT_BATCH": "1",
    "TRIBE_ASR_ENGINE": "whisperx",
    "TRIBE_ASR_COMPUTE": "float32",
}.items():
    os.environ.setdefault(_k, _v)

_MEDIA = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
}

BANNER = """\
[bold cyan]╭───────────────────────────────────────╮
│    █████   ███    ████  █████  █████  │
│      █    █   █  █        █    █      │
│      █    █████   ███     █    ███    │
│      █    █   █      █    █    █      │
│      █    █   █  ████     █    █████  │
│                                       │
│    ████   █████  █   █   ████  █   █  │
│    █   █  █      ██  █  █      █   █  │
│    ████   ███    █ █ █  █      █████  │
│    █   █  █      █  ██  █      █   █  │
│    ████   █████  █   █   ████  █   █  │
╰───────────────────────────────────────╯[/]
[dim] a private focus group for your drafts[/]"""


def _key(p: str) -> str:
    return str(Path(p))


def _parse_drop(raw: str) -> list[str]:
    """A dropped/typed line -> de-duped absolute media paths (URLs rejected)."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        toks = shlex.split(raw)
    except ValueError:
        toks = raw.split()
    out: list[str] = []
    for t in toks:
        t = os.path.expanduser(t.strip().strip("'\""))
        for c in (glob(t) if any(g in t for g in "*?[") else [t]):
            if os.path.isfile(c) and Path(c).suffix.lower() in _MEDIA:
                k = _key(os.path.abspath(c))
                if k not in out:
                    out.append(k)
    return out


def prompt_flow(_input=None) -> int:
    """The drop prompt. `_input` is an injectable line reader for tests."""
    from rich.console import Console

    console = Console()
    if _input is None and not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(
            "[yellow]tastebench's drop prompt needs a real terminal.[/] "
            "Use the CLI instead, e.g.:\n"
            "  [cyan]tastebench compare ref1.wav ref2.wav --to demo.wav[/]")
        return 0
    ask = _input or console.input

    from .engine import models_available

    have = models_available()
    auto = have                       # automatic TRIBE on iff weights present
    refs: list[str] = []              # absolute media paths, in drop order
    cache: dict[str, dict] = {}       # path -> precomputed signature

    console.print(BANNER)
    console.print()
    if have:
        console.print("[green]✓ brain weights ready.[/] Every file you drop "
                      "is auto-processed through TRIBE as it lands.")
    else:
        console.print(
            "[yellow]● brain weights not installed (~20 GB).[/] Auto-TRIBE "
            "stays off until you have them —\n  type [b cyan]download[/] to "
            "fetch them now, or just drop files for the instant craft read.")
    console.print(
        "\n[b]Welcome.[/] Drop the work you admire below — drag the files "
        "straight in, then Enter.\nWhen the board feels right, type "
        "[b cyan]grade[/] and drop your own draft.\n"
        "[dim]commands: grade · auto · download · clear · q[/]\n")

    def _process(paths: list[str]) -> None:
        from .signature import signature_for
        ub = auto and have
        new = [p for p in paths if p not in refs]
        for p in new:
            refs.append(p)
            with console.status(
                f"[yellow]⚙ TRIBE-processing[/] {Path(p).name} "
                f"{'(brain — can take minutes)' if ub else '(craft)'} …"):
                try:
                    sig = signature_for(p, use_brain=ub)
                    cache[p] = sig
                    got_brain = sig["brain"].get("available")
                    ok = got_brain or sig["craft"].get("available")
                    if ok:
                        tag = "  [magenta](brain)[/]" if got_brain else ""
                        console.print(f"  [green]✓[/] {Path(p).name}{tag}")
                    else:
                        console.print(
                            f"  [red]✗[/] {Path(p).name} (unreadable)")
                except Exception as e:  # noqa: BLE001
                    console.print(f"  [red]✗[/] {Path(p).name}: {e}")
        console.print(f"[b]board:[/] {len(refs)} reference(s)\n")

    def _grade(draft: str) -> None:
        from .compare import compare
        from .profile import build_profile
        from .report import to_verdict
        ub = auto and have
        try:
            with console.status(
                f"[yellow]⚙ grading[/] {Path(draft).name} against "
                f"{len(refs)} references {'(brain)' if ub else '(craft)'} …"):
                prof = build_profile(refs, use_brain=ub, precomputed=cache)
                pay = compare(draft, prof, use_brain=ub)
                pay["_kind"] = "compare"
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]grading failed:[/] {e}\n")
            return
        console.print(to_verdict(pay))
        top = [r for r in pay.get("craft_deltas", [])
               if abs(r.get("spread_norm", 0)) >= 0.5][:4]
        if top:
            console.print("\n[b]biggest gaps vs the board[/]")
            for r in top:
                console.print(f"  • {r['voice']}")
        console.print()

    while True:
        try:
            line = ask("[bold cyan]drop ▸[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        low = line.lower()

        if low in ("q", "quit", "exit"):
            return 0
        if low in ("clear", "x"):
            refs.clear()
            cache.clear()
            console.print("[dim]board cleared[/]\n")
            continue
        if low == "auto":
            auto = not auto
            if auto and not have:
                console.print("[yellow]auto-TRIBE on, but no weights — "
                              "files get the craft read until you "
                              "[b cyan]download[/].[/]\n")
            else:
                console.print(
                    f"[dim]auto-TRIBE {'on' if auto else 'off'} "
                    f"{'(brain)' if auto and have else '(craft only)'}[/]\n")
            continue
        if low == "download":
            have = _download(console)
            auto = have
            continue
        if low in ("grade", "g"):
            if not refs:
                console.print("[red]drop some references first.[/]\n")
                continue
            try:
                d = ask("[bold cyan]drop your draft ▸[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return 0
            df = _parse_drop(d)
            if not df:
                console.print("[red]no local media in that.[/]\n")
                continue
            _grade(df[0])
            continue

        files = _parse_drop(line)
        if not files:
            console.print("[yellow]no local media there — drop a file, "
                          "not a URL.[/]\n")
            continue
        _process(files)


def _download(console) -> bool:
    """Run the bundled ~20 GB weights downloader, then re-check availability."""
    from .engine import models_available

    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "download_models.py")
    if not script.exists():
        console.print(f"[red]download script not found: {script}[/]\n")
        return models_available()
    console.print("[yellow]downloading the ~20 GB model cache — this is "
                  "long, and needs a Hugging Face login for the gated "
                  "Llama-3.2 weights …[/]")
    try:
        subprocess.run([sys.executable, str(script)], check=False)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]download failed: {e}[/]")
    ok = models_available()
    console.print("[green]✓ weights ready — auto-TRIBE on.[/]\n" if ok
                  else "[yellow]still unavailable; craft-only for now.[/]\n")
    return ok
