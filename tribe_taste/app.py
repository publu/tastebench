"""tribe_taste.app — the simple drop-and-grade page (Textual).

One screen. Two modes, toggled with [space]:

  • REFERENCE — drop work you admire; it's added to your taste set.
  • GRADE     — drop your draft; it's graded against that set and the
                grade is printed.

Drag files from Finder straight onto the drop bar (single or multi; the
terminal pastes the path). Grading runs on a worker thread so the UI
never freezes. M-series-tuned: the verified-faithful Apple-Silicon env
(MPS, bf16 text, whisperx-float32) is set on import.

Launched by bare `tribe-taste` on a real terminal; non-TTY falls back.
"""

from __future__ import annotations

import os
import shlex
from glob import glob
from pathlib import Path

# --- M-series-faithful perf env (BEFORE any tribe engine import) ----------
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

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle, Vertical
from textual.widgets import Footer, Input, RichLog, Static

_MEDIA = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
}


def _parse_drop(raw: str) -> list[str]:
    """A dropped/typed string -> existing media paths (single, multi,
    quoted, backslash-escaped, glob)."""
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
                ap = os.path.abspath(c)
                if ap not in out:
                    out.append(ap)
    return out


class TasteApp(App):
    TITLE = "tribe-taste"

    CSS = """
    Screen { background: $background; align: center middle; }
    #card { width: 92%; max-width: 110; height: auto; }
    #brand {
        height: 1; color: $accent; text-style: bold; margin-bottom: 1;
    }
    #mode {
        height: 3; border: round $accent; background: $surface;
        padding: 1 2; text-style: bold; content-align: center middle;
    }
    #drop {
        height: 3; border: round $surface; background: $surface;
        padding: 0 2; margin: 1 0;
    }
    #drop:focus { border: round $accent; }
    #refs {
        height: 3; color: $text-muted; padding: 0 1; margin-bottom: 1;
    }
    #out {
        height: 22; border: round $surface; background: $surface;
        padding: 1 2; scrollbar-size: 1 1;
    }
    #status {
        dock: bottom; height: 1; padding: 0 2;
        background: $panel; color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("space", "toggle", "Switch add ⇄ grade", show=True),
        Binding("b", "brain", "Brain on/off", show=True),
        Binding("x", "clear", "Clear references", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.mode = "ref"          # "ref" | "grade"
        self.use_brain = False
        self.busy = False

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                with Vertical(id="card"):
                    yield Static(
                        "◢ tribe-taste · a focus group for the work "
                        "you make alone", id="brand")
                    yield Static(id="mode")
                    yield Input(placeholder="drag file(s) here, or type "
                                "a path / glob, then ⏎", id="drop")
                    yield Static(id="refs")
                    yield RichLog(id="out", wrap=True, markup=True,
                                  highlight=False)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#out", RichLog).write(
            "[b]How it works[/b]\n"
            "1. Leave it on [cyan]REFERENCE[/] and drop a few things you "
            "wish your work felt like — they teach it your taste.\n"
            "2. Press [b]space[/] to switch to [magenta]GRADE[/], then drop "
            "your own draft. It prints the grade: how close it is and the "
            "one biggest thing to change.\n"
            "[dim]Brain layer off by default (instant). Press b for the "
            "TRIBE neural read (needs the model).[/dim]")

    # ---- state ------------------------------------------------------------
    def _refresh(self) -> None:
        n = len(self.refs)
        if self.mode == "ref":
            self.query_one("#mode", Static).update(
                "[reverse] REFERENCE [/]  drop work you admire — it builds "
                "your taste     [dim]space → switch to grade[/]")
        else:
            self.query_one("#mode", Static).update(
                f"[reverse] GRADE [/]  drop your draft — graded against your "
                f"[b]{n}[/] references     [dim]space → switch to reference[/]")
        names = ", ".join(Path(r).name for r in self.refs[-5:]) or "none yet"
        extra = f"  (+{n-5} more)" if n > 5 else ""
        self.query_one("#refs", Static).update(
            f"[b]references ({n})[/]  {names}{extra}")
        brain = "[magenta]brain ON[/]" if self.use_brain else \
            "[dim]brain off · craft[/]"
        state = "[yellow]⟳ grading…[/]" if self.busy else "ready"
        self.query_one("#status", Static).update(
            f"▸ {n} refs · {brain} · {state}   "
            f"[dim]space add⇄grade · b brain · x clear · q quit[/]")

    # ---- the one interaction: a drop --------------------------------------
    def on_input_submitted(self, e: Input.Submitted) -> None:
        files = _parse_drop(e.value)
        e.input.value = ""
        if not files:
            self.notify("no media files in that drop/path",
                        severity="warning")
            return
        if self.mode == "ref":
            for f in files:
                if f not in self.refs:
                    self.refs.append(f)
            self.notify(f"+{len(files)} reference(s)")
            self._refresh()
            return
        # grade mode
        if not self.refs:
            self.notify("add references first (space → REFERENCE, drop a "
                        "few things you like)", severity="error")
            return
        if self.busy:
            self.notify("still grading the last one…", severity="warning")
            return
        self.busy = True
        self._refresh()
        cand = files[0]
        self.query_one("#out", RichLog).write(
            f"\n[dim]── grading {Path(cand).name}"
            f"{' (brain — can take minutes)' if self.use_brain else ''} ──[/]")
        self._grade(cand)

    @work(thread=True, exclusive=True, group="grade")
    def _grade(self, cand: str) -> None:
        try:
            from rich.text import Text

            from .compare import compare
            from .report import to_verdict

            pay = compare(cand, self.refs, use_brain=self.use_brain)
            pay["_kind"] = "compare"
            lines = [Text.from_markup(to_verdict(pay))]
            top = [r for r in pay.get("craft_deltas", [])
                   if abs(r.get("spread_norm", 0)) >= 0.5][:4]
            if top:
                lines.append(Text.from_markup(
                    "\n[b]biggest gaps[/]\n" + "\n".join(
                        f"  • {r['voice']}" for r in top)))
            self.call_from_thread(self._done, lines)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, repr(exc))

    def _done(self, lines: list) -> None:
        out = self.query_one("#out", RichLog)
        for ln in lines:
            out.write(ln)
        self.busy = False
        self._refresh()

    def _fail(self, msg: str) -> None:
        self.query_one("#out", RichLog).write(f"[red]grading failed:[/] {msg}")
        self.busy = False
        self._refresh()
        self.notify("grading failed — see the panel", severity="error")

    # ---- controls ---------------------------------------------------------
    def action_toggle(self) -> None:
        self.mode = "grade" if self.mode == "ref" else "ref"
        self._refresh()

    def action_brain(self) -> None:
        self.use_brain = not self.use_brain
        self._refresh()
        if self.use_brain:
            self.notify("brain ON — needs the ~20GB model; first grade "
                        "loads it (minutes on M-series). Falls back to "
                        "craft if absent.", timeout=6)

    def action_clear(self) -> None:
        self.refs.clear()
        self._refresh()
        self.notify("references cleared")


def run_app() -> int:
    TasteApp().run()
    return 0
