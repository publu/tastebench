"""tastebench.app — the moodboard (Textual).

Your references are a board of inspiration tiles. Drop work you admire and
it lands on the board; press [space] to switch to GRADE and drop your own
draft — it's graded against the board and the grade is printed.

Drag files from Finder straight onto the drop bar (single or multi).
Grading runs on a worker thread so the UI never freezes. M-series-tuned:
the verified-faithful Apple-Silicon env is set on import.

Launched by bare `tastebench` on a real terminal; non-TTY falls back.
"""

from __future__ import annotations

import os
import shlex
from glob import glob
from pathlib import Path

# --- M-series-faithful perf env (BEFORE any tastebench engine import) -----
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
from textual.containers import Center, Middle, Vertical, VerticalScroll
from textual.widgets import Footer, Input, RichLog, Static

BANNER = """\
╭───────────────────────────────────────╮
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
╰───────────────────────────────────────╯
[dim] a private focus group for your drafts
 ∿╱╲∿_╱╲∿╱╲__╱╲∿╱╲∿  drop references, then grade your draft[/dim]"""

_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_MEDIA = _AUDIO | _IMAGE | _VIDEO

_COLS = 4
_TW = 15  # tile inner width


def _parse_drop(raw: str) -> list[str]:
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


def _kind(p: str) -> tuple[str, str]:
    e = Path(p).suffix.lower()
    if e in _AUDIO:
        return "♪", "cyan"
    if e in _IMAGE:
        return "▦", "magenta"
    return "▶", "green"


def _tile(label: str, glyph: str, color: str, dim: bool = False) -> list[str]:
    nm = label if len(label) <= _TW - 2 else label[: _TW - 3] + "…"
    c = "grey37" if dim else color
    return [
        f"[{c}]╭{'─' * _TW}╮[/]",
        f"[{c}]│[/] [b {c}]{glyph}[/]{' ' * (_TW - 2)}[{c}]│[/]",
        f"[{c}]│[/] {nm.ljust(_TW - 2)} [{c}]│[/]",
        f"[{c}]╰{'─' * _TW}╯[/]",
    ]


def _board(refs: list[str]) -> str:
    cells: list[list[str]] = []
    for r in refs:
        g, c = _kind(r)
        cells.append(_tile(Path(r).name, g, c))
    while len(cells) < _COLS or len(cells) % _COLS:
        cells.append(_tile("drop here", "＋", "grey37", dim=True))
        if len(cells) >= _COLS and len(cells) % _COLS == 0:
            break
    rows: list[str] = []
    for i in range(0, len(cells), _COLS):
        grp = cells[i : i + _COLS]
        for ln in range(4):
            rows.append("  ".join(t[ln] for t in grp))
        rows.append("")
    return "\n".join(rows).rstrip()


class TasteBench(App):
    TITLE = "tastebench"

    CSS = """
    Screen { background: $background; align: center middle; }
    #card { width: 96%; max-width: 120; height: auto; }
    #brand { height: auto; color: $accent; text-style: bold;
             text-align: center; margin-bottom: 1; }
    #mode { height: 1; text-style: bold; margin-bottom: 1; }
    #boardwrap {
        height: 1fr; min-height: 12; border: round $surface;
        background: $surface; padding: 1 2; scrollbar-size: 1 1;
    }
    #drop {
        height: 3; border: round $surface; background: $surface;
        padding: 0 2; margin: 1 0;
    }
    #drop:focus { border: round $accent; }
    #out {
        height: 14; border: round $surface; background: $surface;
        padding: 1 2; scrollbar-size: 1 1;
    }
    #status { dock: bottom; height: 1; padding: 0 2;
              background: $panel; color: $text-muted; }
    """

    BINDINGS = [
        Binding("space", "toggle", "Add ⇄ Grade", show=True),
        Binding("b", "brain", "Brain on/off", show=True),
        Binding("x", "clear", "Clear board", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.mode = "ref"
        self.use_brain = False
        self.busy = False

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                with Vertical(id="card"):
                    yield Static(BANNER, id="brand")
                    yield Static(id="mode")
                    with VerticalScroll(id="boardwrap"):
                        yield Static(id="board")
                    yield Input(placeholder="drag file(s) here, or a path "
                                "/ glob, then ⏎", id="drop")
                    yield RichLog(id="out", wrap=True, markup=True,
                                  highlight=False)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#out", RichLog).write(
            "[b]the board is your taste.[/b]  Drop a few things you wish "
            "your work felt like — they pin to the board.  Press "
            "[b]space[/] → [magenta]GRADE[/], drop your draft, and it's "
            "graded against the board.  [dim]b = TRIBE neural read.[/]")

    def _refresh(self) -> None:
        n = len(self.refs)
        if self.mode == "ref":
            self.query_one("#mode", Static).update(
                "[reverse] REFERENCE [/]  drop work you admire — it pins to "
                "the board   [dim]space → grade[/]")
        else:
            self.query_one("#mode", Static).update(
                f"[reverse] GRADE [/]  drop your draft — graded against the "
                f"[b]{n}[/]-tile board   [dim]space → reference[/]")
        self.query_one("#board", Static).update(_board(self.refs))
        brain = "[magenta]brain ON[/]" if self.use_brain else \
            "[dim]brain off · craft[/]"
        state = "[yellow]⟳ grading…[/]" if self.busy else "ready"
        self.query_one("#status", Static).update(
            f"▸ {n} on the board · {brain} · {state}   "
            f"[dim]space add⇄grade · b brain · x clear · q quit[/]")

    def on_input_submitted(self, e: Input.Submitted) -> None:
        files = _parse_drop(e.value)
        e.input.value = ""
        if not files:
            self.notify("no media in that drop/path", severity="warning")
            return
        if self.mode == "ref":
            for f in files:
                if f not in self.refs:
                    self.refs.append(f)
            self.notify(f"+{len(files)} pinned to the board")
            self._refresh()
            return
        if not self.refs:
            self.notify("pin some references first (space → REFERENCE)",
                        severity="error")
            return
        if self.busy:
            self.notify("still grading…", severity="warning")
            return
        self.busy = True
        self._refresh()
        cand = files[0]
        self.query_one("#out", RichLog).write(
            f"\n[dim]── grading {Path(cand).name}"
            f"{' (brain — minutes)' if self.use_brain else ''} ──[/]")
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
                    "\n[b]biggest gaps vs the board[/]\n" + "\n".join(
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
        self.query_one("#out", RichLog).write(
            f"[red]grading failed:[/] {msg}")
        self.busy = False
        self._refresh()
        self.notify("grading failed — see the panel", severity="error")

    def action_toggle(self) -> None:
        self.mode = "grade" if self.mode == "ref" else "ref"
        self._refresh()

    def action_brain(self) -> None:
        self.use_brain = not self.use_brain
        self._refresh()
        if self.use_brain:
            self.notify("brain ON — needs the ~20GB model; first grade "
                        "loads it (minutes on M-series). Craft fallback "
                        "if absent.", timeout=6)

    def action_clear(self) -> None:
        self.refs.clear()
        self._refresh()
        self.notify("board cleared")


# back-compat alias (older imports referenced TasteApp)
TasteApp = TasteBench


def run_app() -> int:
    TasteBench().run()
    return 0
