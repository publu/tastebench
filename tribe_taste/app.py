"""tribe_taste.app — the full-screen Textual app (the good TUI).

Workflow-shaped, not a chat box: curate a reference set, set a demo, run
an analysis, read the result, tweak, re-run. Drag files from Finder onto
either input (the terminal pastes the path — single or multi) or browse
with the keyboard. Analyses run on a worker thread so the UI never
freezes. Tuned for Apple Silicon (MPS, bf16 text, whisperx-float32 — the
verified-faithful TRIBE config) — the M-series env is set on import.

Launched by bare `tribe-taste` (or `tribe-taste tui`) in a real terminal.
Non-TTY / no Textual → callers fall back to the rich flow.
"""

from __future__ import annotations

import os
import shlex
from glob import glob
from pathlib import Path

# --- M-series-faithful perf env (set BEFORE any tribe engine import) -------
# device auto->mps, bf16 text (fp16 collapses corr to 0.64), whisperx
# float32, MPS CPU-fallback. setdefault so an explicit override wins.
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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    DirectoryTree, Footer, Input, RichLog, Static,
)

_MEDIA = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
}


def _parse_drop(raw: str) -> list[str]:
    """A dropped/typed string -> existing media paths.

    Finder drag pastes a path; multi-select pastes several, space-joined,
    spaces backslash-escaped (sometimes quoted). Handle all + globs.
    """
    raw = raw.strip()
    if not raw:
        return []
    toks: list[str]
    try:
        toks = shlex.split(raw)
    except ValueError:
        toks = raw.split()
    out: list[str] = []
    for t in toks:
        t = os.path.expanduser(t.strip().strip("'\""))
        cands = glob(t) if any(c in t for c in "*?[") else [t]
        for c in cands:
            if os.path.isfile(c) and Path(c).suffix.lower() in _MEDIA:
                ap = os.path.abspath(c)
                if ap not in out:
                    out.append(ap)
    return out


class _MediaTree(DirectoryTree):
    """Browser: directories + media files only."""

    def filter_paths(self, paths):
        return [
            p for p in paths
            if (p.is_dir() and not p.name.startswith("."))
            or p.suffix.lower() in _MEDIA
        ]


class TasteApp(App):
    TITLE = "tribe-taste"
    SUB_TITLE = "an MRI for your taste"

    CSS = """
    Screen { background: $background; }
    #brand {
        height: 1; content-align: left middle; padding: 0 2;
        color: $accent; text-style: bold;
    }
    #body { height: 1fr; }
    #left { width: 38%; border-right: tall $surface; }
    #tree { height: 1fr; }
    #refs_in, #demo_in { margin: 0 1; }
    #right { width: 1fr; }
    #set {
        height: auto; max-height: 9; padding: 0 2;
        border-bottom: tall $surface; color: $text;
    }
    #out { height: 1fr; padding: 0 1; background: $background; }
    #status {
        height: 1; dock: bottom; padding: 0 2;
        background: $panel; color: $text-muted;
    }
    Input { border: tall $surface; }
    Input:focus { border: tall $accent; }
    .lbl { color: $text-muted; text-style: bold; }
    """

    BINDINGS = [
        Binding("v", "run('vibe')", "Vibe", show=True),
        Binding("c", "run('compare')", "Compare", show=True),
        Binding("o", "run('optimize')", "Optimize", show=True),
        Binding("p", "run('profile')", "Profile", show=True),
        Binding("b", "brain", "Brain on/off", show=True),
        Binding("d", "set_demo_sel", "Tree→Demo", show=True),
        Binding("x", "clear", "Clear refs", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.demo: str | None = None
        self.use_brain = False
        self.busy = False

    def compose(self) -> ComposeResult:
        yield Static("◢ tribe-taste — focus group for the work you make alone", id="brand")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield _MediaTree(os.getcwd(), id="tree")
                yield Static("references — drag files here ▸", classes="lbl")
                yield Input(placeholder="drop files / path / glob, ⏎", id="refs_in")
                yield Static("demo ▸", classes="lbl")
                yield Input(placeholder="drop your draft, ⏎", id="demo_in")
            with Vertical(id="right"):
                yield Static(id="set")
                yield RichLog(id="out", wrap=True, markup=True, highlight=False)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        out = self.query_one("#out", RichLog)
        out.write("[b]Welcome.[/b] Drag a few reference files onto "
                  "[cyan]references[/], drag your draft onto [cyan]demo[/], "
                  "then press [b]v[/b] (vibe) or [b]c[/b] (compare).")
        out.write("[dim]Brain layer is off by default (instant craft). "
                  "Press [b]b[/b] to enable the TRIBE neural read.[/dim]")

    # ---- state / status ---------------------------------------------------
    def _refresh(self) -> None:
        rl = "\n".join(f"  [green]•[/] {Path(r).name}" for r in self.refs[-6:]) \
            or "  [dim]none yet[/]"
        more = f"  [dim]… +{len(self.refs)-6} more[/]" if len(self.refs) > 6 else ""
        dm = Path(self.demo).name if self.demo else "[dim]none[/]"
        self.query_one("#set", Static).update(
            f"[b cyan]REFERENCES[/] ({len(self.refs)})\n{rl}{more}\n"
            f"[b cyan]DEMO[/]  {dm}")
        brain = "[magenta]brain ON[/]" if self.use_brain else "[dim]brain off · craft[/]"
        st = "[yellow]⟳ analyzing…[/]" if self.busy else "ready"
        self.query_one("#status", Static).update(
            f"▸ {len(self.refs)} refs · demo {dm} · {brain} · {st}   "
            f"[dim]v vibe  c compare  o optimize  p profile  b brain  q quit[/]")

    # ---- input / tree (drag-drop + browse) --------------------------------
    def on_input_submitted(self, e: Input.Submitted) -> None:
        files = _parse_drop(e.value)
        e.input.value = ""
        if not files:
            self.notify("no media files in that drop/path", severity="warning")
            return
        if e.input.id == "demo_in":
            self.demo = files[0]
            self.notify(f"demo → {Path(self.demo).name}")
        else:
            for f in files:
                if f not in self.refs:
                    self.refs.append(f)
            self.notify(f"+{len(files)} reference(s)")
        self._refresh()

    def on_directory_tree_file_selected(
        self, e: "DirectoryTree.FileSelected"
    ) -> None:
        p = os.path.abspath(str(e.path))
        if Path(p).suffix.lower() not in _MEDIA:
            return
        if p not in self.refs:
            self.refs.append(p)
            self._refresh()
            self.notify(f"+ref {Path(p).name}  ([b]d[/] = use as demo)")
        self._last_tree = p

    def action_set_demo_sel(self) -> None:
        p = getattr(self, "_last_tree", None)
        if p:
            self.demo = p
            self._refresh()
            self.notify(f"demo → {Path(p).name}")

    def action_clear(self) -> None:
        self.refs.clear()
        self._refresh()
        self.notify("references cleared")

    def action_brain(self) -> None:
        self.use_brain = not self.use_brain
        self._refresh()
        if self.use_brain:
            self.notify("brain ON — needs the ~20GB model; first run loads "
                        "it (minutes on M-series). Falls back to craft if absent.",
                        timeout=6)

    # ---- run an analysis (worker thread → UI stays live) ------------------
    def action_run(self, kind: str) -> None:
        if self.busy:
            self.notify("already analyzing…", severity="warning")
            return
        if not self.refs:
            self.notify("add references first (drag onto 'references')",
                        severity="error")
            return
        if kind != "profile" and not self.demo:
            self.notify("set a demo first (drag onto 'demo')", severity="error")
            return
        self.busy = True
        self._refresh()
        self.query_one("#out", RichLog).write(
            f"\n[dim]── running {kind}"
            f"{' (brain — this can take minutes)' if self.use_brain else ''} ──[/]")
        self._analyze(kind)

    @work(thread=True, exclusive=True, group="analyze")
    def _analyze(self, kind: str) -> None:
        try:
            from rich.markdown import Markdown
            from rich.text import Text

            from . import tui
            from .compare import compare
            from .optimize import optimize
            from .profile import build_profile, profile_summary
            from .report import to_markdown, to_verdict

            ub = self.use_brain
            items = []
            if kind == "profile":
                prof = build_profile(self.refs, use_brain=ub)
                pay = profile_summary(prof)
                pay["_kind"] = "profile"
                items = [Markdown(to_markdown(pay))]
            elif kind == "vibe":
                pay = compare(self.demo, self.refs, use_brain=ub)
                pay["_kind"] = "compare"
                items = [Text.from_markup(to_verdict(pay))]
            elif kind == "optimize":
                pay = optimize(self.demo, self.refs, use_brain=ub)
                pay["_kind"] = "optimize"
                items = [Markdown(to_markdown(pay))]
            else:  # compare — the full visual
                prof = build_profile(self.refs, use_brain=ub)
                cmp = compare(self.demo, prof, use_brain=ub)
                opt = optimize(self.demo, prof, use_brain=False)
                items = [
                    tui._header(cmp, prof["n_refs"]),
                    tui._dial(cmp.get("overall_distance")),
                    tui._heatmap(cmp),
                    tui._craft_table(cmp),
                    tui._edits_panel(opt),
                ]
            self.call_from_thread(self._show, items)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, repr(exc))

    def _show(self, items: list) -> None:
        out = self.query_one("#out", RichLog)
        for it in items:
            out.write(it)
        self.busy = False
        self._refresh()

    def _fail(self, msg: str) -> None:
        self.query_one("#out", RichLog).write(f"[red]analysis failed:[/] {msg}")
        self.busy = False
        self._refresh()
        self.notify("analysis failed — see the results pane", severity="error")


def run_app() -> int:
    """Entry point used by the CLI. Returns 0."""
    TasteApp().run()
    return 0
