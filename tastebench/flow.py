"""tastebench.flow — the default experience: a prompt you drop files into.

This is a *line prompt*, not a full-screen app — and that's the whole
point. Dragging a file from Finder onto a terminal pastes its path **at a
prompt**. A full-screen (alt-screen) TUI never receives that, which is why
the Textual version's drag-drop didn't work. Here it works because it's
the same mechanism every other terminal tool uses.

Flow: welcome → drop the work you admire (auto-processed through TRIBE
as it lands, and it stays visible on a numbered board) → either drop a
separate draft after `grade`, or `grade N` to lift item N off the board
and grade it against the rest; `rm N` undoes a mis-drop. A drop
auto-submits the instant it lands (no Enter — bracketed-paste detection).
When the weights are missing we offer to fetch them in the background
with a live progress bar instead of making you type a command.

M-series-faithful env is set on import (mps / bf16 text / whisperx f32).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
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
[dim] Your own focus group for your drafts[/]"""


_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"


def _supports_raw() -> bool:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except Exception:
        return False
    return True


def _read_drop_or_line(console, prompt: str) -> str:
    """Print `prompt`, then return either a typed line (submitted on Enter)
    **or** a drag/paste — auto-submitted the instant the paste completes,
    no Enter. A Finder drag arrives wrapped in the terminal's
    bracketed-paste envelope (ESC[200~ … ESC[201~); we enable that mode,
    read the tty in cbreak, and return the moment the envelope closes.

    Any termios/echo trouble falls back to a plain Enter-required prompt,
    so this can only ever be as good as before, never worse."""
    import os
    import select
    import termios
    import tty

    console.print(prompt, end="")
    sys.stdout.flush()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    out = sys.stdout
    typed: list[str] = []
    buf = b""

    def _more(block: bool = True) -> bytes:
        if not block and not select.select([fd], [], [], 0.06)[0]:
            return b""
        return os.read(fd, 4096)

    try:
        tty.setcbreak(fd)
        out.write("\x1b[?2004h")  # ask the terminal to bracket pastes
        out.flush()
        while True:
            if not buf:
                chunk = os.read(fd, 4096)
                if not chunk:
                    raise EOFError
                buf += chunk
            b, buf = buf[:1], buf[1:]

            if b == b"\x1b":
                seq = b
                while True:
                    if not buf:
                        nxt = _more(block=False)
                        if not nxt:
                            break          # lone ESC — ignore, don't hang
                        buf += nxt
                    nb, buf = buf[:1], buf[1:]
                    seq += nb
                    if nb == b"~" or nb.isalpha() or len(seq) > 8:
                        break
                if seq == _PASTE_START:
                    paste = b""
                    while not paste.endswith(_PASTE_END):
                        if not buf:
                            buf += _more()
                        paste += buf[:1]
                        buf = buf[1:]
                    text = paste[: -len(_PASTE_END)].decode(
                        "utf-8", "ignore")
                    out.write(text + "\n")
                    out.flush()
                    return text            # ← the drop, no Enter needed
                continue                   # arrow keys etc. — ignore

            if b in (b"\r", b"\n"):
                out.write("\n")
                out.flush()
                return "".join(typed)
            if b == b"\x03":
                raise KeyboardInterrupt
            if b == b"\x04":
                raise EOFError
            if b in (b"\x7f", b"\x08"):
                if typed:
                    typed.pop()
                    out.write("\b \b")
                    out.flush()
                continue
            try:
                ch = b.decode("utf-8")
            except UnicodeDecodeError:
                continue                   # stray multibyte while typing
            typed.append(ch)
            out.write(ch)
            out.flush()
    except (termios.error, OSError, ValueError):
        # terminal didn't cooperate — degrade to Enter-required input
        return console.input("")
    finally:
        try:
            out.write("\x1b[?2004l")
            out.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:  # noqa: BLE001
            pass


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


_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_WEIGHTS_EST = 20 * 1024 ** 3   # ~20 GB total — drives the progress bar


def _glyph(p: str) -> tuple[str, str]:
    e = Path(p).suffix.lower()
    if e in _AUDIO:
        return "♪", "cyan"
    if e in _IMAGE:
        return "▦", "magenta"
    return "▶", "green"


def _print_board(console, refs: list[str], cache: dict) -> None:
    """The persistent, numbered view of what you've dropped + its status."""
    if not refs:
        console.print(
            "[dim]  board is empty — drop the work you admire below.[/]\n")
        return
    console.print("[dim]  ── your board " + "─" * 26 + "[/]")
    for i, p in enumerate(refs, 1):
        g, c = _glyph(p)
        sig = cache.get(p)
        if sig is None:
            tag = "[grey46]· not processed[/]"
        elif sig.get("brain", {}).get("available"):
            tag = "[magenta]✓ brain[/]"
        elif sig.get("craft", {}).get("available"):
            tag = "[cyan]✓ craft[/]"
        else:
            tag = "[red]✗ unreadable[/]"
        ext = Path(p).suffix.lower().lstrip(".").upper() or "?"
        nm = Path(p).name
        if len(nm) > 30:
            nm = nm[:29] + "…"
        console.print(
            f"  [bold]{i:>2}[/] [{c}]{g}[/] [dim]{ext:>4}[/]  "
            f"{nm:<31}{tag}")
    console.print("[dim]  " + "─" * 40 + "[/]\n")


def _dir_bytes(d: Path) -> int:
    tot = 0
    try:
        for root, _dirs, files in os.walk(d):
            for f in files:
                try:
                    tot += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return tot


class _DL:
    """A backgrounded run of scripts/download_models.py (log → cache dir)."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._fh = None
        self.log = Path(os.path.expanduser(
            "~/.cache/tastebench/download.log"))

    def start(self) -> tuple[bool, str]:
        script = (Path(__file__).resolve().parent.parent
                  / "scripts" / "download_models.py")
        if not script.exists():
            return False, f"download script not found: {script}"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log, "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=self._fh, stderr=subprocess.STDOUT)
        return True, str(self.log)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def finished(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None


def _watch_download(console, dl: _DL, cache_dir: Path) -> None:
    """Animate a real progress bar (cache bytes / ~20 GB) until the
    download finishes — Ctrl-C hides the bar but keeps it running."""
    from rich.progress import (BarColumn, DownloadColumn, Progress,
                               TextColumn, TimeElapsedColumn)

    console.print("[dim]  fetching in the background — Ctrl-C to hide this "
                  "bar and start dropping (the download keeps going).[/]")
    try:
        with Progress(
            TextColumn("[cyan]⬇ brain weights[/]"),
            BarColumn(bar_width=32),
            TextColumn("[b]{task.percentage:>3.0f}%[/]"),
            DownloadColumn(),
            TimeElapsedColumn(),
            console=console, transient=True,
        ) as pr:
            t = pr.add_task("dl", total=_WEIGHTS_EST)
            while dl.running():
                pr.update(t, completed=min(
                    _dir_bytes(cache_dir), _WEIGHTS_EST - 1))
                time.sleep(1.0)
            pr.update(t, completed=_WEIGHTS_EST)
    except KeyboardInterrupt:
        console.print("[dim]  …still downloading. drop files meanwhile "
                      "(craft read); brain flips on automatically when "
                      "it's done.[/]\n")


def prompt_flow(_input=None) -> int:
    """The drop prompt. `_input` is an injectable line reader for tests."""
    from rich.console import Console

    console = Console()
    interactive = _input is None
    if interactive and not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(
            "[yellow]tastebench's drop prompt needs a real terminal.[/] "
            "Use the CLI instead, e.g.:\n"
            "  [cyan]tastebench compare ref1.wav ref2.wav --to demo.wav[/]")
        return 0
    if _input is not None:
        ask = _input
    elif _supports_raw():
        ask = lambda p: _read_drop_or_line(console, p)  # noqa: E731
    else:
        ask = console.input

    from .engine import models_available

    have = models_available()
    auto = have                       # automatic TRIBE on iff weights present
    refs: list[str] = []              # absolute media paths, in drop order
    cache: dict[str, dict] = {}       # path -> precomputed signature
    dl: _DL | None = None

    def _cache_dir() -> Path:
        try:
            from .engine import _model_cache_dir
            return _model_cache_dir()
        except Exception:  # noqa: BLE001
            return Path(os.path.expanduser(
                "~/.cache/tastebench/model-cache"))

    def _start_download() -> None:
        nonlocal dl, have, auto
        if have:
            console.print("[green]weights already installed.[/]\n")
            return
        if dl is not None and dl.running():
            _watch_download(console, dl, _cache_dir())
        else:
            dl = _DL()
            ok, msg = dl.start()
            if not ok:
                console.print(f"[red]{msg}[/]\n")
                dl = None
                return
            _watch_download(console, dl, _cache_dir())
        if dl is not None and dl.finished():
            have = models_available()
            auto = have
            console.print("[green]✓ weights ready — auto-TRIBE on.[/]\n"
                          if have else
                          "[yellow]download ended but weights still "
                          f"missing — see {dl.log}[/]\n")

    console.print(BANNER)
    console.print()
    if have:
        console.print("[green]✓ brain weights ready.[/] Every file you drop "
                      "is auto-processed through TRIBE as it lands.")
        try:
            from .engine import describe_video_autoconfig

            _vline = describe_video_autoconfig()
            if _vline:
                console.print(f"[dim]  {_vline}[/]")
        except Exception:
            pass
        console.print()
    elif interactive:
        console.print("[yellow]● brain weights not installed (~20 GB).[/]")
        try:
            ans = ask("  Download them now in the background? [Y/n] ")
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if str(ans).strip().lower() in ("", "y", "yes"):
            _start_download()
        else:
            console.print("[dim]  skipped — drop files for the instant "
                          "craft read; type [b cyan]download[/] anytime "
                          "to fetch the weights.[/]\n")
    else:
        console.print("[yellow]● brain weights not installed — "
                      "craft layer only.[/]\n")

    console.print(
        "[b]Welcome.[/] Drop the work you admire below — drag the files "
        "straight in. A drop is read the instant it lands (no Enter); "
        "typed\ncommands still take Enter. Then [b cyan]grade[/] and drop "
        "your draft, or [b cyan]grade N[/] to grade a board item against "
        "the rest.\n[dim]commands: grade · grade N · rm N · auto · "
        "download · clear · q[/]\n")
    _print_board(console, refs, cache)

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
                    cache[p] = signature_for(p, use_brain=ub)
                except Exception as e:  # noqa: BLE001
                    console.print(f"  [red]✗[/] {Path(p).name}: {e}")
        _print_board(console, refs, cache)

    def _grade(draft: str, against: list[str]) -> None:
        from .compare import compare
        from .profile import build_profile
        from .report import to_verdict
        ub = auto and have
        try:
            with console.status(
                f"[yellow]⚙ grading[/] {Path(draft).name} against "
                f"{len(against)} reference(s) "
                f"{'(brain)' if ub else '(craft)'} …"):
                prof = build_profile(against, use_brain=ub,
                                     precomputed=cache)
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
        if dl is not None and dl.finished() and not have:
            have = models_available()
            auto = have
            if have:
                console.print(
                    "[green]✓ brain weights finished downloading — "
                    "auto-TRIBE is now on.[/]\n")
        try:
            line = ask("[bold cyan]drop ▸[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        parts = line.split()
        low = line.lower()
        head = parts[0].lower() if parts else ""

        if low in ("q", "quit", "exit"):
            return 0
        if low in ("clear", "x"):
            refs.clear()
            cache.clear()
            console.print("[dim]board cleared[/]\n")
            _print_board(console, refs, cache)
            continue
        if low in ("board", "ls", "b"):
            _print_board(console, refs, cache)
            continue
        if low == "auto":
            auto = not auto
            if auto and not have:
                console.print("[yellow]auto-TRIBE on, but no weights yet — "
                              "files get the craft read until the download "
                              "finishes.[/]\n")
            else:
                console.print(
                    f"[dim]auto-TRIBE {'on' if auto else 'off'} "
                    f"{'(brain)' if auto and have else '(craft only)'}[/]\n")
            continue
        if low == "download":
            _start_download()
            continue
        if head in ("rm", "remove", "del"):
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                if 1 <= idx <= len(refs):
                    gone = refs.pop(idx - 1)
                    cache.pop(gone, None)
                    console.print(f"[dim]removed {Path(gone).name}[/]\n")
                else:
                    console.print(f"[red]no board item {idx}.[/]\n")
            else:
                console.print("[yellow]usage: [b]rm N[/] (the number "
                              "from the board)[/]\n")
            _print_board(console, refs, cache)
            continue
        if head in ("grade", "g"):
            if len(parts) >= 2 and parts[1].isdigit():
                idx = int(parts[1])
                if not 1 <= idx <= len(refs):
                    console.print(f"[red]no board item {idx}.[/]\n")
                    continue
                draft = refs[idx - 1]
                others = [r for r in refs if r != draft]
                if not others:
                    console.print("[red]nothing to grade against — that's "
                                  "the only thing on the board. Drop more, "
                                  "or use plain [b]grade[/] with a separate "
                                  "draft.[/]\n")
                    continue
                console.print(f"[dim]grading board #{idx} "
                              f"({Path(draft).name}) against the other "
                              f"{len(others)}[/]")
                _grade(draft, others)
                continue
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
            _grade(df[0], refs)
            continue

        files = _parse_drop(line)
        if not files:
            console.print("[yellow]no local media there — drop a file, "
                          "not a URL.[/]\n")
            continue
        _process(files)
