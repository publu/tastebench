"""tastebench.worker — the folder worker (the default experience).

You run *one thing* and never touch a CLI verb again. The worker creates
and then watches a folder tree::

    tastebench/
      references/
        <taste-name>/
          refs/    ← drop the work you admire here
          draft/   ← drop the draft you want graded here

Each ``references/<name>/`` is one self-contained experiment. Whatever is
in ``refs/`` defines that taste; whatever is in ``draft/`` is graded
against it. Drop, wait, read — the worker profiles the refs, grades each
draft against them, prints the verdict live, and writes a full
``<draft-stem>.report.md`` next to the taste so you can just open the file.

It is poll-based (no extra dependency) and *settle-aware*: a file is only
acted on once its size/mtime has held steady for a poll, so a half-copied
or multi-file drag never triggers a partial run. A file's combined
(refs+draft) signature is cached, so nothing is re-graded until it
actually changes.

Brain layer is automatic: TRIBE if the ~20 GB weights are present
(offered as a background download on first run, flips on the moment it
finishes), craft-only otherwise. Importing ``tastebench.flow`` sets the
M-series-faithful env (mps / bf16 text / whisperx f32) and is also where
the background-download helper lives, so it is reused here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Reuses flow's media-extension set + background-download helper, and (as a
# side effect of import) its M-series-faithful env defaults.
from .flow import _MEDIA, _DL

DEFAULT_ROOT = "tastebench"
_REPORT_SUFFIX = ".report.md"
_POLL_SECONDS = 2.0

_SKELETON_README = """\
tastebench — how this works
===========================

This folder is watched. You don't run any commands.

  references/<a-name-you-choose>/
      refs/     put a few tracks / videos / images you ADMIRE in here
      draft/    put YOUR draft you want graded in here

Each references/<name>/ is one experiment. The worker learns the taste
shared by everything in refs/, then grades every file in draft/ against
it — printing the verdict in the worker window and writing a full
<draft-name>.report.md next to this file.

Make as many references/<name>/ folders as you like; they're independent.
Drop more files anytime — the worker notices and re-grades automatically.
This 'example' folder is just a placeholder; rename it or make your own.
"""


def _media_in(d: Path) -> list[Path]:
    """Sorted media files under ``d`` (recursive); skips dotfiles + reports."""
    if not d.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.name.endswith(_REPORT_SUFFIX):
            continue
        if p.suffix.lower() in _MEDIA:
            out.append(p)
    return out


def _stamp(p: Path) -> tuple[float, int]:
    try:
        st = p.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, -1)


def _sig(paths: list[Path]) -> tuple:
    """Order-independent fingerprint of a file set (path + mtime + size)."""
    return tuple(sorted((str(p), *_stamp(p)) for p in paths))


def _ensure_skeleton(root: Path) -> None:
    refs_root = root / "references"
    if refs_root.is_dir() and any(
        c.is_dir() for c in refs_root.iterdir() if not c.name.startswith(".")
    ):
        return
    example = refs_root / "example"
    (example / "refs").mkdir(parents=True, exist_ok=True)
    (example / "draft").mkdir(parents=True, exist_ok=True)
    readme = refs_root / "HOW-THIS-WORKS.txt"
    if not readme.exists():
        readme.write_text(_SKELETON_README, encoding="utf-8")


def _discover(root: Path) -> list[tuple[str, Path, Path]]:
    """Every ``references/<name>/`` → (name, refs_dir, draft_dir).

    Ensures each taste has both ``refs/`` and ``draft/`` so a freshly
    made-by-hand folder is immediately usable.
    """
    refs_root = root / "references"
    out: list[tuple[str, Path, Path]] = []
    if not refs_root.is_dir():
        return out
    for d in sorted(refs_root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        refs_dir, draft_dir = d / "refs", d / "draft"
        refs_dir.mkdir(exist_ok=True)
        draft_dir.mkdir(exist_ok=True)
        out.append((d.name, refs_dir, draft_dir))
    return out


def scan_once(
    root: Path, state: dict, use_brain: bool, console, ready: dict
) -> list[tuple[str, str, str]]:
    """One settle-aware pass over the tree. Returns the list of
    ``(taste, draft_name, report_path)`` graded this pass (for tests).

    ``state`` carries ``profiles`` (taste → (refs_sig, profile)) and
    ``graded`` ((taste, draft_path) → combined_sig). ``ready`` carries the
    previous poll's per-file stamps so a file is only acted on once its
    size/mtime has held steady across two polls (no half-copied reads).
    """
    from .compare import compare
    from .profile import build_profile
    from .report import to_markdown, to_verdict

    profiles: dict = state.setdefault("profiles", {})
    graded: dict = state.setdefault("graded", {})
    did: list[tuple[str, str, str]] = []

    def _stable(paths: list[Path]) -> list[Path]:
        stable = []
        for p in paths:
            k, now = str(p), _stamp(p)
            was = ready.get(k)
            ready[k] = now
            if was == now and now[1] >= 0:
                stable.append(p)
        return stable

    for name, refs_dir, draft_dir in _discover(root):
        # Settle refs and drafts every pass (even before refs exist) so a
        # draft dropped first still stabilizes in parallel.
        refs = _stable(_media_in(refs_dir))
        drafts = _stable(_media_in(draft_dir))
        if not refs:
            if name not in state.get("warned", set()):
                console.print(
                    f"[grey50]· [bold]{name}[/] — waiting for references "
                    f"in[/] [grey50]{refs_dir}[/]"
                )
                state.setdefault("warned", set()).add(name)
            continue
        state.get("warned", set()).discard(name)

        rsig = _sig(refs)
        cached = profiles.get(name)
        if cached is None or cached[0] != rsig:
            with console.status(
                f"[yellow]⚙ learning taste[/] [bold]{name}[/] "
                f"from {len(refs)} reference(s) "
                f"{'(brain — can take minutes)' if use_brain else '(craft)'}…"
            ):
                try:
                    prof = build_profile(
                        [str(p) for p in refs], use_brain=use_brain
                    )
                except Exception as e:  # noqa: BLE001
                    console.print(f"  [red]✗ {name}: profiling failed:[/] {e}")
                    continue
            profiles[name] = (rsig, prof)
            console.print(
                f"[green]✓[/] taste [bold]{name}[/] learned "
                f"([{len(refs)}] refs)"
            )
        prof = profiles[name][1]

        if not drafts:
            continue
        for draft in drafts:
            key = (name, str(draft))
            csig = (rsig, *_stamp(draft))
            if graded.get(key) == csig:
                continue
            try:
                with console.status(
                    f"[yellow]⚙ grading[/] [bold]{draft.name}[/] vs "
                    f"taste [bold]{name}[/] "
                    f"{'(brain)' if use_brain else '(craft)'}…"
                ):
                    pay = compare(str(draft), prof, use_brain=use_brain)
                    pay["_kind"] = "compare"
            except Exception as e:  # noqa: BLE001
                console.print(f"  [red]✗ {draft.name}: grading failed:[/] {e}")
                continue
            report_path = root / "references" / name / (
                draft.stem + _REPORT_SUFFIX
            )
            try:
                report_path.write_text(to_markdown(pay), encoding="utf-8")
            except OSError as e:
                console.print(f"  [red]✗ couldn't write report:[/] {e}")

            console.print(
                f"\n[dim]── [bold]{name}[/] · {draft.name} "
                + "─" * 18 + "[/]"
            )
            console.print(to_verdict(pay))
            top = [
                r for r in pay.get("craft_deltas", [])
                if abs(r.get("spread_norm", 0)) >= 0.5
            ][:4]
            if top:
                console.print("[b]biggest gaps vs the refs[/]")
                for r in top:
                    console.print(f"  • {r['voice']}")
            console.print(f"[dim]  → full report: {report_path}[/]\n")
            graded[key] = csig
            did.append((name, draft.name, str(report_path)))
    return did


def run(root: str | Path | None = None, use_brain: bool | None = None) -> int:
    """Launch the folder worker. ``use_brain=None`` → auto-detect TRIBE
    weights (and offer a background download on first run)."""
    from rich.console import Console

    from .engine import models_available

    console = Console()
    root = Path(root or DEFAULT_ROOT).expanduser().resolve()
    # Footgun guard: the default workspace name matches the package dir, so
    # running bare `tastebench` from the source checkout would otherwise
    # write the watched tree *into the package*. Refuse and ask for a path.
    if (root / "__init__.py").exists() and (root / "worker.py").exists():
        console.print(
            f"[red]Refusing to use[/] [cyan]{root}[/] [red]— that's the "
            "tastebench package source, not a workspace.[/]\n"
            "Run the worker from another directory, or pass one explicitly:"
            "\n  [cyan]tastebench worker ~/my-project[/]\n"
        )
        return 2
    root.mkdir(parents=True, exist_ok=True)
    _ensure_skeleton(root)

    have = models_available()
    auto = use_brain if use_brain is not None else have
    dl: _DL | None = None

    console.print(
        "[bold cyan]tastebench worker[/] "
        "[grey50]· a private focus group for your drafts[/]\n"
    )
    console.print(f"[b]Watching[/] [cyan]{root}[/]\n")
    console.print(
        "Make a folder per taste under "
        f"[cyan]{root / 'references'}[/]:\n"
        "  [bold]references/<name>/refs/[/]   the work you admire\n"
        "  [bold]references/<name>/draft/[/]  the draft to grade\n"
        "Drop files in, then just watch — drafts are graded automatically "
        "and a\n[bold]<draft>.report.md[/] is written next to each taste. "
        "Ctrl-C to stop.\n"
    )

    if use_brain is False:
        console.print("[grey50]● craft layer only (brain disabled).[/]\n")
    elif have:
        console.print("[green]✓ brain weights ready — TRIBE on.[/]\n")
    else:
        console.print("[yellow]● brain weights not installed (~20 GB).[/]")
        ask = console.input if sys.stdin.isatty() else (lambda _p: "n")
        try:
            a = ask("  Download them now in the background? [Y/n] ")
        except (EOFError, KeyboardInterrupt):
            a = "n"
        if str(a).strip().lower() in ("", "y", "yes"):
            dl = _DL()
            ok, msg = dl.start()
            if ok:
                console.print(
                    "[dim]  fetching in the background — drafts get the "
                    "instant craft read meanwhile; brain flips on "
                    "automatically when it's done.[/]\n"
                )
            else:
                console.print(f"[red]  {msg}[/]\n")
                dl = None
        else:
            console.print(
                "[dim]  craft-only for now — restart anytime to fetch "
                "the weights.[/]\n"
            )

    state: dict = {}
    ready: dict = {}
    brain_on = bool(auto and have)  # craft-only until weights actually exist
    try:
        while True:
            if dl is not None and dl.finished() and not brain_on:
                if use_brain is not False and models_available():
                    brain_on = True
                    state["profiles"] = {}  # rebuild tastes with the brain
                    console.print(
                        "[green]✓ brain weights finished — TRIBE is now on; "
                        "re-grading.[/]\n"
                    )
                else:
                    console.print(
                        "[yellow]download ended but weights still missing — "
                        "staying craft-only.[/]\n"
                    )
                dl = None
            try:
                scan_once(root, state, brain_on, console, ready)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]worker pass failed:[/] {e}")
            time.sleep(_POLL_SECONDS)
    except KeyboardInterrupt:
        console.print("\n[grey50]worker stopped.[/]")
        return 0
