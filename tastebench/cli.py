"""tastebench.cli — the command-line entry point.

    tastebench                               the folder worker (default)
    tastebench worker [DIR]                  the folder worker, explicitly
    tastebench drop                          the drag-into-terminal prompt
    tastebench vibe      <demo> --like <refs...>    one-command verdict
    tastebench profile  <refs...>            build a taste signature
    tastebench compare   <refs...> --to <demo>   demo vs taste
    tastebench optimize  <demo> --toward <refs...>  ranked edits
    tastebench glossary  [TERM]              the explainer dictionary
    tastebench tui       ...                 the product TUI

Bare `tastebench` launches the **worker**: it creates and watches a
`tastebench/references/<name>/{refs,draft}/` tree and auto-grades every
draft against that taste's refs — no CLI verbs to learn.

`vibe` is the fast path: one screen — verdict + the single biggest lever.
Add `--deep` for the full report or `--fix` for the ranked edit list.

Every analysis command takes:
    --format {markdown,json}   output format (default markdown)
    --llm                      emit the LLM-ready bundle instead
    --no-brain                 skip the TRIBE model (craft layer only)
    -o / --out FILE            write to a file instead of stdout
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="output format (default: markdown)",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="emit a self-contained LLM bundle (raw numbers + full glossary "
        "+ framing question) instead of the formatted report",
    )
    p.add_argument(
        "--no-brain",
        action="store_true",
        help="skip the TRIBE brain model; use the craft layer only",
    )
    p.add_argument(
        "-o",
        "--out",
        metavar="FILE",
        help="write output to FILE instead of stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tastebench",
        description=(
            "Learn the taste signature of media you admire, then see how "
            "your own demo diverges and what to change. Craft layer is "
            "model-free; the brain layer needs the TRIBE model "
            "(scripts/download_models.py)."
        ),
    )
    ap.add_argument("--version", action="version", version=f"tastebench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=False)

    p_prof = sub.add_parser("profile", help="build a taste profile from references")
    p_prof.add_argument("refs", nargs="+", help="reference media files")
    _add_common(p_prof)

    p_cmp = sub.add_parser("compare", help="compare a demo against references")
    p_cmp.add_argument("refs", nargs="+", help="reference media files")
    p_cmp.add_argument(
        "--to", required=True, metavar="DEMO", help="the demo to compare"
    )
    _add_common(p_cmp)

    p_opt = sub.add_parser("optimize", help="ranked edits toward a taste")
    p_opt.add_argument("demo", help="the demo to optimize")
    p_opt.add_argument(
        "--toward",
        required=True,
        nargs="+",
        metavar="REF",
        help="reference media files to move toward",
    )
    p_opt.add_argument(
        "--top", type=int, default=8, help="max edits to show (default 8)"
    )
    _add_common(p_opt)

    p_vibe = sub.add_parser(
        "vibe", help="one-command verdict: how your demo sits vs a taste"
    )
    p_vibe.add_argument("demo", help="your demo (audio/video/image)")
    p_vibe.add_argument(
        "--like",
        required=True,
        nargs="+",
        metavar="REF",
        help="reference media you admire",
    )
    p_vibe.add_argument(
        "--deep",
        action="store_true",
        help="full report instead of the one-screen verdict",
    )
    p_vibe.add_argument(
        "--fix",
        action="store_true",
        help="show the ranked edit list instead of the verdict",
    )
    p_vibe.add_argument(
        "--top", type=int, default=8, help="max edits with --fix (default 8)"
    )
    _add_common(p_vibe)

    p_web = sub.add_parser(
        "web",
        help="record a live URL (silent screen capture) and grade it vs a "
        "taste — the web-QA front-end",
    )
    p_web.add_argument("url", help="http(s) URL to capture")
    p_web.add_argument(
        "--like", nargs="+", metavar="REF", default=None,
        help="reference media you admire (omit → just save the recording)",
    )
    p_web.add_argument(
        "--seconds", type=float, default=12.0,
        help="capture / scroll duration (default 12)",
    )
    p_web.add_argument(
        "--mp4", metavar="PATH", default=None,
        help="where to save the recording (default ./<site>.mp4)",
    )
    p_web.add_argument(
        "--deep", action="store_true",
        help="full report instead of the one-screen verdict",
    )
    _add_common(p_web)

    p_gloss = sub.add_parser("glossary", help="print the explainer dictionary")
    p_gloss.add_argument(
        "term", nargs="?", help="a single term to explain in full"
    )
    p_gloss.add_argument(
        "--json", action="store_true", help="emit the raw glossary JSON"
    )

    p_wk = sub.add_parser(
        "worker", help="watch a references/<name>/{refs,draft}/ tree and "
        "auto-grade (the default when you run bare `tastebench`)"
    )
    p_wk.add_argument(
        "root", nargs="?", default=None,
        help="working dir to create/watch (default: ./tastebench)",
    )
    p_wk.add_argument(
        "--no-brain", action="store_true",
        help="craft layer only; never use / download the TRIBE model",
    )

    sub.add_parser(
        "drop", help="the legacy drag-files-into-the-terminal prompt"
    )

    p_tui = sub.add_parser(
        "tui", help="interactive TUI (omit refs); or pass refs for a one-shot view"
    )
    p_tui.add_argument(
        "refs", nargs="*", help="reference media files (omit → interactive)"
    )
    p_tui.add_argument(
        "--demo", metavar="DEMO", help="optional demo to overlay/compare"
    )
    p_tui.add_argument(
        "--no-brain", action="store_true", help="craft layer only"
    )

    return ap


def _emit(text: str, out: str | None) -> None:
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


def _launch_worker(root=None, use_brain=None) -> int:
    """Bare `tastebench` → the folder worker.

    The default experience: run it once, drop files into the
    `references/<name>/{refs,draft}/` tree it creates, and drafts get
    graded automatically. No CLI verbs to learn."""
    from .worker import run

    return run(root, use_brain=use_brain)


def _launch_interactive() -> int:
    """`tastebench drop` → the legacy drop prompt.

    A *line prompt*, not a full-screen app, on purpose: a Finder file-drag
    only pastes its path at a prompt, never into an alt-screen TUI — so
    this is the form where drag-drop actually works. `prompt_flow` prints
    the CLI hint itself when there is no TTY."""
    from .flow import prompt_flow

    return prompt_flow()


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.cmd is None:
        return _launch_worker()

    if args.cmd == "worker":
        return _launch_worker(
            args.root, use_brain=False if args.no_brain else None
        )

    if args.cmd == "drop":
        return _launch_interactive()

    if args.cmd == "glossary":
        from . import explainers

        if args.json:
            import json

            print(json.dumps(explainers.load_all(), indent=2))
        else:
            print(explainers.glossary_text(args.term))
        return 0

    if args.cmd == "tui":
        from . import tui

        if not args.refs:
            return _launch_interactive()
        return tui.run(args.refs, demo=args.demo, use_brain=not args.no_brain)

    if args.cmd == "web":
        from .webcap import WebCaptureUnavailable, capture_site

        try:
            mp4 = capture_site(args.url, args.mp4, seconds=args.seconds)
        except WebCaptureUnavailable as e:
            print(str(e), file=sys.stderr)
            return 3
        except ValueError as e:
            ap.error(str(e))
            return 2
        print(f"recorded {mp4}", file=sys.stderr)
        if not args.like:
            print(mp4)
            return 0
        from . import report as _report
        from .compare import compare

        payload = compare(str(mp4), args.like, use_brain=not args.no_brain)
        payload["_kind"] = "compare"
        brief = not args.deep and not args.llm and args.format == "markdown"
        text = (
            _report.to_verdict(payload)
            if brief
            else _report.render(payload, fmt=args.format, llm=args.llm)
        )
        _emit(text, args.out)
        return 0

    use_brain = not args.no_brain
    from . import report as _report

    if args.cmd == "profile":
        from .profile import build_profile, profile_summary

        prof = build_profile(args.refs, use_brain=use_brain)
        payload = profile_summary(prof)
        payload["_kind"] = "profile"

    elif args.cmd == "compare":
        from .compare import compare

        payload = compare(args.to, args.refs, use_brain=use_brain)
        payload["_kind"] = "compare"

    elif args.cmd == "optimize":
        from .optimize import optimize

        payload = optimize(
            args.demo, args.toward, use_brain=use_brain, top=args.top
        )
        payload["_kind"] = "optimize"
    elif args.cmd == "vibe":
        if args.fix:
            from .optimize import optimize

            payload = optimize(
                args.demo, args.like, use_brain=use_brain, top=args.top
            )
            payload["_kind"] = "optimize"
        else:
            from .compare import compare

            payload = compare(args.demo, args.like, use_brain=use_brain)
            payload["_kind"] = "compare"
    else:  # pragma: no cover
        ap.error(f"unknown command {args.cmd}")
        return 2

    brief = (
        args.cmd == "vibe"
        and not getattr(args, "deep", False)
        and not getattr(args, "fix", False)
        and not args.llm
        and args.format == "markdown"
    )
    text = (
        _report.to_verdict(payload)
        if brief
        else _report.render(payload, fmt=args.format, llm=args.llm)
    )
    _emit(text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
