"""tribe_taste.cli — the command-line entry point.

    tribe-taste vibe      <demo> --like <refs...>    one-command verdict
    tribe-taste profile  <refs...>            build a taste signature
    tribe-taste compare   <refs...> --to <demo>   demo vs taste
    tribe-taste optimize  <demo> --toward <refs...>  ranked edits
    tribe-taste glossary  [TERM]              the explainer dictionary
    tribe-taste tui       ...                 the product TUI

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
        prog="tribe-taste",
        description=(
            "Learn the taste signature of media you admire, then see how "
            "your own demo diverges and what to change. Craft layer is "
            "model-free; the brain layer needs the TRIBE model "
            "(scripts/download_models.py)."
        ),
    )
    ap.add_argument("--version", action="version", version=f"tribe-taste {__version__}")
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

    p_gloss = sub.add_parser("glossary", help="print the explainer dictionary")
    p_gloss.add_argument(
        "term", nargs="?", help="a single term to explain in full"
    )
    p_gloss.add_argument(
        "--json", action="store_true", help="emit the raw glossary JSON"
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


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.cmd is None:
        from . import tui

        return tui.interactive()

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
            return tui.interactive()
        return tui.run(args.refs, demo=args.demo, use_brain=not args.no_brain)

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
