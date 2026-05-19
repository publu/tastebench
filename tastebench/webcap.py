"""tastebench.webcap — turn a live URL into a silent screen recording.

The "web QA" front-end: drive a real browser to the page, autoscroll it
top-to-bottom, record the viewport to ``webm`` (Playwright's built-in
context recorder), and transcode to a silent ``mp4`` that the normal
video path then grades like any other clip. A website *is* a video to
this tool — its motion/layout/density/contrast is its taste signature.

Silent by design: the QA target is the *visual experience*, there is no
audio track, so the brain pipeline auto-drops the audio/text extractors
and runs video-only (no transcription — see README "Hardware reality").

Heavy deps are optional and lazily imported, exactly like the brain
layer: the package imports and the craft/CLI work with no browser
installed. ``WebCaptureUnavailable`` carries the one-time setup steps.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

_DEFAULT_SECONDS = 12.0
# Browser *viewport* — the page must lay out like a real desktop so its
# visual signature (density, layout, contrast) is the real thing.
_DEFAULT_W = 1280
_DEFAULT_H = 720
_DEFAULT_FPS = 24
# The recorded *video* is a separate size. The TRIBE video extractor
# (`vjepa2-vitg-fpc64-256`) is natively 256 px; recording at full 720p
# just feeds it pixels it throws away — and on a CUDA / >=128 GiB box the
# engine's RAM-aware cap is intentionally off, so those wasted frames go
# in at full res (issue #1). So we render the page large but record the
# video downscaled to the encoder's native short side. 0 disables it.
_ENCODER_PX = 256


def _record_size(vw: int, vh: int, cap: int = _ENCODER_PX) -> dict:
    """Viewport (vw, vh) -> recorded video size, short side capped to
    ``cap``, aspect preserved, both dims even (libx264 yuv420p needs it).
    ``cap <= 0`` records at the viewport size (full fidelity)."""
    if cap <= 0 or min(vw, vh) <= cap:
        rw, rh = vw, vh
    else:
        scale = cap / float(min(vw, vh))
        rw, rh = round(vw * scale), round(vh * scale)
    rw -= rw & 1
    rh -= rh & 1
    return {"width": max(2, rw), "height": max(2, rh)}

_SETUP_HINT = (
    "Web QA needs a browser engine (not a core dep, like the brain layer):\n"
    "  - pip install 'tastebench[web]'\n"
    "  - playwright install chromium\n"
    "Then: tastebench web https://example.com --like <refs…>"
)


class WebCaptureUnavailable(RuntimeError):
    """Playwright / its chromium build / ffmpeg is not installed."""


def url_to_stem(url: str) -> str:
    """A filesystem-safe slug for an URL (used for default output names
    and for the worker's ``<stem>.mp4`` sibling)."""
    s = re.sub(r"^https?://", "", url.strip().rstrip("/"))
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return (s or "site")[:80]


def _require(tool_ok: bool, what: str) -> None:
    if not tool_ok:
        raise WebCaptureUnavailable(f"{what} not found.\n{_SETUP_HINT}")


def capture_site(
    url: str,
    out_path: str | Path | None = None,
    *,
    seconds: float = _DEFAULT_SECONDS,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
    fps: int = _DEFAULT_FPS,
    record_px: int = _ENCODER_PX,
) -> Path:
    """Record ``url`` to a silent mp4 and return its path.

    ``seconds`` is the total capture/scroll duration. The page is loaded,
    given a moment to settle, then smooth-scrolled top→bottom over the
    window so the recording sweeps the whole page. No audio is recorded.

    The page renders at ``width``×``height`` (desktop layout) but the
    recorded video is downscaled so its short side is ``record_px``
    (default 256, the TRIBE video encoder's native size) — this stops
    every brain path, including the uncapped CUDA / ≥128 GiB one, from
    ingesting wasted full-res frames. ``record_px=0`` records full size.

    Raises ``WebCaptureUnavailable`` if Playwright (and its chromium
    build) or ffmpeg are missing, ``ValueError`` for a non-http(s) URL.
    """
    if not re.match(r"^https?://", url.strip(), re.I):
        raise ValueError(f"not an http(s) URL: {url!r}")
    _require(shutil.which("ffmpeg") is not None, "ffmpeg")
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise WebCaptureUnavailable(f"playwright import failed: {exc!r}\n{_SETUP_HINT}")

    if out_path is None:
        out_path = Path.cwd() / f"{url_to_stem(url)}.mp4"
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        vdir = Path(td)
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    record_video_dir=str(vdir),
                    # Page lays out at full viewport; the recording is
                    # downscaled to the encoder's native scale so no path
                    # (incl. CUDA / >=128 GiB) is fed wasted full-res
                    # frames (issue #1). record_px<=0 keeps full size.
                    record_video_size=_record_size(width, height, record_px),
                )
                page = context.new_page()
                page.goto(url, wait_until="load", timeout=60_000)
                try:  # let late content/fonts settle, but never hang on it
                    page.wait_for_load_state("networkidle", timeout=8_000)
                except Exception:  # noqa: BLE001
                    pass
                _autoscroll(page, seconds)
                context.close()  # finalizes the webm
                browser.close()
            webms = list(vdir.glob("*.webm"))
            if not webms:
                raise WebCaptureUnavailable(
                    "Playwright produced no recording (chromium build "
                    f"missing?).\n{_SETUP_HINT}"
                )
            _to_mp4(webms[0], out_path, fps)
        except WebCaptureUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            # A missing chromium build surfaces here as a launch error.
            msg = repr(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                raise WebCaptureUnavailable(
                    f"chromium build not installed: {msg}\n{_SETUP_HINT}"
                )
            raise
    return out_path


# One scroll step. Robust to the common failure mode: many sites scroll
# an inner container (SPAs/dashboards), not the document — so pick the
# largest genuinely-scrollable target each step, force instant (not
# smooth) scrolling, and report whether it actually advanced.
_STEP_JS = """() => {
  const seen = new Set();
  const cands = [document.scrollingElement, document.documentElement,
                 document.body];
  let best = null, bestGain = 0;
  const consider = (el) => {
    if (!el || seen.has(el)) return; seen.add(el);
    const gain = el.scrollHeight - el.clientHeight;
    if (gain <= bestGain || gain < 8) return;
    const oy = el === document.scrollingElement ? 'auto'
             : getComputedStyle(el).overflowY;
    if (oy === 'auto' || oy === 'scroll' || el === document.scrollingElement)
      { best = el; bestGain = gain; }
  };
  cands.forEach(consider);
  const all = document.querySelectorAll('div,main,section,article,ul,ol');
  for (let i = 0; i < all.length && i < 4000; i++) consider(all[i]);
  if (!best) return {moved: false, done: true, y: 0, h: 0};
  best.style.scrollBehavior = 'auto';
  const before = best.scrollTop;
  best.scrollTop = before + Math.round(best.clientHeight * 0.6);
  const after = best.scrollTop;
  return {moved: after > before + 1, done: after <= before + 1,
          y: after, h: bestGain};
}"""


def _autoscroll(page, seconds: float) -> None:
    """Top→bottom sweep over ``seconds`` (then hold at the end).

    Paced from Python (not one blocking JS call) so the recorder captures
    the motion and the duration is predictable regardless of page size.
    Only declares the page exhausted after several consecutive no-move
    steps, so a momentarily-stuck frame (lazy load, sticky header) does
    not end the scroll on the top fold.
    """
    end = time.monotonic() + max(2.0, seconds)
    step = 0.2
    stuck = 0
    while time.monotonic() < end:
        try:
            r = page.evaluate(_STEP_JS)
        except Exception:  # noqa: BLE001
            r = {"moved": False, "done": True}
        stuck = stuck + 1 if not r.get("moved") else 0
        if r.get("done") and stuck >= 5:  # ~1s of no progress → at the end
            remaining = end - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)  # hold the final view for the rest
            break
        time.sleep(step)


def _to_mp4(webm: Path, out: Path, fps: int) -> None:
    """Transcode the webm to a silent, broadly-decodable mp4."""
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(webm),
            "-an",                       # silent by design
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-movflags", "+faststart",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.is_file():
        raise WebCaptureUnavailable(
            f"ffmpeg webm→mp4 failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )


def expand_url_drops(directory: Path) -> list[Path]:
    """Worker hook: capture every ``*.url`` / ``*.webloc`` in ``directory``
    to a sibling ``<stem>.mp4`` (skipped if the mp4 is already fresh).

    Returns the mp4s produced this pass. A capture failure is recorded as
    a ``<stem>.url.error`` next to the drop and never aborts the scan, so
    one bad link can't stall the whole worker.
    """
    made: list[Path] = []
    if not directory.is_dir():
        return made
    for drop in sorted(directory.glob("*")):
        if drop.suffix.lower() not in (".url", ".webloc"):
            continue
        url = _read_url(drop)
        if not url:
            continue
        mp4 = drop.with_suffix(".mp4")
        if mp4.is_file() and mp4.stat().st_mtime >= drop.stat().st_mtime:
            continue
        try:
            capture_site(url, mp4)
            made.append(mp4)
        except Exception as exc:  # noqa: BLE001
            drop.with_suffix(drop.suffix + ".error").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
    return made


def _read_url(drop: Path) -> str | None:
    """Pull the URL out of a ``.url`` (INI ``URL=``), ``.webloc`` (plist
    ``<string>``), or a one-line text file."""
    try:
        txt = drop.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r"^\s*URL\s*=\s*(\S+)", txt, re.I | re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"<string>\s*(https?://[^<\s]+)\s*</string>", txt, re.I)
    if m:
        return m.group(1).strip()
    for line in txt.splitlines():
        line = line.strip()
        if re.match(r"^https?://", line, re.I):
            return line
    return None
