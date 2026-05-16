#!/usr/bin/env python3
"""Synthesize lawful, public-domain example clips (no copyrighted audio).

These are pure synthetic tones/structure generated here at runtime, so they
carry no third-party rights. Audio is .gitignored by design (the repo never
ships media); run this to (re)create local examples for the quickstart.

  python examples/make_examples.py

Creates (audio — music craft):
  examples/ref_a.wav      a "reference" — fast hook, tight loop, bright
  examples/ref_b.wav      another reference in the same taste
  examples/demo.wav       a "demo" that diverges (slow hook, key wander, dark)
Creates (images — visual craft):
  examples/ref_a.png      a warm-palette "reference"
  examples/ref_b.png      another warm reference in the same taste
  examples/demo.png       a cool-palette "demo" that diverges

Then try:
  tribe-taste profile examples/ref_a.wav examples/ref_b.wav --no-brain
  tribe-taste compare examples/ref_a.wav examples/ref_b.wav --to examples/demo.wav --no-brain
  tribe-taste optimize examples/demo.wav --toward examples/ref_a.wav examples/ref_b.wav
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SR = 22050


def _note(freq, dur, sr=SR, amp=0.4):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # a few harmonics for a non-trivial spectrum
    w = (
        amp * np.sin(2 * np.pi * freq * t)
        + 0.25 * amp * np.sin(2 * np.pi * 2 * freq * t)
        + 0.12 * amp * np.sin(2 * np.pi * 3 * freq * t)
    )
    env = np.minimum(1.0, np.minimum(t * 8, (dur - t) * 8))
    return (w * env).astype(np.float32)


def _seq(freqs, beat=0.25, **kw):
    return np.concatenate([_note(f, beat, **kw) for f in freqs])


def _silence(dur):
    return np.zeros(int(SR * dur), dtype=np.float32)


# C-major scale frequencies
C, D, E, F, G, A, B, C2 = 261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3


def reference(bright=True, intro=0.5):
    """Fast hook, consistent key, bright, loopable."""
    hook = _seq([C, E, G, C2, G, E], beat=0.22, amp=0.55)
    verse = _seq([C, E, G, E], beat=0.30, amp=0.32)
    track = np.concatenate([
        _silence(intro),
        hook, hook, verse, hook, hook, verse, hook, hook,
    ])
    if bright:
        # add airy high partials
        t = np.arange(len(track)) / SR
        track = track + 0.06 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
    return track


def demo():
    """Slow to start, tonal wander, darker, less loopable."""
    intro = _seq([C, C, C], beat=0.5, amp=0.12)  # long quiet intro
    a = _seq([C, E, G, E], beat=0.32, amp=0.35)
    b = _seq([D, F, A, F], beat=0.32, amp=0.35)   # different tonal center
    c = _seq([E, G, B, G], beat=0.32, amp=0.35)
    track = np.concatenate([
        _silence(2.0), intro, a, b, c, a, b, c, a,
    ])
    # darker: low-pass-ish (attenuate by simple moving average)
    k = 12
    track = np.convolve(track, np.ones(k) / k, mode="same").astype(np.float32)
    return track


def _gradient(c1, c2):
    """A simple 360x360 left->right colour gradient (synthetic, no rights)."""
    a = np.zeros((360, 360, 3), np.uint8)
    for x in range(360):
        t = x / 359.0
        a[:, x] = [int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)]
    return a


def make_visual(here: str) -> None:
    """Synthetic image examples: two warm 'references', one cool 'demo'."""
    from PIL import Image

    imgs = {
        "ref_a.png": _gradient((200, 90, 30), (230, 180, 60)),   # warm
        "ref_b.png": _gradient((210, 100, 40), (240, 170, 70)),  # warm
        "demo.png": _gradient((20, 60, 150), (40, 140, 170)),    # cool
    }
    for name, arr in imgs.items():
        Image.fromarray(arr, "RGB").save(os.path.join(here, name))
        print(f"wrote examples/{name}  (360x360)")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {
        "ref_a.wav": reference(bright=True, intro=0.4),
        "ref_b.wav": reference(bright=True, intro=0.6),
        "demo.wav": demo(),
    }
    for name, audio in out.items():
        peak = float(np.max(np.abs(audio))) or 1.0
        sf.write(os.path.join(here, name), audio / peak * 0.9, SR)
        dur = len(audio) / SR
        print(f"wrote examples/{name}  ({dur:.1f}s)")
    make_visual(here)
    print("\nNow try:")
    print("  tribe-taste compare examples/ref_a.wav examples/ref_b.wav "
          "--to examples/demo.wav --no-brain")
    print("  tribe-taste vibe examples/demo.png "
          "--like examples/ref_a.png examples/ref_b.png   # visual craft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
