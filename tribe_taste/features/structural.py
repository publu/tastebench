"""tribe_taste.features.structural — musician-actionable craft features.

Derives the Layer-1 "song-bones" features (SONG_BENCHMARK_THESIS.md S3) from
the raw librosa report's series / chroma / beats — no re-extraction, no
model. These are the features the taste profile aggregates, `compare`
deltas, and `optimize` perturbs.

Every feature here has a matching entry in the explainer dictionary
(kind="craft"); the keys below are the explainer terms.

A feature value of ``None`` means "not computable for this input" (e.g. a
craft feature on a video) and is skipped by all downstream math.
"""

from __future__ import annotations

from typing import Optional

import math


def _finite(xs):
    return [v for v in (xs or []) if v is not None and _isnum(v)]


def _isnum(v) -> bool:
    try:
        return v == v and abs(float(v)) != float("inf")
    except (TypeError, ValueError):
        return False


def _mean(xs):
    xs = _finite(xs)
    return sum(xs) / len(xs) if xs else None


def _pct(xs, q):
    xs = sorted(_finite(xs))
    if not xs:
        return None
    k = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def _series_t(report, n):
    """Return (time_s list aligned to a length-n series) using duration."""
    dur = float(report.get("scalars", {}).get("duration_s") or 0.0)
    if dur <= 0 or n <= 0:
        return [0.0] * max(n, 0)
    return [dur * i / (n - 1) if n > 1 else 0.0 for i in range(n)]


def time_to_hook(report) -> Optional[float]:
    """Seconds until the first sustained energy peak.

    The single most-cited short-form virality lever. Computed as the first
    time the smoothed RMS crosses 85% of its global peak and stays in the
    top third for >= ~3 s.
    """
    series = report.get("series", {})
    rms = series.get("rms_db")
    if not rms:
        return None
    vals = [v if v is not None else float("-inf") for v in rms]
    finite = _finite(rms)
    if not finite:
        return None
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-6:
        return 0.0
    thr = lo + 0.85 * (hi - lo)
    n = len(vals)
    t = _series_t(report, n)
    # require the peak band to hold for ~3s worth of frames
    hold = max(1, int(round(n * 3.0 / max(report.get("scalars", {}).get(
        "duration_s") or n, 1.0))))
    run = 0
    for i, v in enumerate(vals):
        if v >= thr:
            run += 1
            if run >= hold:
                return round(t[i - run + 1], 2)
        else:
            run = 0
    # never sustained — fall back to the single global-peak time
    pk = max(range(n), key=lambda i: vals[i])
    return round(t[pk], 2)


def intro_length(report) -> Optional[float]:
    """Dead time before anything happens: seconds until RMS first exceeds
    the 35th percentile of its own distribution (energy ramp-in)."""
    series = report.get("series", {})
    rms = series.get("rms_db")
    if not rms:
        return None
    p35 = _pct(rms, 35)
    if p35 is None:
        return None
    n = len(rms)
    t = _series_t(report, n)
    for i, v in enumerate(rms):
        if v is not None and v >= p35:
            return round(t[i], 2)
    return round(t[-1], 2) if t else None


def chorus_lift_db(report) -> Optional[float]:
    """Loudest-section vs quietest-section RMS contrast in dB.

    Proxy for verse->chorus dynamic punch: p90 minus p10 of the RMS-dB
    series (production-robust within one track since both ends move with
    the gain).
    """
    rms = report.get("series", {}).get("rms_db")
    hi = _pct(rms, 90)
    lo = _pct(rms, 10)
    if hi is None or lo is None:
        return None
    return round(hi - lo, 3)


def loopability(report) -> Optional[float]:
    """Chroma self-similarity of the track's two halves [0..1].

    The literal short-form loop mechanic: how tonally consistent the song
    is across time. Mean cosine similarity between the first-half and
    second-half average chroma vectors of the normalized grid.
    """
    grid = report.get("chroma", {}).get("grid")
    if not grid or len(grid) != 12:
        return None
    ncol = len(grid[0]) if grid[0] else 0
    if ncol < 4:
        return None
    half = ncol // 2

    def _avgvec(c0, c1):
        out = []
        for p in range(12):
            col = [grid[p][c] for c in range(c0, c1) if grid[p][c] is not None]
            out.append(sum(col) / len(col) if col else 0.0)
        return out

    a = _avgvec(0, half)
    b = _avgvec(half, ncol)
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return None
    cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return round(max(0.0, min(1.0, cos)), 4)


def tempo(report) -> Optional[float]:
    v = report.get("scalars", {}).get("tempo_global")
    return round(float(v), 2) if _isnum(v) else None


def tempo_stability(report) -> Optional[float]:
    """1 / (1 + tempo_std) in [0..1] — higher = steadier pocket."""
    std = report.get("scalars", {}).get("tempo_std")
    if not _isnum(std):
        return None
    return round(1.0 / (1.0 + max(0.0, float(std))), 4)


def key_stability(report) -> Optional[float]:
    """Fraction of beats whose detected key equals the global key [0..1].

    Low value = the tonal center wanders (hurts loopability / sing-along).
    """
    beats = report.get("beats") or []
    gk = report.get("scalars", {}).get("global_key")
    if not beats or not gk:
        return None
    keyed = [b for b in beats if b.get("key")]
    if not keyed:
        return None
    same = sum(1 for b in keyed if b["key"] == gk)
    return round(same / len(keyed), 4)


def brightness(report) -> Optional[float]:
    v = report.get("scalars", {}).get("centroid_mean_hz")
    return round(float(v), 1) if _isnum(v) else None


def dynamic_range_db(report) -> Optional[float]:
    v = report.get("scalars", {}).get("dynamic_range_db")
    return round(float(v), 3) if _isnum(v) else None


def voiced_fraction(report) -> Optional[float]:
    v = report.get("scalars", {}).get("f0_voiced_frac")
    return round(float(v), 4) if _isnum(v) else None


def f0_range_octaves(report) -> Optional[float]:
    v = report.get("scalars", {}).get("f0_range_oct")
    return round(float(v), 4) if _isnum(v) else None


def hook_density_per_min(report) -> Optional[float]:
    """Distinct high-energy events per minute.

    Count of RMS upcrossings above the 80th percentile, normalized to the
    track length — a coarse "how often something grabs you" rate.
    """
    series = report.get("series", {})
    rms = series.get("rms_db")
    if not rms:
        return None
    p80 = _pct(rms, 80)
    dur = float(report.get("scalars", {}).get("duration_s") or 0.0)
    if p80 is None or dur <= 0:
        return None
    crossings = 0
    prev_above = False
    for v in rms:
        above = v is not None and v >= p80
        if above and not prev_above:
            crossings += 1
        prev_above = above
    return round(crossings / (dur / 60.0), 3)


def flatness(report) -> Optional[float]:
    v = report.get("scalars", {}).get("flatness_mean")
    return round(float(v), 6) if _isnum(v) else None


# term -> (callable, is_musician_actionable)
# "actionable" features are the ones `optimize` is allowed to perturb.
CRAFT_FEATURES = {
    "time_to_hook": (time_to_hook, True),
    "intro_length": (intro_length, True),
    "chorus_lift_db": (chorus_lift_db, True),
    "loopability": (loopability, True),
    "tempo": (tempo, True),
    "tempo_stability": (tempo_stability, True),
    "key_stability": (key_stability, True),
    "brightness": (brightness, True),
    "dynamic_range_db": (dynamic_range_db, True),
    "hook_density_per_min": (hook_density_per_min, True),
    "voiced_fraction": (voiced_fraction, False),
    "f0_range_octaves": (f0_range_octaves, False),
    "flatness": (flatness, False),
}

ACTIONABLE = [k for k, (_, act) in CRAFT_FEATURES.items() if act]


def craft_vector(report: dict) -> dict:
    """report -> {feature_name: float|None} for every craft feature.

    Returns all-None if the report carries an extractor error (non-audio,
    too short, decode failure) so callers can detect "craft unavailable".
    """
    if not report or report.get("_error"):
        return {k: None for k in CRAFT_FEATURES}
    out = {}
    for name, (fn, _act) in CRAFT_FEATURES.items():
        try:
            out[name] = fn(report)
        except Exception:
            out[name] = None
    return out
