"""tribe_taste.metric — the shared taste-distance metric.

A feature delta (demo - taste centroid) is normalized by a *robust* scale,
not the bare reference spread. Using only the spread blows up to infinity
when a reference set is perfectly consistent on a feature (spread = 0) —
a common, valid case (e.g. two refs at the same tempo). The robust scale
combines three terms and takes the largest:

    scale = max( spread,                 # the taste's own tolerance
                 REL_FRAC * |centroid|,  # a relative floor (% of the value)
                 ABS_FLOOR[feature] )    # an absolute floor (sane units)

So a zero-spread feature still yields a bounded, musically sensible
deviation: e.g. a 30% tempo change reads as a few units, not millions.
Both `compare` and `optimize` use this so their numbers agree.
"""

from __future__ import annotations

# relative floor: a deviation of this fraction of the centroid magnitude
# counts as ~1 normalized unit when the reference set is degenerate.
REL_FRAC = 0.15

# per-feature absolute floors (units of the feature). Used when both the
# spread and the relative floor are tiny (e.g. centroid ~0).
ABS_FLOOR = {
    "time_to_hook": 1.5,          # seconds
    "intro_length": 1.0,          # seconds
    "chorus_lift_db": 1.0,        # dB
    "loopability": 0.05,          # ratio
    "key_stability": 0.05,        # ratio
    "tempo": 3.0,                 # BPM
    "tempo_stability": 0.05,      # ratio
    "brightness": 80.0,           # Hz
    "dynamic_range_db": 1.0,      # dB
    "hook_density_per_min": 0.5,
    "voiced_fraction": 0.05,
    "f0_range_octaves": 0.05,
    "flatness": 0.01,
}

# brain net.* signals are z-scored already (~unit scale); a small floor.
_BRAIN_FLOOR = 0.25


def scale_for(feature: str, spread: float, centroid: float) -> float:
    """Robust normalizing scale for one feature."""
    spread = abs(spread or 0.0)
    rel = REL_FRAC * abs(centroid or 0.0)
    if feature.startswith("net."):
        floor = _BRAIN_FLOOR
    else:
        floor = ABS_FLOOR.get(feature, max(1e-6, rel if rel > 0 else 1.0))
    return max(spread, rel, floor)


def zdelta(feature: str, delta: float, spread: float, centroid: float) -> float:
    """Robust spread-normalized delta for one feature."""
    return delta / scale_for(feature, spread, centroid)
