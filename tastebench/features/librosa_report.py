#!/usr/bin/env python3
"""
librosa_report.py -- the craft-feature extractor.

This is the audio-only "craft" layer of tastebench: a deterministic
librosa extractor that turns one audio file into a structured report of
musically meaningful scalars + time series + a chroma grid + a per-beat
key/strength path. It needs NO model and is the graceful-degradation path
when the TRIBE brain model is not installed.

Schema (one report per track):
    scalars : duration_s tempo_global tempo_mean tempo_std n_beats
              n_frames global_key global_key_strength is_minor
              f0_median_hz f0_voiced_frac f0_p10_hz f0_p90_hz
              f0_range_oct centroid_mean_hz flatness_mean
              dynamic_range_db tempo_band spectrum_label
    series  : time_s rms_db spectral_centroid_hz spectral_rolloff_hz
              spectral_flatness f0_hz voiced_prob  (each 360 pts;
              tempo_bpm 180 pts) -- None where unvoiced/invalid
    chroma  : time_s(240) grid(12x240, per-col max-normalized,
              None for empty cols) pitch_classes(12)
    beats   : [{t, key, strength}, ...]

This schema is held stable on purpose so reference and demo reports are
directly comparable.

Use as a library:
    from tastebench.features.librosa_report import compute_report, extract
    report = extract("song.mp3")      # decodes to wav if needed, returns dict

Use as a CLI (kept for the historical isolated-subprocess contract -- a
wedged librosa/numba JIT can be SIGKILLed by a parent without taking the
caller down):
    python -m tastebench.features.librosa_report path/to/decoded.wav
    -> single JSON object on stdout, or {"_error": "..."} on failure.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

# Pin BLAS/numba thread pools BEFORE numpy/librosa import (matches the
# corpus extractor's env; avoids thread-explosion contention on the
# M-series box while TRIBE also runs).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

KEY_NAMES_MAJ = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES_MIN = [k + "m" for k in KEY_NAMES_MAJ]

# Krumhansl-Kessler major/minor profiles (same constants the corpus
# extractor + compute_audio_features.py / extract_drake_librosa.py use).
_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

SR = 22050
HOP = 512


def _krumhansl_key(chroma_avg):
    """chroma_avg (12,) -> (key_name, mode_is_minor, strength).

    Matches extract_drake_librosa.krumhansl_key: np.corrcoef of the
    rolled K-K profile against the average chroma, pick best of 24, but
    the corpus 'strength' scale (~20-30, not a [-1,1] corr) is the
    raw dot of the (unnormalized) profile with the chroma energy sum —
    reproduced here so global_key_strength / beats[].strength land in
    the corpus range. Key NAME selection (the load-bearing part for the
    grader's Krumhansl bins) is the corrcoef argmax, identical to corpus.
    """
    import numpy as np

    maj = np.asarray(_MAJ, dtype=float)
    minp = np.asarray(_MIN, dtype=float)
    ca = np.asarray(chroma_avg, dtype=float)
    if not np.any(np.isfinite(ca)) or float(ca.sum()) == 0.0:
        return None, False, 0.0
    maj_c = [np.corrcoef(np.roll(maj, i), ca)[0, 1] for i in range(12)]
    min_c = [np.corrcoef(np.roll(minp, i), ca)[0, 1] for i in range(12)]
    bi_maj = int(np.nanargmax(maj_c))
    bi_min = int(np.nanargmax(min_c))
    s_maj = maj_c[bi_maj]
    s_min = min_c[bi_min]
    if (s_maj if s_maj == s_maj else -9) >= (s_min if s_min == s_min else -9):
        idx, name, is_minor = bi_maj, KEY_NAMES_MAJ[bi_maj], False
        prof = np.roll(maj, idx)
    else:
        idx, name, is_minor = bi_min, KEY_NAMES_MIN[bi_min], True
        prof = np.roll(minp, idx)
    # corpus-scale strength: dot of the (unnormalized) K-K profile with
    # the raw mean chroma_cqt energy. The corpus 'strength' (~20-30 for
    # global, 0-40 per-beat) is NOT a [-1,1] corrcoef; this raw dot lands
    # in that band on real polyphonic audio (more pitch classes carry
    # energy than in a pure tone). NB the grader only ever consumes this
    # as a rounded pass-through scalar — never a threshold/comparison —
    # so exact magnitude is non-load-bearing; the KEY NAME (the part the
    # Krumhansl bins + per-beat transition analysis depend on) is the
    # corrcoef-argmax above, byte-identical to the corpus extractor.
    strength = float(np.dot(prof, ca))
    return name, is_minor, strength


def _resample_to(arr, n):
    """Linear-resample a 1-D float array to n points (np.interp), keeping
    NaN where the source frame is NaN — the corpus encodes those as JSON
    null. Matches the corpus series length contract (360 / 180 / 240)."""
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return [None] * n
    if a.size == 1:
        v = a[0]
        out = [None if not np.isfinite(v) else float(v)] * n
        return out
    xs_src = np.linspace(0.0, 1.0, num=a.size)
    xs_dst = np.linspace(0.0, 1.0, num=n)
    # Interpolate over the finite support only, then re-mask any
    # destination point whose nearest SOURCE frame was NaN — this keeps
    # unvoiced/invalid gaps honest (corpus encodes them as JSON null)
    # instead of smearing a fake value across them.
    finite = np.isfinite(a)
    if not finite.any():
        return [None] * n
    src_idx = np.clip(np.round(xs_dst * (a.size - 1)).astype(int), 0, a.size - 1)
    base = np.interp(xs_dst, xs_src[finite], a[finite])
    out = []
    for k in range(n):
        if not finite[src_idx[k]]:
            out.append(None)
        else:
            val = float(base[k])
            out.append(val if np.isfinite(val) else None)
    return out


def _round_series(xs, nd):
    return [None if v is None else round(float(v), nd) for v in xs]


def _band(tempo_global):
    """tempo_band from tempo_global. Boundaries reverse-engineered from
    the corpus (tempo_global is the decider; clean non-overlapping
    ranges: slow 81-99, mid 103-117, uptempo_deadzone 123-136, fast
    144+). slow<100, mid 100-120, uptempo_deadzone 120-140, fast>=140."""
    t = float(tempo_global or 0.0)
    if t <= 0:
        return None
    if t < 100.0:
        return "slow"
    if t < 120.0:
        return "mid"
    if t < 140.0:
        return "uptempo_deadzone"
    return "fast"


def _spectrum_label(centroid_mean_hz):
    """spectrum_label from centroid_mean_hz. Corpus cuts: dark<1800,
    neutral 1800-2400, bright>=2400 (verified against 366 reports:
    dark 1103-1784, neutral 1810-2399, bright 2400-3375)."""
    c = float(centroid_mean_hz or 0.0)
    if c <= 0:
        return None
    if c < 1800.0:
        return "dark"
    if c < 2400.0:
        return "neutral"
    return "bright"


def compute_report(wav_path: str) -> dict:
    import librosa
    import numpy as np

    # Full-rate, full-duration load (NOT 16k; NOT capped) — corpus tempo/
    # chroma/n_frames need full resolution. The corpus n_frames/duration
    # ratio is exactly sr/hop (22050/512≈43.07), confirming no time cap.
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    if y is None or len(y) == 0:
        return {"_error": "empty audio"}
    dur = float(len(y) / sr)
    if dur < 5.0:
        return {"_error": f"too_short ({dur:.1f}s)"}

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    # --- global tempo + beats ---
    tg, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP
    )
    tempo_global = float(tg[0]) if hasattr(tg, "__len__") and len(tg) else float(tg or 0.0)
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=HOP
    ).tolist() if len(beat_frames) else []

    # per-frame tempo (tempogram) -> tempo_mean / tempo_std
    try:
        dtempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=HOP, aggregate=None
        )
        dtempo = np.asarray(dtempo, dtype=float)
        dtempo = dtempo[np.isfinite(dtempo)]
        if dtempo.size:
            tempo_mean = float(dtempo.mean())
            tempo_std = float(dtempo.std())
        else:
            tempo_mean, tempo_std = tempo_global, 0.0
    except Exception:
        dtempo = np.asarray([tempo_global], dtype=float)
        tempo_mean, tempo_std = tempo_global, 0.0

    # --- spectral series (full-res, hop 512) ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    n_frames = int(len(rms))
    times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=HOP
    )

    centroid_mean_hz = float(np.mean(centroid)) if centroid.size else 0.0
    flatness_mean = float(np.mean(flat)) if flat.size else 0.0
    # corpus dynamic_range_db: p95-p5 of FULL-res rms_db, sentinel-capped
    # at 100.0 (the grader treats >=99.999 as a saturation flag).
    if rms_db.size:
        dr = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    else:
        dr = 0.0
    dynamic_range_db = min(dr, 100.0)

    # --- f0 (vocal pitch) via pyin (corpus uses fmin=80 fmax=800) ---
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=80, fmax=800, sr=sr, hop_length=HOP
        )
    except Exception:
        f0 = np.full(n_frames, np.nan)
        voiced_prob = np.zeros(n_frames)
    f0 = np.asarray(f0, dtype=float)
    voiced_prob = np.asarray(voiced_prob, dtype=float)
    f0v = f0[np.isfinite(f0)]
    if f0v.size > 5:
        f0_median = float(np.median(f0v))
        f0_p10 = float(np.percentile(f0v, 10))
        f0_p90 = float(np.percentile(f0v, 90))
        f0_range_oct = (
            float(np.log2(max(f0_p90, 1.0) / max(f0_p10, 1.0)))
            if f0_p10 > 0 else 0.0
        )
        f0_voiced_frac = float(np.isfinite(f0).mean())
    else:
        f0_median = f0_p10 = f0_p90 = f0_range_oct = f0_voiced_frac = 0.0

    # --- chroma (cqt, like corpus) + per-frame max-normalized grid ---
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    except Exception:
        # numba/cqt edge: fall back to stft chroma with tuning pinned
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=HOP, tuning=0.0
        )
    chroma = np.asarray(chroma, dtype=float)  # (12, F)
    chroma_avg = chroma.mean(axis=1) if chroma.shape[1] else np.zeros(12)
    global_key, is_minor, gk_strength = _krumhansl_key(chroma_avg)

    # per-column (per-frame) max-normalized chroma -> 12x240, None cols
    F = chroma.shape[1]
    if F:
        colmax = chroma.max(axis=0)
        norm = np.zeros_like(chroma)
        nz = colmax > 1e-9
        norm[:, nz] = chroma[:, nz] / colmax[nz]
        cols_idx = (
            np.linspace(0, F - 1, num=240).round().astype(int)
            if F > 1 else np.zeros(240, dtype=int)
        )
        grid = []
        for p in range(12):
            row = []
            for ci in cols_idx:
                if colmax[ci] <= 1e-9:
                    row.append(None)
                else:
                    row.append(round(float(norm[p, ci]), 4))
            grid.append(row)
        chroma_times = librosa.frames_to_time(
            cols_idx, sr=sr, hop_length=HOP
        ).tolist()
        chroma_time_s = [round(float(t), 4) for t in chroma_times]
    else:
        grid = [[None] * 240 for _ in range(12)]
        chroma_time_s = [0.0] * 240

    # --- per-beat key + strength (chroma window around each beat) ---
    beats = []
    if beat_frames is not None and len(beat_frames) and F:
        bf = np.asarray(beat_frames, dtype=int)
        for bi in range(len(bf)):
            lo = bf[bi]
            hi = bf[bi + 1] if bi + 1 < len(bf) else F
            lo = max(0, min(lo, F - 1))
            hi = max(lo + 1, min(hi, F))
            seg = chroma[:, lo:hi]
            cavg = seg.mean(axis=1) if seg.shape[1] else chroma_avg
            kname, _kmin, kstr = _krumhansl_key(cavg)
            t = (
                float(beat_times[bi])
                if bi < len(beat_times)
                else float(librosa.frames_to_time(bf[bi], sr=sr, hop_length=HOP))
            )
            beats.append({
                "t": round(t, 4),
                "key": kname,
                "strength": round(float(kstr), 4),
            })
    n_beats = len(beats)

    report = {
        "schema_version": 1,
        "scalars": {
            "duration_s": dur,
            "tempo_global": tempo_global,
            "tempo_mean": tempo_mean,
            "tempo_std": tempo_std,
            "n_beats": int(n_beats),
            "n_frames": int(n_frames),
            "global_key": global_key,
            "global_key_strength": gk_strength,
            "is_minor": bool(is_minor),
            "f0_median_hz": f0_median,
            "f0_voiced_frac": f0_voiced_frac,
            "f0_p10_hz": f0_p10,
            "f0_p90_hz": f0_p90,
            "f0_range_oct": f0_range_oct,
            "centroid_mean_hz": centroid_mean_hz,
            "flatness_mean": flatness_mean,
            "dynamic_range_db": dynamic_range_db,
            "tempo_band": _band(tempo_global),
            "spectrum_label": _spectrum_label(centroid_mean_hz),
        },
        "series": {
            "time_s": _round_series(_resample_to(times, 360), 4),
            "rms_db": _round_series(_resample_to(rms_db, 360), 4),
            "spectral_centroid_hz": _round_series(
                _resample_to(centroid, 360), 3
            ),
            "spectral_rolloff_hz": _round_series(
                _resample_to(rolloff, 360), 3
            ),
            "spectral_flatness": _round_series(_resample_to(flat, 360), 6),
            "f0_hz": _round_series(_resample_to(f0, 360), 4),
            "voiced_prob": _round_series(_resample_to(voiced_prob, 360), 4),
            "tempo_bpm": _round_series(_resample_to(dtempo, 180), 4),
        },
        "chroma": {
            "time_s": chroma_time_s,
            "grid": grid,
            "pitch_classes": list(KEY_NAMES_MAJ),
        },
        "beats": beats,
    }
    return report


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}


def is_audio(path) -> bool:
    """True if the path looks like an audio file the craft layer handles."""
    import os

    return os.path.splitext(str(path))[1].lower() in _AUDIO_EXTS


def extract(path) -> dict:
    """Library entry point: compute the craft report for a media file.

    Accepts any audio container (decodes non-wav to a temp 22.05 kHz wav
    via ffmpeg, matching the extractor's expected rate). Returns the report
    dict on success, or ``{"_error": ...}`` for non-audio / unreadable
    input -- so the brain layer can carry video/image while craft no-ops
    gracefully. Never raises for an unsupported type.
    """
    import os
    import subprocess
    import tempfile

    path = str(path)
    if not os.path.isfile(path):
        return {"_error": f"not_a_file: {path}"}
    ext = os.path.splitext(path)[1].lower()
    if ext not in _AUDIO_EXTS:
        return {"_error": f"craft_layer_audio_only (got {ext or '<none>'})"}

    if ext == ".wav":
        return compute_report(path)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", str(SR), "-ac", "1", wav],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            return {"_error": "ffmpeg_not_found (needed to decode non-wav audio)"}
        except subprocess.CalledProcessError as e:
            return {"_error": f"ffmpeg_decode_failed: {e.stderr[-300:]!r}"}
        return compute_report(wav)


def series_seconds(report: dict) -> float:
    """Best-effort track duration in seconds from a report's scalars."""
    try:
        return float(report.get("scalars", {}).get("duration_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    if len(sys.argv) != 2:
        sys.stdout.write(json.dumps({"_error": "usage: librosa_report.py <wav>"}))
        return 2
    try:
        rep = compute_report(sys.argv[1])
    except Exception as e:  # noqa: BLE001 — surface to parent, never crash silently
        import traceback
        sys.stdout.write(json.dumps({
            "_error": f"{type(e).__name__}: {e}",
            "_tb": traceback.format_exc()[-1500:],
        }))
        return 1
    sys.stdout.write(json.dumps(rep, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
