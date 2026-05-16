"""tastebench.features.visual_report — model-free visual craft features.

The image/video analogue of `librosa_report` + `structural`: turns a still
image or a video into the visual "song-bones" — colour scheme, palette,
contrast, saturation, composition, and (video only) cut pacing + motion.
No model, no network; Pillow + numpy + the system ffmpeg already required
for audio decode.

`extract(path) -> report` mirrors the audio report shape
(`{schema_version, scalars, _error?}`). `visual_vector(report)` mirrors
`structural.craft_vector`: a flat `{feature: float|None}` whose keys each
have a `kind="craft"` explainer entry, so profile/compare/optimize/glossary
treat a video exactly like a track.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import Optional

SCHEMA_VERSION = 1
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MAX_FRAMES = 60          # cap sampled video frames
_FPS = 2                  # video sample rate
_SCALE = 256              # downscale longest side for speed


def is_visual(path) -> bool:
    return os.path.splitext(str(path))[1].lower() in (IMAGE_EXTS | VIDEO_EXTS)


# --------------------------------------------------------------------------
# per-frame measurements (numpy on an RGB uint8 HxWx3 array)
# --------------------------------------------------------------------------

def _to_hsv(rgb):
    import numpy as np
    from PIL import Image

    im = Image.fromarray(rgb, "RGB").convert("HSV")
    h = np.asarray(im, dtype=np.float32) / 255.0  # H,S,V in 0..1
    return h[..., 0], h[..., 1], h[..., 2]


def _luma(rgb):
    import numpy as np

    r, g, b = (rgb[..., i].astype(np.float32) / 255.0 for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _palette_size(rgb) -> Optional[int]:
    """Distinct dominant colours: adaptive-quantize to 16, count clusters
    holding > 5% of pixels (median-cut; no sklearn)."""
    from PIL import Image

    try:
        im = Image.fromarray(rgb, "RGB").quantize(colors=16, method=Image.MEDIANCUT)
        cols = im.getcolors() or []
        tot = sum(c for c, _ in cols) or 1
        return int(sum(1 for c, _ in cols if c / tot > 0.05))
    except Exception:
        return None


def _colorfulness(rgb) -> float:
    """Hasler-Susstrunk colourfulness metric (0 ~ greyscale, ~>60 vivid)."""
    import numpy as np

    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    mean = math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return round(std + 0.3 * mean, 3)


def _warm_cool(h, s) -> float:
    """Saturation-weighted warm vs cool balance in [-1, 1].

    warm = reds/oranges/yellows, cool = cyans/blues; near-grey pixels
    (low S) are ignored. +1 fully warm, -1 fully cool.
    """
    import numpy as np

    m = s > 0.15
    if not m.any():
        return 0.0
    hh = h[m]
    warm = ((hh < 0.14) | (hh > 0.92)).sum()
    cool = ((hh > 0.45) & (hh < 0.75)).sum()
    n = hh.size
    return round(float(warm - cool) / n, 4)


def _hue_spread(h, s) -> float:
    """Circular stdev of hue over saturated pixels, 0 (monochrome) .. 1."""
    import numpy as np

    m = s > 0.15
    if not m.any():
        return 0.0
    ang = h[m] * 2.0 * math.pi
    c, sn = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    rlen = math.sqrt(c * c + sn * sn)
    return round(math.sqrt(max(0.0, -2.0 * math.log(max(rlen, 1e-9)))) /
                 (2.0 * math.pi), 4)


def _edges(luma):
    import numpy as np

    gx = np.abs(np.diff(luma, axis=1))
    gy = np.abs(np.diff(luma, axis=0))
    m = min(gx.shape[0], gy.shape[0]), min(gx.shape[1], gy.shape[1])
    mag = gx[: m[0], : m[1]] + gy[: m[0], : m[1]]
    return mag


def _frame_scalars(rgb) -> dict:
    import numpy as np

    h, s, v = _to_hsv(rgb)
    lum = _luma(rgb)
    mag = _edges(lum)
    thr = 0.08
    H, W = mag.shape
    cy0, cy1 = H // 3, 2 * H // 3
    cx0, cx1 = W // 3, 2 * W // 3
    total_e = float(mag.sum()) or 1.0
    center_e = float(mag[cy0:cy1, cx0:cx1].sum())
    return {
        "brightness_mean": round(float(v.mean()), 4),
        "contrast": round(float(lum.std()), 4),
        "saturation_mean": round(float(s.mean()), 4),
        "saturation_std": round(float(s.std()), 4),
        "colorfulness": _colorfulness(rgb),
        "warm_cool_balance": _warm_cool(h, s),
        "hue_spread": _hue_spread(h, s),
        "palette_size": _palette_size(rgb),
        "edge_density": round(float((mag > thr).mean()), 4),
        "center_bias": round(center_e / total_e, 4),
    }


# --------------------------------------------------------------------------
# frame loading (image: 1 frame; video: ffmpeg-sampled)
# --------------------------------------------------------------------------

def _load_image(path):
    import numpy as np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((_SCALE, _SCALE))
    return [np.asarray(im)], (im.width, im.height), None


def _load_video(path, tmp):
    import numpy as np
    from PIL import Image

    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            check=True, capture_output=True, text=True).stdout.strip() or 0.0)
    except Exception:
        dur = 0.0
    pat = os.path.join(tmp, "f_%04d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vf",
         f"fps={_FPS},scale={_SCALE}:-1", "-frames:v", str(_MAX_FRAMES), pat],
        check=True, capture_output=True)
    files = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
    frames = [np.asarray(Image.open(os.path.join(tmp, f)).convert("RGB"))
              for f in files]
    if not frames:
        return [], (0, 0), dur
    h, w = frames[0].shape[:2]
    return frames, (w, h), dur


def _temporal(frames, dur) -> dict:
    """Video-only: motion energy, cut pacing, brightness drift."""
    import numpy as np

    if len(frames) < 2:
        return {k: None for k in
                ("motion_energy", "cut_rate_per_min", "avg_shot_len_s",
                 "brightness_drift")}
    lums = [_luma(f) for f in frames]
    shp = min(l.shape[0] for l in lums), min(l.shape[1] for l in lums)
    lums = [l[: shp[0], : shp[1]] for l in lums]
    diffs = [float(np.abs(lums[i] - lums[i - 1]).mean())
             for i in range(1, len(lums))]
    motion = round(float(np.mean(diffs)), 5)
    bright = [float(l.mean()) for l in lums]
    drift = round(float(np.std(bright)), 5)
    # cuts: per-frame coarse RGB histogram, L1 distance spikes
    hists = []
    for f in frames:
        hh, _ = np.histogramdd(
            f.reshape(-1, 3), bins=(4, 4, 4),
            range=((0, 255), (0, 255), (0, 255)))
        hh = hh.ravel()
        hists.append(hh / (hh.sum() or 1))
    hd = np.array([np.abs(hists[i] - hists[i - 1]).sum()
                   for i in range(1, len(hists))])
    thr = float(hd.mean() + 2.0 * hd.std())
    cuts = int((hd > max(thr, 0.35)).sum())
    secs = dur if dur > 0 else len(frames) / _FPS
    return {
        "motion_energy": motion,
        "cut_rate_per_min": round(cuts / (secs / 60.0), 3) if secs else None,
        "avg_shot_len_s": round(secs / (cuts + 1), 3) if secs else None,
        "brightness_drift": drift,
    }


def extract(path) -> dict:
    """Visual craft report for an image or video. Returns the report dict,
    or `{_error: ...}` for non-visual / unreadable input (so the brain
    layer can still carry it and the craft math degrades gracefully)."""
    import numpy as np

    path = str(path)
    if not os.path.isfile(path):
        return {"_error": f"not_a_file: {path}"}
    ext = os.path.splitext(path)[1].lower()
    if ext not in (IMAGE_EXTS | VIDEO_EXTS):
        return {"_error": f"visual_layer_image_video_only (got {ext or '<none>'})"}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            if ext in IMAGE_EXTS:
                frames, (w, h), dur = _load_image(path)
            else:
                frames, (w, h), dur = _load_video(path, tmp)
            if not frames:
                return {"_error": "no_frames_decoded"}

            per = [_frame_scalars(f) for f in frames]

            def agg(k):
                xs = [p[k] for p in per if p.get(k) is not None]
                return round(float(np.mean(xs)), 5) if xs else None

            scalars = {k: agg(k) for k in per[0]}
            scalars["aspect_ratio"] = round(w / h, 4) if h else None
            scalars["modality"] = "video" if ext in VIDEO_EXTS else "image"
            scalars["duration_s"] = round(dur, 3) if dur else None
            scalars["n_frames_sampled"] = len(frames)
            scalars.update(_temporal(frames, dur) if ext in VIDEO_EXTS
                           else {k: None for k in
                                 ("motion_energy", "cut_rate_per_min",
                                  "avg_shot_len_s", "brightness_drift")})
            return {"schema_version": SCHEMA_VERSION, "scalars": scalars}
    except FileNotFoundError as e:
        return {"_error": f"ffmpeg/ffprobe_not_found: {e!r}"}
    except subprocess.CalledProcessError as e:
        return {"_error": f"decode_failed: {str(e.stderr)[-200:]!r}"}
    except Exception as e:  # pragma: no cover - defensive
        return {"_error": f"visual_extract_failed: {e!r}"}


# --------------------------------------------------------------------------
# report -> flat named vector (mirrors structural.craft_vector)
# term -> (scalar_key, musician-actionable?)
# --------------------------------------------------------------------------

VISUAL_FEATURES = {
    "brightness_mean": ("brightness_mean", True),
    "contrast": ("contrast", True),
    "saturation_mean": ("saturation_mean", True),
    "colorfulness": ("colorfulness", True),
    "palette_size": ("palette_size", True),
    "warm_cool_balance": ("warm_cool_balance", True),
    "hue_spread": ("hue_spread", True),
    "edge_density": ("edge_density", True),
    "center_bias": ("center_bias", True),
    "saturation_std": ("saturation_std", False),
    "aspect_ratio": ("aspect_ratio", False),
    "motion_energy": ("motion_energy", True),
    "cut_rate_per_min": ("cut_rate_per_min", True),
    "avg_shot_len_s": ("avg_shot_len_s", True),
    "brightness_drift": ("brightness_drift", False),
}

ACTIONABLE = [k for k, (_, a) in VISUAL_FEATURES.items() if a]


def visual_vector(report: dict) -> dict:
    """report -> {feature: float|None}. All-None on an extractor error."""
    if not report or report.get("_error"):
        return {k: None for k in VISUAL_FEATURES}
    sc = report.get("scalars", {})
    out = {}
    for name, (key, _act) in VISUAL_FEATURES.items():
        v = sc.get(key)
        out[name] = float(v) if isinstance(v, (int, float)) else None
    return out
