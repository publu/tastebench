"""Smoke tests — the model-free craft path must work end to end with NO
TRIBE model present, and the engine must degrade with a clear message.

These tests synthesize their own audio (no third-party media).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SR = 22050


def _tone(path, freqs, beat=0.25, reps=8):
    seg = []
    for _ in range(reps):
        for f in freqs:
            t = np.linspace(0, beat, int(SR * beat), endpoint=False)
            w = 0.5 * np.sin(2 * np.pi * f * t)
            env = np.minimum(1.0, np.minimum(t * 8, (beat - t) * 8))
            seg.append((w * env).astype(np.float32))
    audio = np.concatenate(seg)
    sf.write(path, audio / (np.max(np.abs(audio)) or 1) * 0.9, SR)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    ref_a = str(d / "ref_a.wav")
    ref_b = str(d / "ref_b.wav")
    demo = str(d / "demo.wav")
    _tone(ref_a, [261.6, 329.6, 392.0], beat=0.22)
    _tone(ref_b, [261.6, 329.6, 392.0], beat=0.24)
    _tone(demo, [293.7, 349.2, 440.0], beat=0.40, reps=5)
    return ref_a, ref_b, demo


def test_package_imports_without_model():
    import tastebench  # noqa: F401
    from tastebench import (  # noqa: F401
        brain,
        compare,
        engine,
        optimize,
        profile,
        report,
        signature,
    )
    from tastebench.features import librosa_report, structural  # noqa: F401


def test_engine_degrades_clearly(clips):
    from tastebench import engine

    assert engine.models_available() in (True, False)
    if not engine.models_available():
        # get_model() is the load path predict() uses for any valid media;
        # with no cache it must raise the clear, actionable message.
        with pytest.raises(engine.ModelsNotDownloaded) as ei:
            engine.get_model()
        msg = str(ei.value).lower()
        assert "model not available" in msg
        assert "download_models.py" in msg
        # predict() on real audio surfaces the same clear error
        with pytest.raises(engine.ModelsNotDownloaded):
            engine.predict(clips[0])


def test_craft_report(clips):
    from tastebench.features import librosa_report
    from tastebench.features.structural import craft_vector

    rep = librosa_report.extract(clips[0])
    assert "_error" not in rep
    assert rep["scalars"]["duration_s"] > 5
    vec = craft_vector(rep)
    assert vec["tempo"] is not None
    assert vec["loopability"] is not None


def test_profile_and_compare(clips):
    from tastebench.compare import compare
    from tastebench.profile import build_profile

    ref_a, ref_b, demo = clips
    prof = build_profile([ref_a, ref_b], use_brain=False)
    assert prof["n_refs"] == 2
    assert prof["consistency"] is not None
    res = compare(demo, prof, use_brain=False)
    # distance must be finite and sane (regression: was 1e7 with bare spread)
    assert res["overall_distance"] is not None
    assert 0 <= res["overall_distance"] < 1e4
    assert res["nearest_reference"]["name"] in ("ref_a.wav", "ref_b.wav")
    assert all("explainer" in r for r in res["craft_deltas"])


def test_optimize_ranked_edits(clips):
    from tastebench.optimize import optimize

    ref_a, ref_b, demo = clips
    res = optimize(demo, [ref_a, ref_b], use_brain=False, top=5)
    assert len(res["edits"]) <= 5
    for e in res["edits"]:
        assert e["predicted_gain"] > 0
        assert e["confidence"] in ("high", "medium", "low")
        assert e["explainer"] is not None
    # ranked descending by predicted gain
    gains = [e["predicted_gain"] for e in res["edits"]]
    assert gains == sorted(gains, reverse=True)


def test_explainers_complete():
    from tastebench import explainers
    from tastebench.brain import NETWORKS
    from tastebench.features.structural import CRAFT_FEATURES

    e = explainers.entries()
    for feat in CRAFT_FEATURES:
        assert feat in e, f"missing craft explainer: {feat}"
    for net in NETWORKS:
        assert net in e, f"missing network explainer: {net}"
    assert len(explainers.by_kind("edit")) >= 6


def test_cli_help_and_glossary():
    cp = subprocess.run(
        [sys.executable, "-m", "tastebench.cli", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert cp.returncode == 0
    assert "profile" in cp.stdout and "optimize" in cp.stdout

    cp = subprocess.run(
        [sys.executable, "-m", "tastebench.cli", "glossary"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert cp.returncode == 0
    assert "CRAFT FEATURES" in cp.stdout
    assert "BRAIN NETWORKS" in cp.stdout
