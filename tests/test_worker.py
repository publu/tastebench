"""Worker smoke tests — the folder worker must lay out its tree, settle
files (no half-copied reads), and grade a draft against a taste's refs
end to end with NO TRIBE model present (craft layer only).

Synthesizes its own audio; never touches the network or a model cache.
"""

import io
import os
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


def _console():
    from rich.console import Console

    return Console(file=io.StringIO(), force_terminal=False)


def test_ensure_skeleton(tmp_path):
    from tastebench.worker import _ensure_skeleton

    _ensure_skeleton(tmp_path)
    base = tmp_path / "references" / "example"
    assert (base / "refs").is_dir()
    assert (base / "draft").is_dir()
    assert (tmp_path / "references" / "HOW-THIS-WORKS.txt").is_file()

    # idempotent + does not clobber a real, user-made taste
    (tmp_path / "references" / "mine").mkdir()
    _ensure_skeleton(tmp_path)
    assert (tmp_path / "references" / "mine").is_dir()


def test_scan_grades_draft_against_refs(tmp_path):
    from tastebench.worker import scan_once

    taste = tmp_path / "references" / "trap"
    (taste / "refs").mkdir(parents=True)
    (taste / "draft").mkdir(parents=True)
    _tone(str(taste / "refs" / "ref_a.wav"), [261.6, 329.6, 392.0], beat=0.22)
    _tone(str(taste / "refs" / "ref_b.wav"), [261.6, 329.6, 392.0], beat=0.24)
    _tone(str(taste / "draft" / "mine.wav"), [293.7, 349.2, 440.0],
          beat=0.40, reps=5)

    console = _console()
    state, ready = {}, {}

    # Pass 1: files are seen for the first time → not yet "stable", so the
    # settle guard must do nothing (no half-copied grade).
    assert scan_once(tmp_path, state, False, console, ready) == []

    # Pass 2: stamps held steady across a poll → profile + grade now run.
    did = scan_once(tmp_path, state, False, console, ready)
    assert ("trap", "mine.wav", str(taste / "mine.report.md")) in did

    report = taste / "mine.report.md"
    assert report.is_file()
    body = report.read_text()
    assert "mine.wav vs your taste" in body
    assert "taste match" in body

    # Pass 3: nothing changed → cached, not re-graded.
    assert scan_once(tmp_path, state, False, console, ready) == []


def test_unprofiled_taste_waits(tmp_path):
    from tastebench.worker import scan_once

    (tmp_path / "references" / "empty" / "refs").mkdir(parents=True)
    console = _console()
    state, ready = {}, {}
    # No refs anywhere → nothing graded, no crash; waits quietly.
    assert scan_once(tmp_path, state, False, console, ready) == []
    assert scan_once(tmp_path, state, False, console, ready) == []


def test_refuses_package_source_as_workspace(tmp_path):
    """Running the worker against the package dir itself must bail, not
    write the watched tree into the source."""
    from tastebench.worker import run

    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "worker.py").write_text("")
    assert run(tmp_path, use_brain=False) == 2
    assert not (tmp_path / "references").exists()


def test_cli_help_lists_worker():
    import subprocess

    cp = subprocess.run(
        [sys.executable, "-m", "tastebench.cli", "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert cp.returncode == 0
    assert "worker" in cp.stdout
    assert "drop" in cp.stdout
