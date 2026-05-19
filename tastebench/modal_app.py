"""tastebench.modal_app — run the heavy brain layer on Modal.

The remote door for the ~20 GB TRIBE brain layer. The local path is
unchanged: `make` / `tastebench` / the CLI run as before. This is just
a GPU alternative for people without one.

Self-serve, account-scoped:
* `modal setup` uses the runner's own Modal workspace.
* The ~20 GB model cache lives in a Modal **Volume** in that workspace,
  so it downloads once and persists between runs.
* Gated Llama-3.2 reads a HF token from a Modal **Secret** named
  ``huggingface`` that the runner creates; this file references it by
  name only.

Usage (from a clone, after `pip install -e ".[modal]"`)
-------------------------------------------------------
    modal setup                                   # once: their account
    modal run tastebench/modal_app.py::download   # once: warm the Volume
    modal secret create huggingface HF_TOKEN=hf_… # once: gated Llama-3.2
    modal run tastebench/modal_app.py \
        --demo demo.wav --refs ref_a.wav,ref_b.wav

The brain layer re-runs on every file each call (no remote profile cache —
the local worker is the place with caching; this is the one-shot door).

GPU is an env knob, same spirit as the other TRIBE_* knobs:
    TASTEBENCH_MODAL_GPU=A100   modal run …       # if A10G OOMs
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# Account-scoped resources: a persistent Volume so the ~20 GB cache
# downloads once, and a Secret for the gated Llama-3.2 weights.
VOLUME = modal.Volume.from_name("tastebench-models", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")  # see module docstring

# CUDA box → the engine's own code keeps full upstream fidelity (64
# video frames, full res, transcription). A10G (24 GB) is the economical
# default; bump to A100 via the env knob if a big video OOMs the GPU.
GPU = os.environ.get("TASTEBENCH_MODAL_GPU", "A10G")

_REPO = Path(__file__).resolve().parent.parent

# DRY: the brain dep set has ONE source of truth — pyproject's [brain]
# extra. We copy the package source into the image and `pip install
# .[brain]` so the image can never drift from `make brain`.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .add_local_file(_REPO / "pyproject.toml", "/src/pyproject.toml", copy=True)
    .add_local_file(_REPO / "README.md", "/src/README.md", copy=True)
    .add_local_dir(
        _REPO / "tastebench",
        "/src/tastebench",
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .run_commands("pip install --no-cache-dir '/src[brain]'")
    # Pin every cache onto the persistent Volume mounted at /cache so the
    # ~20 GB survives between runs. engine._model_cache_dir() honours
    # MODEL_CACHE explicitly; HF/torch honour their own vars.
    .env(
        {
            "MODEL_CACHE": "/cache/model-cache",
            "HF_HOME": "/cache/hf",
            "TORCH_HOME": "/cache/torch",
            # Cold Volume must be allowed to self-populate; the engine's
            # offline pins are a speed optimisation for a warm local box.
            "TRIBE_ALLOW_NET": "1",
        }
    )
)

app = modal.App("tastebench")


@app.function(
    image=image,
    volumes={"/cache": VOLUME},
    secrets=[HF_SECRET],
    gpu=GPU,
    timeout=2 * 60 * 60,  # the ~20 GB warm fetch can be slow
)
def download() -> str:
    """Warm the Volume once: fetch the ~20 GB TRIBE cache into the
    cloner's own Volume. Mirrors scripts/download_models.py, but the
    cache lands on the persistent Volume instead of ~/.cache.
    """
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from tastebench import native

    native.apply_runtime_env()
    device = native.resolve_device()  # "cuda" on a Modal GPU box

    cache = Path(os.environ["MODEL_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)

    from tribev2.demo_utils import TribeModel

    TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(cache),
        config_update={"data.text_feature.model_name": "unsloth/Llama-3.2-3B"},
        device=device,
    )
    if hasattr(VOLUME, "commit"):
        VOLUME.commit()  # persist the 20 GB so future runs skip the fetch
    return f"warmed {cache} on device={device}"


@app.function(
    image=image,
    volumes={"/cache": VOLUME},
    secrets=[HF_SECRET],
    gpu=GPU,
    timeout=60 * 60,
)
def grade(
    demo: tuple[str, bytes],
    refs: list[tuple[str, bytes]],
    use_brain: bool = True,
    fmt: str = "markdown",
    llm: bool = False,
    deep: bool = False,
) -> dict:
    """Run the existing compare() on a Modal GPU and render it there.

    Rendering happens in-image (tastebench.report is available here) so
    the local clone needs nothing but the `modal` SDK to print the
    verdict — no local core install, no local model.
    """
    import tempfile

    from tastebench import report as _report
    from tastebench.compare import compare

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        dpath = tdp / demo[0]
        dpath.write_bytes(demo[1])
        rpaths = []
        for name, blob in refs:
            rp = tdp / name
            rp.write_bytes(blob)
            rpaths.append(str(rp))

        payload = compare(str(dpath), rpaths, use_brain=use_brain)
        payload["_kind"] = "compare"

    brief = not deep and not llm and fmt == "markdown"
    text = (
        _report.to_verdict(payload)
        if brief
        else _report.render(payload, fmt=fmt, llm=llm)
    )
    return {"text": text, "payload": payload}


def _read(p: str) -> tuple[str, bytes]:
    fp = Path(p).expanduser()
    if not fp.is_file():
        raise SystemExit(f"not a file: {p}")
    return fp.name, fp.read_bytes()


@app.local_entrypoint()
def main(
    demo: str,
    refs: str,
    no_brain: bool = False,
    deep: bool = False,
    llm: bool = False,
    fmt: str = "markdown",
    out: str = "",
) -> None:
    """`modal run tastebench/modal_app.py --demo X --refs a,b,c`.

    --demo  your draft (audio / video / image)
    --refs  comma-separated reference files you admire
    --deep  full report instead of the one-screen verdict
    --llm   emit the LLM-ready bundle
    --out   write the rendered text to a file too
    """
    ref_list = [r for r in (s.strip() for s in refs.split(",")) if r]
    if not ref_list:
        raise SystemExit("--refs needs at least one file (comma-separated)")

    res = grade.remote(
        _read(demo),
        [_read(r) for r in ref_list],
        use_brain=not no_brain,
        fmt=fmt,
        llm=llm,
        deep=deep,
    )
    print(res["text"])
    if out:
        Path(out).expanduser().write_text(
            res["text"] + ("" if res["text"].endswith("\n") else "\n"),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
