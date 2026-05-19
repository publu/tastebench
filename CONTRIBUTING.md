# Contributing

Thanks for poking at tastebench. It's a small, opinionated tool — PRs
that keep it that way are welcome.

## Setup (model-free, seconds)

```bash
git clone https://github.com/publu/tastebench && cd tastebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                 # the model-free smoke suite (synthesizes its own audio)
```

The core install is model-free (numpy / librosa / rich — no torch,
nothing gated). You can run, test, and develop the whole craft layer,
CLI, worker, and web path with **no model download**.

## The fast path while developing

You almost never need the ~20 GB neural model to work on the tool:

```bash
tastebench compare a.wav b.wav --to demo.wav --no-brain   # instant, model-free
tastebench glossary <term>                                 # the explainer dictionary
```

`--no-brain` skips the TRIBE forward pass entirely. Touch the neural
path only when your change is *in* it; otherwise the craft layer is the
full, fast feedback loop. The optional model setup is in the README
("Set up the neural read").

## Tests

- `pytest -q` must pass. The suite is **model-free by design** —
  `tests/` synthesize their own audio and never download or call the
  neural model. Keep it that way: a test that needs the ~20 GB model is
  not a unit test.
- Adding a brain network, reward-proxy member, or craft metric key?
  `tests/test_explainers.py` enforces that every one has an entry in
  `tastebench/explainers/explainers.json`. Add the explainer with your
  metric.

## Style

Match the surrounding code: same comment density, naming, and idiom as
the file you're editing. The codebase favours small, honest functions
and plain-language docstrings that say *why*, not *what*. The product
voice is in `identity.py` — anything a user reads goes through it.

## What not to commit

No audio/video, secrets, or model weights (see `.gitignore`). The repo
ships **code only**, with the two deliberate, owned/permitted media
fixtures in `examples/` documented there. The shareable rendered
artifact is the ASCII read in `examples/README.md` — keep it text, not
an image.

## PRs

- One focused change per PR; explain the *why* in the description.
- Tests pass, model-free tests stay model-free.
- No new heavy/required dependency in the core install — heavy things
  are optional extras (`[brain]`, `[web]`, `[modal]`), exactly like the
  existing ones.
