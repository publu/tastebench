<!-- One focused change per PR. Explain the WHY. -->

**What & why**
What this changes and the reason.

**Checklist**
- [ ] `pytest -q` passes
- [ ] Model-free tests stayed model-free (no model download/call added to `tests/`)
- [ ] No new heavy/required dep in the core install (heavy = optional extra)
- [ ] New brain network / craft metric key → matching `explainers.json` entry
- [ ] No audio/video/secrets/weights committed
- [ ] User-facing text goes through the producer voice (`identity.py`)
