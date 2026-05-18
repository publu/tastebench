# tastebench — clone, then `make`. That's it.
#
#   make            create .venv (core deps) and launch the worker
#   make setup      just build the venv (core, model-free craft layer)
#   make brain      add the ~20 GB TRIBE stack (optional, gated Llama)
#   make test       run the model-free smoke suite
#   make clean      delete the venv
#
# Core install pulls only numpy/librosa/rich etc. — no torch, no model,
# sub-second. The worker watches ./workspace (never the package source).

# Prefer a 3.11/3.12 interpreter (the brain extra wants it); fall back to
# python3 for the model-free core, which installs on 3.13+ too. Override
# with `make PY=python3.12`.
PY      ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3)
VENV    := .venv
BIN     := $(VENV)/bin
STAMP   := $(VENV)/.core-deps
WORKDIR ?= workspace

.DEFAULT_GOAL := run

$(VENV):
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install -q --upgrade pip

$(STAMP): $(VENV) pyproject.toml
	$(BIN)/python -m pip install -q -e .
	@touch $(STAMP)

.PHONY: setup
setup: $(STAMP)
	@echo "ready — run 'make' to start the worker, or '$(BIN)/tastebench --help'"

.PHONY: run
run: $(STAMP)
	@mkdir -p $(WORKDIR)
	$(BIN)/tastebench worker $(WORKDIR)

.PHONY: brain
brain: $(STAMP)
	$(BIN)/python -m pip install -e ".[brain]"
	@echo "brain extra installed — now: huggingface-cli login && $(BIN)/python scripts/download_models.py"

.PHONY: test
test: $(VENV)
	$(BIN)/python -m pip install -q -e ".[dev]"
	$(BIN)/python -m pytest -q

.PHONY: clean
clean:
	rm -rf $(VENV)
