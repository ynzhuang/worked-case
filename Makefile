# Adverse event evidence layer.
#
# Everything here runs offline. No target requires a network call; the model
# path degrades to deterministic-only and says so.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
SEED    ?= 7
STUDIES ?= 6
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install demo generate ingest normalize extract definitions \
        compare evaluate retrieve discover ask trace eval replay knowledge \
        serve test coverage clean distclean

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

venv: $(BIN)/python  ## Create the virtual environment

install: venv  ## Install the package and its development dependencies
	$(BIN)/pip install --quiet -e ".[dev]"

demo: install  ## Generate, normalize, extract, reconcile, evaluate — end to end
	$(BIN)/aelayer demo --seed $(SEED) --studies $(STUDIES)

generate: install  ## Generate the synthetic corpus (six renderings of one truth)
	$(BIN)/aelayer generate --seed $(SEED) --studies $(STUDIES)

ingest: install  ## Load the corpus and report what is in it
	$(BIN)/aelayer ingest

normalize: install  ## Run the deterministic path and report collection states
	$(BIN)/aelayer normalize

extract: install  ## Normalize, enrich from narrative, reconcile episodes, index
	$(BIN)/aelayer extract

definitions: install  ## List the phenotype definitions and their content hashes
	$(BIN)/aelayer definitions

compare: install  ## Compare v1 and v2 by the episodes each one claims (scope required)
	$(BIN)/aelayer definitions --compare te_symptomatic_hypoglycemia:1:2 \
	  --scope "hypoglycemia incidence after dose escalation"

evaluate: install  ## Evaluate the v1 definition over episodes
	$(BIN)/aelayer evaluate --definition te_symptomatic_hypoglycemia --version 1

retrieve: install  ## Precise path: adjudicated episodes, usable as a cohort
	$(BIN)/aelayer retrieve HYPOGLYCEMIA --window 0:14

discover: install  ## Discovery path: narrative mentions, every one a candidate
	$(BIN)/aelayer retrieve HYPOGLYCEMIA --mode lexical --assertion present

ask: install  ## Compile a question, execute it, and trace the number to source
	$(BIN)/aelayer ask "how many subjects had symptomatic hypoglycemia?"

eval: install  ## Run the full evaluation harness and write reports/eval.md
	$(BIN)/aelayer eval --report reports/eval.md --json reports/eval.json

knowledge: install  ## What the program knowledge layer actually holds
	$(BIN)/aelayer knowledge status

serve: install  ## Serve the API and the UI on http://127.0.0.1:$(PORT)/
	$(BIN)/aelayer serve --port $(PORT)

test: install  ## Run the test suite
	$(BIN)/pytest -q

coverage: install  ## Run the tests with coverage over the whole package
	$(BIN)/pytest --cov=aelayer --cov-report=term-missing --cov-fail-under=85

clean:  ## Remove generated data, the store, runs and reports
	rm -rf store.db runs reports/eval.md reports/eval.json
	find data/synthetic -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage

distclean: clean  ## Also remove the virtual environment
	rm -rf $(VENV) src/*.egg-info
