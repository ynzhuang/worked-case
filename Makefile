# Adverse event evidence layer.
#
# Everything here runs offline. No target requires a network call.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
SEED    ?= 7
STUDIES ?= 4
PORT    ?= 8000

.DEFAULT_GOAL := help
.PHONY: help venv install demo generate extract evaluate eval retrieve ask \
        replay serve test coverage lint clean distclean

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

venv: $(BIN)/python  ## Create the virtual environment

install: venv  ## Install the package and its development dependencies
	$(BIN)/pip install --quiet -e ".[dev]"

demo: install  ## Generate, extract, evaluate v1, print the case table with reasons
	$(BIN)/aelayer demo --seed $(SEED) --studies $(STUDIES)

generate: install  ## Generate the synthetic corpus
	$(BIN)/aelayer generate --seed $(SEED) --studies $(STUDIES)

extract: install  ## Extract event objects and build the retrieval index
	$(BIN)/aelayer extract

evaluate: install  ## Evaluate the v1 definition
	$(BIN)/aelayer evaluate --definition te_symptomatic_hypoglycemia --version 1

retrieve: install  ## Example retrieval, assertion filter on
	$(BIN)/aelayer retrieve HYPOGLYCEMIA --assertion present --window 0:14

ask: install  ## Compile a question into a spec, without executing it
	$(BIN)/aelayer ask "symptomatic hypoglycemia within 14 days of escalation"

eval: install  ## Run the full evaluation harness and write reports/eval.md
	$(BIN)/aelayer eval --report reports/eval.md --json reports/eval.json

serve: install  ## Serve the API and the UI on http://127.0.0.1:$(PORT)/
	$(BIN)/aelayer serve --port $(PORT)

test: install  ## Run the test suite
	$(BIN)/pytest

coverage: install  ## Run the tests with coverage on extract/, phenotype/, retrieval/
	$(BIN)/pytest --cov=aelayer.extract --cov=aelayer.phenotype \
	              --cov=aelayer.retrieval --cov-report=term-missing \
	              --cov-fail-under=80

clean:  ## Remove generated data, the store, runs and reports
	rm -rf store.db runs reports/eval.md reports/eval.json
	find data/synthetic -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage

distclean: clean  ## Also remove the virtual environment
	rm -rf $(VENV) src/*.egg-info
