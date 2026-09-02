# Adverse event evidence layer.
#
# Everything here runs offline. No target requires a network call; the model
# path degrades to the deterministic rules baseline and says so.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
SEED    ?= 7
PORT    ?= 8000
HOLDOUT ?= P4_sponsor,P5_comment,P6_both

.DEFAULT_GOAL := help
.PHONY: help venv install demo generate ingest normalize extract definitions \
        compare evaluate retrieve discover ask trace replay eval silver \
        transport ablation knowledge tools serve test coverage clean distclean

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

venv: $(BIN)/python  ## Create the virtual environment

install: venv  ## Install the package and its development dependencies
	$(BIN)/pip install --quiet -e ".[dev]"

demo: install  ## Generate, normalize, extract, score, evaluate — end to end
	$(BIN)/aelayer demo --seed $(SEED)

generate: install  ## Generate the corpus: one truth, six renderings
	$(BIN)/aelayer generate --seed $(SEED)

ingest: install  ## Load the corpus and report what is in it
	$(BIN)/aelayer ingest

normalize: install  ## The deterministic path: where each attribute came from
	$(BIN)/aelayer normalize

extract: install  ## Normalize, extract from text, reconcile episodes, index
	$(BIN)/aelayer extract

definitions: install  ## List the definitions and which routes each accepts
	$(BIN)/aelayer definitions

compare: install  ## Compare v1 and v2 by the episodes each claims (scope required)
	$(BIN)/aelayer definitions --compare te_truncal_rash:1:2 \
	  --scope "truncal rash incidence after first exposure"

evaluate: install  ## Evaluate te_truncal_rash v1, with the route behind each verdict
	$(BIN)/aelayer evaluate --definition te_truncal_rash --version 1

silver: install  ## THE CENTREPIECE: extraction vs the study's own structured field
	$(BIN)/aelayer eval silver --attribute location \
	  --json reports/silver.json

transport: install  ## Hold out whole studies and report the drop
	$(BIN)/aelayer eval transport --holdout $(HOLDOUT)

ablation: install  ## What text recovery is worth, as a count of events
	$(BIN)/aelayer eval all --report reports/eval.md --json reports/eval.json

eval: install  ## Run every harness and write reports/eval.md
	$(BIN)/aelayer eval all --report reports/eval.md --json reports/eval.json

retrieve: install  ## Precise path: adjudicated episodes, usable as a cohort
	$(BIN)/aelayer retrieve RASH --region trunk --verdict case

discover: install  ## Discovery path: modifiers no catalogue value covers yet
	$(BIN)/aelayer retrieve rash --mode hybrid --unnormalized

ask: install  ## Compile a question, execute it, and trace the number to source
	$(BIN)/aelayer ask "how many rash cases were there after first exposure?"

tools: install  ## The agent's entire callable surface
	$(BIN)/aelayer knowledge tools

knowledge: install  ## What the program knowledge layer actually holds
	$(BIN)/aelayer knowledge status

serve: install  ## Serve the API and the UI on http://127.0.0.1:$(PORT)/
	$(BIN)/aelayer serve --port $(PORT)

test: install  ## Run the test suite
	$(BIN)/pytest -q

coverage: install  ## Run the tests with coverage over the whole package
	$(BIN)/pytest --cov=aelayer --cov-report=term-missing --cov-fail-under=85

clean:  ## Remove generated data, the store, runs and reports
	rm -rf store.db runs reports/eval.md reports/eval.json reports/silver.json \
	       reports/adjudication.jsonl
	find data/synthetic -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage

distclean: clean  ## Also remove the virtual environment
	rm -rf $(VENV) src/*.egg-info
