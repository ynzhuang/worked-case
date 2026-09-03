# Adverse event evidence layer.
#
# Everything here runs offline. No target requires a network call; the model
# path degrades to the deterministic rules baseline and says so.
#
# Two targets are the point of the whole build:
#
#   make ablation   is reading narrative text worth it? Ends in a decision.
#   make silver     what the extraction is worth, with both caveats printed.
#
# If you run nothing else, run those two.

PYTHON  ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
SEED    ?= 7
PORT    ?= 8000
DEFN    ?= cutaneous_mucosal
HOLDOUT ?= P_absent,P_negated,P_version,P_concept_variant

.DEFAULT_GOAL := help
.PHONY: help venv install demo generate ingest normalize extract \
        supportability definitions compare evaluate ablation silver transport \
        eval retrieve discover ask tools knowledge serve test coverage lint \
        conflict clean distclean

help:  ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip

venv: $(BIN)/python  ## Create the virtual environment

install: venv  ## Install the package and its development dependencies
	$(BIN)/pip install --quiet -e ".[dev]"

demo: install  ## The whole path end to end, ending in a decision
	$(BIN)/aelayer demo --seed $(SEED)

# -- the corpus ---------------------------------------------------------------

generate: install  ## Generate the corpus: one truth, seven renderings
	$(BIN)/aelayer generate --seed $(SEED)

ingest: install  ## Load the corpus and report what is in it
	$(BIN)/aelayer ingest

normalize: install  ## The deterministic path, and the reconciliation split
	$(BIN)/aelayer normalize

extract: install  ## The model path: assertions, spans, and the abstention rate
	$(BIN)/aelayer extract

# -- the two that matter ------------------------------------------------------

ablation: install  ## THE EXPERIMENT: is reading text worth it? States a decision
	$(BIN)/aelayer ablation $(DEFN)

silver: install  ## THE HARNESS: extraction vs a masked comparator, with caveats
	$(BIN)/aelayer eval silver --queue reports/adjudication.jsonl

# -- the rest -----------------------------------------------------------------

supportability: install  ## Which studies can answer, from metadata alone
	$(BIN)/aelayer supportability

definitions: install  ## List the definitions and which routes each accepts
	$(BIN)/aelayer definitions

compare: install  ## Compare v1 and v2 by the records each claims (scope required)
	$(BIN)/aelayer definitions $(DEFN) --compare 1:2 \
	  --scope "cutaneous adverse events with mucosal involvement"

evaluate: install  ## Evaluate the definition, with denominators per study
	$(BIN)/aelayer evaluate $(DEFN)

transport: install  ## Hold out whole studies and report the drop
	$(BIN)/aelayer eval transport $(DEFN) --holdout $(HOLDOUT)

eval: install  ## Run every harness and write reports/evaluation.md
	$(BIN)/aelayer eval all $(DEFN) --report reports/evaluation.md

retrieve: install  ## Precise path: documented negatives, usable as a cohort
	$(BIN)/aelayer retrieve --assertion absent --verdict non_case

discover: install  ## Discovery path: text mentions, all of them candidates
	$(BIN)/aelayer retrieve --text "mucosal"

ask: install  ## Compile a question, execute it, and trace the number to source
	$(BIN)/aelayer ask "incidence of cutaneous events with mucosal involvement"

conflict: install  ## A question the agent refuses to answer, and why
	-$(BIN)/aelayer ask "cutaneous events without mucosal involvement"

tools: install  ## The agent's entire callable surface
	$(BIN)/aelayer knowledge tools

knowledge: install  ## What the program knowledge layer actually holds
	$(BIN)/aelayer knowledge status

serve: install  ## Serve the API and the UI on http://127.0.0.1:$(PORT)/
	$(BIN)/aelayer serve --port $(PORT)

# -- checks -------------------------------------------------------------------

test: install  ## Run the test suite
	$(BIN)/pytest -q

coverage: install  ## Run the tests with coverage over the whole package
	$(BIN)/pytest --cov=aelayer --cov-report=term-missing --cov-fail-under=85

lint: install  ## Byte-compile everything, so a syntax error cannot ship
	$(BIN)/python -m compileall -q src tests

clean:  ## Remove generated data, the store, runs and reports
	rm -rf store.db runs reports/evaluation.md reports/adjudication.jsonl
	find data/synthetic -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage

distclean: clean  ## Also remove the virtual environment
	rm -rf $(VENV) src/*.egg-info
