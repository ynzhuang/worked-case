# Adverse event evidence layer

A working prototype of a clinical evidence layer over completed-trial adverse
event data. It takes one clinically relevant attribute — **anatomical
location** — and reads it out of whichever of **five different places** a study
happened to record it, into a single provenance-bearing shape. Above that it
derives episodes and trajectories, evaluates **versioned phenotype definitions**
with four verdicts, and scores its own text extraction against a **silver
standard** built from the studies' own structured fields.

```bash
make demo      # generate, normalize, extract, score, evaluate — offline, ~1s
make silver    # the centrepiece: extraction vs the study's own structured field
make eval      # every harness -> reports/eval.md
make serve     # the API and the single-page UI on http://127.0.0.1:8000/
make test      # 241 tests, 86% statement coverage
```

---

## Read this first

**All data is synthetic.** The repository generates its own corpus. No real
patient data, and nothing derived from real patient data, is present anywhere
here. Every table row carries a `SYNTHETIC` column.

**The extraction backend is a configurable rule and lexicon baseline.** By
default it is a deterministic system of dictionaries and scope rules driven
entirely by `config/`. It is not a trained clinical NLP model and must not be
described as one. An LLM backend can be swapped in behind the same interface;
the manifest records which one ran, and with the network disconnected the model
path degrades to the rules baseline and says so.

**Coded terms in the configuration are illustrative placeholders.** This
repository holds no terminology licence and ships no licensed dictionary
content. The terms in `config/concepts.yaml` are stand-ins with the right
*shape* — a preferred term, lower-level terms, and different sets under
different dictionary versions — so that version bridging can be exercised.

**The metrics measure signal recovery, not clinical performance.** Gold labels
are the generator's own intent. A number in `reports/eval.md` says the pipeline
recovered a signal that was deliberately planted in a corpus it also wrote.

**Nothing is trained.** There is no model in this repository to train, and no
code path that fits parameters to data.

### What this is not

- Not a replacement for coding. Coded terms are inputs: preserved, and used.
- Not a pharmacovigilance system. It supports secondary research on locked data.
- Not a clinical decision support tool.
- Not a claim that consistency is validity. See *Representation invariance*.

---

## The claim

One clinically relevant attribute can live in any of five places, and which one
applies is a per-study collection decision:

| # | home | example variable |
| --- | --- | --- |
| 1 | a standard structured qualifier | `AELOC` |
| 2 | a sponsor-defined supplemental variable | `SUPPAE.RASHSITE` |
| 3 | the investigator-reported term | `AETERM` — "skin rash on chest" |
| 4 | a linked comment | `CO.COVAL` |
| 5 | nowhere | — |

Terminology guidance is why (3) happens routinely: where no available term
covers both the event and the body site, the **event** term is selected and the
site is not carried in the coded term. Nothing is coded incorrectly. The site
simply survives somewhere else, or not at all.

**The consequence this prototype demonstrates:** one versioned rule runs across
all five, returning the same verdict where the evidence supports it and
`not_ascertainable` where it does not.

---

## The worked example

Concept **rash**; modifiers **location** and **pattern**. `te_truncal_rash` v1
requires a rash concept, a location in {CHEST, ABDOMEN, BACK}, and an onset
within 14 days of first exposure.

| | P1 | P2 | P3 | P4 | P5 | P6 |
| --- | --- | --- | --- | --- | --- | --- |
| coded event | rash | rash | rash | rash | rash | rash |
| location home | `AELOC` | `AETERM` | none | `SUPPAE.RASHSITE` | `CO.COVAL` | `AELOC` + `AETERM` |
| reported term | terse | rich | prespecified | terse | terse | rich |
| method | direct | extracted | — | normalized | extracted | direct |
| verdict | case | case | **not ascertainable** | case | case | case |

**P3 is the one to get right.** The rash is real, the timing qualifies, and the
phenotype still cannot be evaluated. That is not a negative case, and it is not
a review item either — no reviewer can settle it. It is counted and reported on
its own.

P6 is the evaluation profile: it records the location in a structured variable
*and* in the investigator's own words, which is what makes a silver standard
possible without anyone hand-annotating anything.

---

## The data model

```python
Method     = Literal["direct", "normalized", "extracted"]        # never "inferred"
SourceKind = Literal["structured_standard", "structured_sponsor",
                     "reported_term", "comment", "linked_form", "derived"]
Availability = Literal["collected", "not_collected_by_protocol",
                       "not_applicable_gated", "pending_ongoing",
                       "not_representable", "unknown"]

class Attribute(BaseModel, Generic[T]):
    value: T | None
    source: SourceKind | None
    source_variable: str | None      # "AELOC", "AETERM", "SUPPAE.RASHSITE", "CO.COVAL"
    method: Method | None
    evidence: list[Span] = []        # required whenever method == "extracted"
    availability: Availability
```

Four invariants, each enforced in the model and asserted by a test:

- `method == "extracted"` ⟹ at least one span. A value with no text behind it
  cannot be checked by anyone.
- `method == "direct"` ⟹ `source == "structured_standard"`, and no model
  touched it.
- `availability != "collected"` ⟹ `value is None`. Only a collected attribute
  has one.
- **There is no `inferred`.** A value the system worked out for itself, with
  nothing in the source to point at, is not an attribute of a patient. A test
  asserts the word appears nowhere in the package.

Source records are immutable. Episodes and trajectories derive above them and
can be recomputed; deleting them loses nothing.

---

## The deterministic and model paths, enforced in code

The deterministic path reads structured variables — `direct` from a standard
one, `normalized` through a declared sponsor mapping. The model path reads
language, and only language.

```python
from aelayer.guards import assert_model_path_permitted, askable_attributes
```

Every model request passes through the guard, which refuses one that names an
already-settled attribute or reads a structured source. A test asserts, over
every record in the corpus, that no structured value ever reaches a backend, and
`config/extraction.yaml` cannot even *declare* a structured source readable.

**Abstention is correct behaviour.** Where the text does not support a value the
backend returns nothing and says so, and the abstention rate is reported as a
metric rather than counted as a failure. A phrase no lexicon carries — "torso",
"midriff", "shoulder blade area" — produces an abstention, not a guess.

---

## The silver-standard harness

`make silver` — this is the piece that turns the prototype from a demo into
something that reports real numbers.

For a study that records the location in **both** a structured variable and the
reported term:

1. mask the structured variable from the extractor
2. run the model path over the text alone
3. normalize both sides to the concept catalogue
4. compare the extracted value against the masked structured value

It is called a **silver** standard everywhere it is reported, never ground
truth. The structured field has its own error rate — a site can mistype a coded
qualifier as easily as it can write an ambiguous phrase — so a disagreement
means the two disagree, not that the extractor is wrong.

The output includes an **adjudication queue**: every disagreement, every
low-confidence prediction, **and a random sample of agreements**. The sample is
not optional. Without it you only ever inspect failures and can never estimate
the silver standard's own error rate.

---

## Four verdicts

```yaml
required_attributes:
  - name: location
    in: [CHEST, ABDOMEN, BACK]
    accept_methods: [direct, normalized, extracted]   # route-agnostic by design
    on_unavailable: not_ascertainable
  - name: onset
    window: {unit: days, min: 0, max: 14, anchor: first_exposure}
    on_unresolved: review
```

| verdict | meaning |
| --- | --- |
| `case` | every required attribute is present and satisfies its rule |
| `not_case` | a required attribute is present and **fails** its rule |
| `not_ascertainable` | a required attribute is unavailable and unrecoverable |
| `review` | present but weakly supported, or an onset that will not resolve |

Precedence: a requirement that is present and fails settles the episode as a
negative whatever else is missing — knowing the rash was on the arm makes it not
a truncal rash even with the onset date gone. Only when nothing has failed does
an unavailable requirement make the episode unascertainable.

`accept_methods` listing all three routes is the point: the rule does not care
whether the location came from `AELOC` or from the investigator's own words,
only that it is present and points at something. `te_truncal_rash` **v2** lists
only the structured routes — a narrower scientific claim, not a bug fix — and
comparing the two measures what text extraction is worth.

---

## What it currently reports

On the shipped corpus (seed 7, six profiles, 228 subjects, 241 source records,
228 episodes), `te_truncal_rash` v1:

| | |
| --- | --- |
| **silver standard** — precision / recall / F1 | 0.857 / 0.714 / 0.779 |
| coverage / abstention rate | 0.795 / 0.205 |
| adjudication queue | 5 disagreements + 20 sampled agreements |
| **phenotype** — PPV / sensitivity | 1.000 / 1.000 |
| **not-ascertainable rate** | **0.162**, agreement with gold 1.000 |
| **value ablation** — cases findable only through text | **20 of 54 (37.0%)** |
| unascertainable episodes resolved by reading text | 37 |
| availability confusion — accuracy | 0.996, 0 missing read as collected |
| **transportability** (whole studies held out) | sensitivity drop 0.000, not-ascertainable rate −0.237 |
| representation invariance | 1.000 where the evidence supports it, 0.333 raw |
| reproducibility | manifest id, results hash and normalization all stable |

Four of these are worth reading carefully rather than admiring.

**37% of qualifying events are findable only through text.** Without the
extraction layer they are not negatives — they are unascertainable, and 37
episodes move out of that bucket when the text is read. That single number is
the business case for the whole layer, which is why it is a named output rather
than something a reader has to derive.

**The silver numbers are the informative ones; the phenotype numbers are near
ceiling by construction.** Given a correct attribute, the verdict logic
reproduces the answer key exactly — that is what it should do. Extraction error
shows up in the silver standard, and the corpus is built so the two stay
separable: the disagreements it seeds between a structured qualifier and the
investigator's prose are always within a verdict class (one truncal site
confused for another), so they populate the adjudication queue without
contaminating the phenotype metrics. That is a deliberate design choice, stated
here so nobody mistakes a clean phenotype table for a hard-won one.

**Transportability shows no sensitivity drop and a large change in the
not-ascertainable rate.** The held-out profiles record the location more often,
so the drop that matters here is not accuracy but *answerability* — which is
exactly the thing a random row split could never surface, because it would leak
every profile's conventions into both sides.

**Raw invariance is 0.333, and that is correct.** A study that never collected
the location returns `not_ascertainable`, and it is right to. Counting that as a
disagreement would score the system for the study's collection decision. Where
the evidence supports a verdict at all, agreement is 1.000.

---

## Retrieval, the knowledge layer, and the agent

**Precise retrieval** runs over normalized attributes and a definition's
verdict, and never consults an embedding for cohort membership. **Discovery**
returns candidates that cannot enter a cohort — `as_cohort()` raises — and
answers the question the precise path cannot: *which modifiers does no catalogue
value cover yet?* That is the honest job of semantic search here.

**The knowledge layer** is forward-capture: it accrues from executions, is empty
on day one, and says so. Manifests record which attribute sources contributed,
so a later reader can see that a cohort depended on text extraction. Comparing
two definition versions is executed rather than textual, and requires a scope —
an unscoped programme-wide sweep is refused, because that is auditing
colleagues' past choices rather than reusing evidence.

**The agent** has a typed, permissioned tool surface and nothing else:

```
phenotype.resolve  cohort.run  evidence.search  exposure.build
covariates.build   stats.compare  omics.run
```

Every call is validated against an input schema before it runs and an output
schema before it returns. There is no SQL surface and no tool that writes to a
source record. There is also no approval gate: approving a specification you
cannot independently evaluate is ceremony. What makes a number checkable is the
trace returned with it —

```
number -> analysis run -> cohort -> definition version -> episodes -> records -> spans
```

— which `aelayer trace <manifest-id>` walks and prints. A number that cannot be
traced end to end is a failing test, not a caveat.

**Trajectories** are deliberately minimal: exposures and episodes on one ordered
timeline with offsets, which is what the phenotype window and a "distribution by
time since exposure" question both need. Not progression modelling.

**No graph store.** Traversal depth here is known and shallow — event → record →
span, and event → exposure → covariate — both joins over a fixed schema.
Relational plus a text index plus a definition registry. Revisit only if
relationships become unknown ahead of query time.

---

## Interfaces

```bash
aelayer generate --seed 7               # one truth, six renderings
aelayer normalize                       # the deterministic path, by route
aelayer extract                         # normalize, extract, reconcile, index
aelayer eval silver --attribute location
aelayer eval transport --holdout P4_sponsor,P5_comment,P6_both
aelayer eval all --report reports/eval.md
aelayer evaluate --version 1            # verdicts, each with the route behind it
aelayer definitions --compare te_truncal_rash:1:2 --scope "..."
aelayer retrieve RASH --region trunk --verdict case
aelayer retrieve rash --mode hybrid --unnormalized
aelayer ask "how many rash cases after first exposure?"
aelayer trace <manifest-id>             # the number, back to the text
aelayer replay <manifest-id>            # hash for hash
aelayer knowledge tools                 # the agent's entire callable surface
```

The HTTP API mirrors the CLI through the same pipeline object. `make serve` also
serves a five-panel UI: one attribute across five homes; the silver standard and
its adjudication queue; the four verdicts and the value ablation; the two
retrieval paths; and the agent with its trace. The UI is API-driven, so opening
`ui/index.html` from the filesystem shows an empty shell — run `make demo`, then
`make serve`.

---

## Layout

```
config/
  concepts.yaml           concepts, attribute catalogues, surface forms
  study_profiles.yaml     where each attribute lives, per study
  extraction.yaml         scope rules, connectors, anchors, confidences
  phenotypes/             one file per definition version
src/aelayer/
  models.py               Attribute, CanonicalAERecord, CanonicalAEEpisode, Trajectory
  profiles.py             the profile config, homes, gates, sponsor codelists
  guards.py               the deterministic/model boundary, in code
  generate.py             one truth, six renderings, and the answer key
  normalize/              the deterministic path
  extract/                the model path: modifiers, backends, engine
  episode.py              records -> episodes, with the route preserved
  trajectory.py           exposures and episodes on one timeline
  phenotype/              definition loading, validation, four-verdict evaluation
  silver.py               the silver-standard harness and adjudication queue
  eval/                   phenotype, ablation, availability, transport, invariance
  retrieval/              precise and discovery paths over SQLite FTS5
  knowledge/              the registry and the executed definition comparison
  agent/                  typed permissioned tools, compile, run, trace
  runs.py                 manifests, the result store, replay
  api.py  cli.py  pipeline.py
ui/                       static single-page client
tests/                    241 tests
```

**Stack.** Python 3.11+, Pydantic v2, FastAPI, Typer, SQLite with FTS5, PyYAML,
pytest. No cloud services. Everything runs on a laptop with the network cable
pulled.

---

## Out of scope

Regulatory signal detection and disproportionality analysis; genetics and omics
analysis, though `omics.run` exports a case-control file shaped for an existing
pipeline — with `status` null for review and not-ascertainable subjects, because
coding them either way would put an unadjudicated judgement into someone else's
analysis; multi-user auth and RBAC; real terminology licences; and model
training of any kind.
