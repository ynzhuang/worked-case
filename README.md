# Adverse event evidence layer

A working prototype of a clinical evidence layer over completed-trial adverse
event data. It converts free-text AE evidence into validated,
provenance-bearing **event objects**, evaluates **versioned phenotype
definitions** over those objects, exposes the result through **retrieval**, and
puts a **specification-first agent** on top.

```bash
make demo     # generate, extract, evaluate v1, print the case table — offline, ~15s
make eval     # the full evaluation harness -> reports/eval.md
make serve    # the API and the single-page UI on http://127.0.0.1:8000/
make test     # 321 tests
```

---

## Read this first

**All data is synthetic.** The repository generates its own corpus. No real
patient data, and nothing derived from real patient data, is present anywhere
here. Every table row carries a `SYNTHETIC` column and every narrative carries a
synthetic header.

**The extractor is a configurable rule and lexicon baseline.** It is a
deterministic system of dictionaries, regular expressions and ConText-style cue
scoping, driven entirely by `config/`. It is not a trained clinical NLP model
and must not be described as one. Its recall depends on the surface forms
written into `config/concepts.yaml`.

**The metrics measure signal recovery, not clinical performance.** Gold labels
are the generator's own intent. A number in `reports/eval.md` says the pipeline
recovered a signal that was deliberately planted in a corpus it also wrote. That
is a much weaker claim than performance on real clinical text, and no figure
here transfers to a real study.

**MedDRA terms in the config are illustrative placeholders.** This repository
holds no MedDRA licence and ships no MedDRA content. Replace
`config/concepts.yaml` with a licensed extract before any real use.

### What this is not

- Not a replacement for MedDRA coding. Coded terms are inputs, preserved and used.
- Not a pharmacovigilance system. It supports secondary research on locked data;
  it does not perform regulatory signal management.
- Not a clinical decision support tool.
- Not trained on anything. There is no model here to train.

---

## The idea

Two artifacts are deliberately separated, and the separation is the point of the
whole system.

An **event object** answers *what happened to this patient*. It is per-record,
extracted, evidence-bearing, and stamped with the extractor version that produced
it. It reports what the text and the tables say. It assigns no evidence state and
decides no cases.

A **phenotype definition** answers *for this scientific question, which event
objects make this patient a case*. It is a declarative, versioned rule over event
objects. It is configuration, not code.

Nobody edits Python to change a case definition. A clinical scientist edits YAML,
the version increments, and prior runs remain reproducible against the definition
that produced them.

```
narrative + SDTM tables
        │
        ▼
   extraction            EventObject: concept, assertion, symptoms, labs,
   (rules + lexicon)     onset, severity, seriousness, action, outcome
        │                — every value carries the span it came from
        ▼
   phenotype             CaseAssignment: verdict, evidence state, the rule
   definition (YAML)     that fired, the spans behind it, the version that
        │                produced it
        ▼
   retrieval  ·  run manifest  ·  agent
```

---

## Why the pieces are shaped this way

### Severity and seriousness are separate fields, always

Severity is the *intensity* of the event: mild, moderate, severe. Seriousness is
a *regulatory category defined by outcome*: death, life-threatening,
hospitalisation, disability, congenital anomaly, other medically important.

A mild event can be serious. A severe event can be non-serious. They are stored
separately, extracted from separate cue lists that never write to each other,
queried separately, and reported separately. The synthetic corpus deliberately
contains counterexamples to their conflation, and a test asserts both directions.

The agent refuses to answer a question that uses both words loosely, because
picking one silently would produce a count that answers neither question.

### Assertion is a first-class field, not a confidence discount

"No evidence of hypoglycemia" is not a low-confidence hypoglycemia event. It is a
documented absence. It is stored as `assertion: absent`, it is queryable as such,
and it is filterable as a structured predicate.

All six classes are extracted: `present`, `absent`, `hypothetical`, `historical`,
`family_history`, `uncertain`. Not just present and absent — a family history of
hypoglycemia and a hypothetical warning about it are different facts, and
collapsing either into "negative" loses information a reviewer will ask for.

### The anchor comes from structured data, the offset from text

A narrative says "six days after dose escalation". The escalation date lives in
the exposure domain, represented in SDTM as a new `EX` record with a higher dose
rather than as a flag. `src/aelayer/anchors.py` is the single place that decides
which `EX` record *is* the escalation, so the extractor and the evaluator can
never disagree about it.

Where no anchor resolves, `onset_offset_days` is populated and `onset_date` stays
null. The phenotype definition then decides what to do with that, via
`window.on_unresolved_onset: case | review | exclude`. The extractor never makes
that call.

### Units are converted once, at extraction

Trials report glucose in mg/dL or mmol/L depending on region. A threshold rule
applied to an unconverted value misclassifies an entire study silently. Every
`LabValue` carries both the value as reported and its canonical equivalent, and
the evaluator converts the *threshold* into canonical units before comparing. A
study reporting 3.1 mmol/L and one reporting 56 mg/dL are describing the same
result and are treated as such.

### The evidence ladder exists because both alternatives are wrong

Pooling only explicit coded events undercounts systematically — one study in the
corpus never coded hypoglycemia at all, and every case in it survives only in
narrative. Treating every symptom mention as a case manufactures signal.

So the definition assigns a state, and the states have a defensible ordering:

| state | meaning |
|---|---|
| `explicit` | a coded term for the concept, or an asserted verbatim mention |
| `supported` | no explicit mention, but corroborating objective evidence plus compatible clinical features |
| `possible` | clinical features and contextual evidence, without confirmation |
| `absent` | the concept is mentioned and negated |
| `none` | no qualifying evidence |

The primary analysis uses a threshold the definition names. The remainder is
routed to adjudication and **reported as a separate count rather than
discarded** — in the CLI, in the API, in the UI, and in the evidence package the
agent returns.

### Every derived value traces to a span

`Span(doc_id, start, end, field, extracted_value, text)`. Every non-null field on
every event object is backed by at least one. Values read from a structured table
point at a rendered form of that row rather than at nothing.

This is enforced, not aspired to: `EventObject.missing_provenance()` names any
populated field without a span, `make demo` and `aelayer extract` fail loudly if
the list is non-empty, the harness reports violations as defects, and a test
asserts the invariant over the whole corpus. The UI highlights the spans in the
source text so you can see the claim rather than take it.

### No hierarchy walking

Concept expansion uses the catalogue's own synonyms and coded terms. Grouping
above the term level is an explicit, named list in `concepts.yaml`
(`concept_groups`), never inferred by walking a dictionary hierarchy as though it
were a subsumption ontology.

---

## The configuration

These are the core deliverable. All three are schema-validated on load; a
definition that fails validation does not run, and the error says which path
failed and why.

### `config/concepts.yaml`

Concepts with their coded terms, lexicons and abbreviations; symptom sets and
their surface forms; lab tests with canonical units, conversion factors and
plausible ranges; and explicit concept groups.

Ambiguous abbreviations declare a context gate. `hypo` fires as hypoglycemia only
when a glucose value or a qualifying symptom sits in the same sentence:

```yaml
abbreviations: ["hypo", "hypos"]
context_required: [abbreviations]
context_gate:
  lab_tests: [GLUCOSE]
  symptom_sets: [neuroglycopenic, autonomic]
  scope: sentence
```

A concept may also declare what raises it as a candidate with no mention at all —
the mechanism behind the `supported` and `possible` states:

```yaml
candidate_evidence:
  symptom_sets: [neuroglycopenic, autonomic]
  min_symptoms: 1
  lab_tests:
    - {test: GLUCOSE, below: 80, unit: mg/dL}
```

The `below: 80` bound is deliberately wider than any case threshold a definition
uses, so the extractor narrows the field without pre-empting the definition's
decision.

### `config/extraction.yaml`

Assertion cues by class and direction, pseudo-cues, terminators and precedence;
temporal patterns with anchor aliases; anchor derivation rules against `EX`; lab
value patterns with conservative unit inference; cue lists for severity,
seriousness, relatedness, action, outcome, rechallenge and rescue.

### `config/phenotypes/<id>.v<n>.yaml`

One file per definition version. See
`config/phenotypes/te_symptomatic_hypoglycemia.v1.yaml` for the full shape.

The rule language is small and closed. Predicates: `coded_term_matches_concept`,
`has_coded_term`, `lexicon_match`, `lab`, `symptoms`, `rescue_treatment`,
`onset_offset_days`, and membership tests on `assertion`, `severity`,
`seriousness`, `relatedness`, `action_taken`, `outcome`, `rechallenge`.
Combinators: `any`, `all`, `not`. An unknown predicate is a load error, not a
silently-false rule.

---

## Versioning and provenance

Three content-derived hashes, stamped on every output row and every run manifest:

| hash | covers |
|---|---|
| `extractor_version` | code version plus `extraction.yaml` and `concepts.yaml` |
| `definition_hash` | the phenotype definition file's content |
| `snapshot_id` | every input file in the data directory |

The **run id is derived from all three plus the resolved spec**, so identical
inputs always produce an identical run id, and `aelayer replay <run_id>`
re-executes and compares. When a replay fails it says *which* input moved — the
data, the config, or the definition — rather than just that it did.

The `snapshot_id` deliberately excludes the gold answer key: gold is evaluation
scaffolding, not input data, and a run id must not change when only the answer
key changes.

### The definition lifecycle

- `draft` — being written. Will not run in a reproducible run without an explicit
  `--allow-draft`, because its content can still change under the same version
  number.
- `frozen` — published. The loader **refuses to overwrite it**. A change to what
  qualifies as a case is a new version, not an edit.
- `superseded` — replaced by a later version, still loadable and still replayable,
  because a prior analysis was built on it.

`v2` is a new file. This repository ships both versions; `v2` raises the glucose
threshold from the ADA Level 1 alert value of 70 mg/dL to the Level 2 value of 54:

```console
$ aelayer definitions --diff te_symptomatic_hypoglycemia:1:2
  evidence_rules[supported].when.all[0].lab.value: 70 -> 54
  supersedes: None -> 'te_symptomatic_hypoglycemia.v1'
  version: 1 -> 2

$ aelayer evaluate --version 1
  verdicts     {'case': 167, 'excluded': 113, 'review': 50}

$ aelayer evaluate --version 2
  verdicts     {'case': 144, 'excluded': 119, 'review': 67}
```

Twenty-three subjects leave the case set — seventeen to `review`, six to
`excluded` — and the v1 run still replays byte for byte. That is the whole
argument for the separation, in three commands.

---

## Interfaces

### CLI

```
aelayer generate --seed 7 --studies 4     # synthetic corpus with gold labels
aelayer ingest data/synthetic             # what is in a corpus
aelayer extract --out store.db            # event objects + retrieval index
aelayer definitions                       # list; --show, --diff id:1:2
aelayer evaluate --definition te_symptomatic_hypoglycemia --version 1
aelayer retrieve HYPOGLYCEMIA --assertion present --window 0:14
aelayer ask "symptomatic hypoglycemia within 14 days of escalation" --approve
aelayer eval --report reports/eval.md
aelayer replay <run_id>
aelayer runs
aelayer demo
aelayer serve
```

### API

`GET /api/summary` · `POST /extract` · `POST /evaluate` · `GET /retrieve` ·
`GET /definitions` · `GET /definitions/{id}/diff?left=&right=` ·
`POST /definitions/candidate` · `POST /agent/compile` · `POST /agent/run` ·
`GET /runs` · `GET /runs/{id}` · `POST /runs/{id}/replay` ·
`GET /api/documents/{doc_id}`

Interactive docs at `/docs` once `make serve` is running.

### UI

A single page over the API, no build step and no external assets, with four
panels matching the pipeline:

1. **Source and event object side by side**, with every span highlighted in the
   narrative and colour-coded by field. Hover a highlight to see which field it
   supports.
2. **The phenotype definition as controls bound to the YAML**, with a version
   badge, a status badge, and a live diff. Editing a control builds a candidate
   next version that can be downloaded. The UI never mutates a frozen definition.
3. **Retrieval with a visible assertion filter**, showing the negation false
   positive rate change as you toggle it.
4. **The agent**, showing the compiled spec with an approval checkbox that gates
   the execute button, and the clarification when the question is
   underdetermined.

---

## The agent

`compile.py` turns a question into a `PhenotypeQuerySpec`. Two backends:

- **deterministic** (default, offline): template and slot matching against the
  definition catalogue.
- **LLM** (optional, only when `ANTHROPIC_API_KEY` is set): the model emits the
  spec as JSON and nothing else. Its output is schema-validated and **rejected on
  failure rather than repaired**, because a repaired spec is one nobody approved.
  It never computes a statistic and never names a case.

Non-negotiable behaviours, each covered by a test:

- The compiled spec is returned and **execution blocks until it is approved**.
- Execution calls only the four functions in `tools.py`: `cohort`, `retrieve`,
  `evaluate`, `summarise`.
- The result is an evidence package: counts by state, per-study breakdown,
  contributing spans, definition version and hash, extractor version, snapshot id,
  and stated limitations.
- Where the question leaves a rule underdetermined, the agent returns a
  clarification naming the specific ambiguity and its effect, rather than choosing
  a default.

```console
$ aelayer ask "serious severe hypoglycemia cases"
Clarification needed before anything can be run.

  Ambiguity: The question uses both severity language and seriousness language,
             and they are different fields.
  Effect:    Severity is the intensity of the event; seriousness is a regulatory
             category defined by outcome. A mild event can be serious and a
             severe event can be non-serious, so the two filters select different
             subjects. Collapsing them would produce a count that answers neither
             question.
  Options:
    - Filter on severity (mild / moderate / severe)
    - Filter on seriousness (death, life-threatening, hospitalisation, ...)
    - Report both as separate counts

  No specification was compiled and nothing was executed.
```

It also refuses to guess when the question names a window or a threshold that
differs from the definition's — because that is a change to the case definition,
which needs a new version, not a query parameter — when no definition matches at
all, and when a named study is not in the snapshot.

---

## The synthetic corpus

`aelayer generate` writes 4–6 studies of 40–80 subjects each: SDTM-shaped `DM`,
`AE`, `EX`, `LB` and `CM` tables plus one case narrative per AE record, and a
gold answer key.

Narratives are assembled from templates across **19 controlled patterns**:
explicit coded mentions, British and US spelling, one-edit misspellings including
transpositions, gated and ungated abbreviations, lab-plus-symptom cases,
contextual-only cases with rescue or a dose action, symptoms alone, negated,
hypothetical, historical and family-history mentions, uncertain mentions,
evidence split across sentences, out-of-window onsets, unresolved onsets, and
distractor concepts.

Onset is phrased eight ways — relative to a named anchor, relative with no
anchor, "within N days of", "the day after", "on the day of", study day, absolute
date, and vague quantifiers — and sometimes appears only in the `AE` table, so
the temporal extractor has to earn its place.

Studies differ on purpose:

| study | units | dictionary | notes |
|---|---|---|---|
| STUDY-01 | mg/dL | MedDRA 26.0 | conventional coding |
| STUDY-02 | mmol/L | MedDRA 24.1 | SI units throughout |
| STUDY-03 | mg/dL | MedDRA 25.0 | severe events only were collected |
| STUDY-04 | mmol/L | MedDRA 21.1 | **never coded hypoglycemia at all**; AE table carries almost nothing |
| STUDY-05 | mg/dL | MedDRA 27.0 | recent conventions |
| STUDY-06 | mmol/L | MedDRA 23.0 | crossover; rechallenge language, partial AE columns |

Every record carries a `gold` block: true concept, assertion, onset offset and
phrasing, symptoms, labs, severity, seriousness, action, outcome, evidence state
and case status under v1. Gold never asserts a value the record does not state —
a test enforces that for onset.

---

## Evaluation

`make eval` writes `reports/eval.md` with real numbers for every metric.

**Extraction** — precision, recall and F1 per field, broken out by assertion class
and by narrative pattern, with the counts behind every rate. A six-by-six
assertion confusion matrix. Onset accuracy broken out by phrasing, which is where
the cost of mapping "several days" to a fixed value stays visible instead of
hiding in a pooled number.

**Phenotype** — PPV and sensitivity of the `case` verdict against the gold labels,
pooled and per study, plus the full `case`/`review`/`excluded` confusion and the
evidence-state confusion.

**Retrieval** — recall@k, precision@k and MRR per study, and the **negation
false positive rate measured with the assertion filter on and off**. That
contrast is the headline number, because it quantifies what a structured
assertion column buys over hoping an embedding encoded the negation. Recall@k is
reported alongside its mathematical ceiling, since with 63 relevant documents
recall@5 cannot exceed 0.079 however perfect the ranking; precision@k is the
figure that is not capped.

A concept filter is structural, not lexical. An event raised from symptoms and a
glucose value carries no surface form of the concept anywhere in its narrative,
so requiring a text match would drop exactly the records the evidence ladder
exists to catch. The lexical index orders results; it does not gate them.

**Stability** — extraction, run id and results hash across repeated invocations.

**Definition sensitivity** — case assignment re-run across a sweep of one rule
parameter (glucose 70 → 63 → 54 → 45, window 28 → 14 → 7 → 3), reporting how the
case count moves. This makes definitional drift a measurement rather than an
argument.

The harness is not decoration. Building it surfaced a series of real defects
that are fixed in this repository, among them: an unbounded glucose arm in
candidate gating that raised hypoglycemia on hyperglycemia narratives; lab and
concomitant-medication records from a neighbouring event attributed to the wrong
record; a `same_day` onset phrasing with no matching extraction pattern; an
outcome cue that never matched the phrase the corpus actually used; and two
places where the gold key asserted something the record never stated.

---

## Layout

```
config/                 concepts, extraction rules, phenotype definitions
  └── phenotypes/       one file per definition version
data/synthetic/         generated tables, narratives and gold labels
src/aelayer/
  models.py             EventObject, Span, LabValue, PhenotypeDefinition, ...
  catalog.py            config loading and validation
  anchors.py            anchor resolution against the exposure record
  ingest.py             SDTM tables, narratives and gold into a store
  generate.py           the synthetic corpus generator
  hashing.py            the three content hashes
  extract/
    text.py             sentence splitting, tokenising, edit distance
    concepts.py         lexicon and coded-term matching, abbreviation gating
    assertion.py        ConText/NegEx-style cue-and-scope classifier
    temporal.py         offset patterns and anchor resolution
    values.py           labs and units, severity, seriousness, action, outcome
    engine.py           orchestration; produces EventObject list
  phenotype/
    loader.py           YAML load, schema validation, version hashing, diffing
    evaluator.py        rule evaluation over EventObjects -> CaseAssignment
  retrieval/
    index.py            SQLite FTS5 plus structured attribute columns
    query.py            compositional query
    dense.py            optional; degrades to lexical with no local model
  agent/
    compile.py          question -> PhenotypeQuerySpec, or a clarification
    tools.py            the only callable surface
    run.py              approval gate, then execution
  eval/
    metrics.py          P/R/F1, confusion matrices, ranking metrics
    harness.py          the five metric families
    report.py           markdown rendering
  pipeline.py           one assembled path shared by CLI, API and agent
  runs.py               run manifests and replay
  api.py                FastAPI
  cli.py                Typer
ui/                     static single-page client
tests/                  321 tests
```

**Stack.** Python 3.11+, Pydantic v2, FastAPI, Typer, SQLite with FTS5, PyYAML,
pytest. No cloud services. No PHI. Everything runs on a laptop with the network
cable pulled.

---

## Out of scope

Regulatory signal detection and disproportionality analysis; genetics and omics
analysis, though `CaseAssignment` is shaped so a case-control export could be
handed to an existing pipeline; multi-user auth and RBAC; real terminology
licences; and model training of any kind.
