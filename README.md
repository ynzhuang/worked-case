# Adverse event evidence layer

A working prototype of a clinical evidence layer over completed-trial adverse
event data. It reads adverse event evidence in the forms studies actually
record it — structured CRF fields, verbatim terms, narrative text, linked event
forms — and turns it into **source-faithful canonical records** that say what
each study collected and, just as carefully, what it did not. Above those
records it derives **episodes**, evaluates **versioned phenotype definitions**
over the episodes, exposes both a precise and a discovery retrieval path, and
puts a specification-first agent on top whose every number can be followed back
to the sentence a site wrote.

```bash
make demo     # generate, normalize, extract, reconcile, evaluate — offline, ~35s
make eval     # the full evaluation harness -> reports/eval.md
make serve    # the API and the single-page UI on http://127.0.0.1:8000/
make test     # 368 tests, 88% statement coverage
```

---

## Read this first

**All data is synthetic.** The repository generates its own corpus. No real
patient data, and nothing derived from real patient data, is present anywhere
here. Every table row carries a `SYNTHETIC` column and every narrative carries a
synthetic header.

**The extraction backend is a configurable rule and lexicon baseline.** By
default it is a deterministic system of dictionaries, regular expressions and
ConText-style cue scoping, driven entirely by `config/`. It is not a trained
clinical NLP model and must not be described as one. An LLM backend can be
configured in its place; the manifest records which one ran, and with the
network disconnected the model path degrades to deterministic-only and says so
in its notes.

**Coded terms in the configuration are illustrative placeholders.** This
repository holds no MedDRA licence and ships no MedDRA content. The terms in
`config/concepts.yaml` are stand-ins with the right *shape* — preferred terms,
lower-level terms, and different sets under different dictionary versions — so
that dictionary-version bridging can be exercised. Replace them with a licensed
extract before any real use.

**The metrics measure signal recovery, not clinical performance.** Gold labels
are the generator's own intent. A number in `reports/eval.md` says the pipeline
recovered a signal that was deliberately planted in a corpus it also wrote. That
is a much weaker claim than performance on real clinical text, and no figure
here transfers to a real study.

**Nothing is trained.** There is no model in this repository to train, and no
code path that fits parameters to data.

### What this is not

- Not a replacement for coding. Coded terms are inputs: preserved, and used.
- Not a pharmacovigilance system. It supports secondary research on locked data;
  it does not perform regulatory signal management.
- Not a clinical decision support tool.
- Not a claim that consistency is validity. See *Representation invariance*.

---

## The two levels, and why there are exactly two

**`CanonicalAERecord` — one per source record.** This is the grain the study
actually collected. It is never merged with another record, never overwritten,
and never edited in place. If a study split one clinical event across three CRF
rows because the severity changed twice, that is three canonical records,
because that is what the study wrote down.

**`CanonicalAEEpisode` — derived, additive, above the records.** An episode is
this system's view of the clinical event the records describe. It is computed
from the records and can be recomputed at any time; deleting every episode loses
nothing that cannot be rebuilt. Deleting a record loses the source.

The separation is the point. A system that merges records in place has thrown
away the study's own account of the event and cannot get it back. A system that
refuses to derive anything above the records makes every downstream question
re-implement episode assembly, differently each time.

Phenotype definitions operate on **episodes**, and the loader rejects a
definition that says otherwise.

---

## A blank is not a value

Every clinical field on a canonical record is a `Field[T]`, and every `Field`
carries a **collection state**. A missing value is not a fact about the patient
until something says which kind of missing it is:

| state | what it means |
| --- | --- |
| `collected` | the study asked, and this is the answer |
| `not_collected_by_protocol` | the CRF has no column for this at all |
| `not_applicable_gated` | a parent question was answered such that this one does not apply |
| `pending_ongoing` | the event has not ended; the answer does not exist yet |
| `intentionally_blank` | the column exists and the protocol instructs the site to leave it empty |
| `not_representable` | the study's codelist cannot express the concept the evidence supports |
| `unknown` | none of the above is established |

Only `collected` is evidence. `Field.is_evidence_of_absence` is true for exactly
one state, and a phenotype definition that tries to treat
`not_collected_by_protocol` or `not_applicable_gated` as absence is **rejected by
the loader**, not merely discouraged.

What each blank means is not guessed at read time. It is declared per study in
`config/collection_semantics.yaml` — which fields the CRF carries, which parent
questions gate which children, which codelists are restricted and what they
cannot express. A study with no declared semantics cannot be read at all: every
blank in it would be guesswork, and the loader says so.

### `not_representable` is the honest answer

Study 5 collects treatment action with a codelist of `none`,
`drug_withdrawn` and `drug_interrupted`. A narrative describing a dose reduction
has no permissible code. Substituting `drug_interrupted` would assert something
the evidence does not support. The field is left unresolved, marked
`not_representable`, and the note says which concept could not be expressed. The
weaker claim is the correct one.

### Verbatim and coded, both

Where a study coded a term, the record keeps **both** the verbatim term the site
wrote and the coded term the dictionary assigned, along with the dictionary
version that assigned it. Neither replaces the other. A definition can then ask
whether a coded term is a catalogue term for its concept *under that episode's
own dictionary version*, or opt into `bridge_dictionary_versions` to union across
versions — which is how you measure what bridging is worth, rather than assuming
it.

---

## Six renderings of one truth

The generator samples a clinical truth — what happened to a patient — and then
renders that same truth into six studies with different collection conventions:

| study | representation | what makes it different |
| --- | --- | --- |
| STUDY-01 | V-A | everything structured, values in mg/dL |
| STUDY-02 | V-B | one episode split across records on every severity change |
| STUDY-03 | V-C | clinical detail on a linked event form, brief narratives |
| STUDY-04 | V-D | minimal coding: no coded terms at all, narrative only |
| STUDY-05 | V-E | a restricted action codelist that cannot express a dose reduction |
| STUDY-06 | V-F | an earlier dictionary version with different coded terms |

The corpus is fully determined by its seed: the same seed produces byte-identical
files. Ground truth is written alongside as `truths.jsonl`, `gold_records.jsonl`
and `gold_episodes.jsonl`, and the gold answer key distinguishes what *happened*
from what a given rendering can *support* — a study that collects no coded terms
cannot reach an `explicit` state on coding, and the answer key says so.

---

## The deterministic and model paths, enforced in code

The deterministic path (`normalize/`) reads structured fields. The model path
(`extract/`) reads text. The boundary between them is not a convention:

```python
from aelayer.guards import assert_model_path_permitted, unresolved_fields
```

`guards.py` computes, for each record, the fields whose state is genuinely
unresolved, and refuses any model request that names a settled field or carries
anything but text. A test asserts that no already-controlled value is ever sent
to a model, over every record in the corpus. The guard is the reason the boundary
holds when someone adds a backend later.

**Abstention is a valid answer.** A backend that cannot support a value from the
text returns `value: null, collection_state: "unknown"` and says so. A guess is a
defect, and the harness scores abstention as its own outcome: how often the model
path correctly declined against how often it answered, and how often each was
right.

A value recovered from narrative does not retroactively make the CRF column
collected. `Field.prior_state` keeps the structured state, and the harness
compares against it.

---

## Episodes, and what happens when linkage is a judgement call

`EpisodeReconciler` assembles records into episodes under an ordered set of
rules, and records which rule decided each one:

1. `explicit_continuation` — the CRF itself says this record continues that one
2. `declared_convention` — the study declares that it splits on severity change
3. `temporal_overlap` — the records' date ranges overlap
4. `gap_within_tolerance` — a short gap, for a concept where recurrence is not expected
5. `recurrence_split` — a gap, for a concept where recurrence *is* expected, so they stay separate

`recurrence_expected` is declared per concept. Hypoglycemia recurs; anaemia,
within a study window, generally does not. The default merge rule is wrong for
exactly one of those, which is why the harness reports over-merge and over-split
separately and again split by whether recurrence is expected.

Where the rule that fired is a judgement call, the episode is **flagged, not
resolved**: `linkage_review_required` travels with it into every verdict, and a
definition decides what to do about it via `episode.on_review_required`. The
evaluation harness counts a case missed because the system declined a flagged
linkage separately from a case missed silently, because they are different
failures.

---

## Phenotype definitions

A definition is a YAML file with a version, a status and a content hash. Frozen
versions are never edited: a change to what qualifies as a case is a new version,
and the loader refuses to overwrite a frozen file.

```yaml
evidence_rules:                  # ordered; the first match assigns the state
  - id: explicit
    state: explicit
    when: {coded_term_matches_concept: true}
  - id: supported
    state: supported
    when:
      all:
        - lab: {test: GLUCOSE, op: "<", value: 70, unit: mg/dL}
        - symptoms: {min_count: 1, from: [neuroglycopenic, autonomic]}

missingness:
  treat_as_absent: []            # nothing is assumed absent
  route_to_review: [pending_ongoing, unknown, not_representable]
```

Three things the evaluator does that a naive one would not:

- **It distinguishes a failed test from an untestable one.** A glucose value of
  90 mg/dL fails the `supported` rule on the evidence. A study that never
  measured glucose fails it for want of evidence. The definition's `missingness`
  policy decides what happens to each, and the evaluator never assumes the second
  is the first.
- **It refuses to read a blank as absence unless the definition says so** — and
  the definition cannot say so for the two states where it would be a lie.
- **It carries linkage uncertainty forward** rather than counting a flagged
  episode as though the assembly were certain.

**Treatment action is deliberately absent from every rule.** Whether the site
reduced the dose is an attribute of the episode, not a criterion for it. Gating a
hypoglycemia case definition on it imports a field the clinical question never
referenced — and one that some studies cannot even express.

**Assertion matters where it exists.** A coded AE row asserts presence by
construction, and filtering it on assertion is theatre. A narrative can name
hypoglycemia precisely in order to record that it did *not* happen, and a
discovery search that ignores that returns documented absences as hits. So
assertion is a structured predicate on the discovery path and on narrative-
sourced fields, and it is not applied to coded rows as though it added
information there.

---

## Two ways in

**Patient-level access** has two paths that cannot be confused for each other.

`retrieve()` is the precise path: adjudicated episodes, filterable by verdict,
evidence state, representation, window, provenance and linkage-review status.
Its result answers `usable_as_cohort` with a real yes.

`discover()` is the discovery path: places in documents where a concept is
named. Every result is marked `candidate`, and calling `as_cohort()` on a
discovery result raises `CandidateInCohort` with an explanation. A mention is not
an episode, and no parameter turns one into the other; a candidate enters a
cohort through adjudication or a definition version that claims it.

**The program knowledge layer** is built from execution manifests. It is
forward-capture: it accrues from governed executions, it is empty on day one, and
its status message says exactly that rather than implying that history was
reconstructed. Historical work is added only by an explicit, one-at-a-time
`aelayer knowledge backfill --manifest <file>`.

Comparing two definition versions is **executed, not textual**: both run against
the same snapshot and the answer is the set of episodes each one claims, with the
deciding rule attached to every one that moved. A scope is mandatory — the
capability exists for evidence reuse within a stated question, and an unscoped
program-wide sweep is refused, because that is auditing colleagues' past choices
rather than reusing evidence.

---

## Traceability replaces the approval gate

There is no approval click anywhere in this system. Approving a specification you
cannot independently evaluate is ceremony: the reviewer sees a plan, not a
result, and clicking approve does not make the result checkable.

What makes a number checkable is that it can be followed back to source, after
the fact, by someone who was not in the room when it was compiled:

```
number -> analysis run -> cohort -> definition version -> episodes -> records -> spans
```

`aelayer trace <manifest-id>` walks that chain and prints it. If any hop cannot
be made, `complete` is false and `broken_at` names the level; the command exits
non-zero. A number that cannot be traced end to end is a failing test, not a
caveat.

Every execution writes a `Manifest` recording what was asked, what was compiled,
every version that produced the answer, and a **pointer** to where the result
lives. It never stores a copy of the result: a second copy is a second thing that
can drift. The manifest id is content-derived, so identical inputs produce an
identical id, and `aelayer replay <manifest-id>` re-executes a recorded run and
reports which input moved if it no longer reproduces — the data, the normalizer,
the extractor, or the definition.

---

## The agent

The agent compiles a question into an inspectable `PhenotypeQuerySpec` and runs
it through code that was written before the question was asked. It computes
nothing itself: every number comes from a registered service, and calling
anything else raises.

Where the question leaves a rule underdetermined, it stops and says which rule:

- **severity and seriousness together** — different fields; a severe event can be
  non-serious and a mild one serious, so a single count answers neither question
- **"severe hypoglycemia"** — an intensity grade, or the diabetes-trial term of
  art for an episode requiring third-party assistance
- **a window or threshold the definition does not use** — that is a change to the
  case definition, so it needs a new version, not a query parameter

In each case nothing is executed and no number is produced.

---

## Evaluation

`make eval` writes `reports/eval.md`. Three layers, a stress test, and a
reproducibility check, reported separately because they answer different
questions.

**Layer 1 — clinical validity.** Field-level precision, recall and F1 against
gold, broken out by collection state, by source path and by representation; a
collection-state confusion matrix; abstention quality; and provenance violations,
of which any number above zero is a defect.

**Layer 2 — episode reconciliation.** Boundary agreement against the true
episodes, with over-merge and over-split reported separately, and separately
again for concepts where recurrence is expected.

**Layer 3 — phenotype.** PPV and sensitivity against the gold case labels, with
misses attributable to declined linkage counted apart; plus transportability
across studies held out from rule development. Nothing here is fitted to data, so
that gap measures how much harder the held-out studies' conventions are, not
overfitting.

**Stress test — representation invariance.** For each sampled truth, does the
final classification agree across the six renderings? This is layered on top of
Layers 1–3, and the report states, in these words:

> Consistency across representations does not establish clinical validity. A
> pipeline can be perfectly consistent and consistently wrong.

**Reproducibility.** Identical inputs produce an identical manifest id and an
identical results hash, and a recorded run replays against a definition version
that has since been superseded.

### What it currently reports

On the shipped corpus (seed 7, six studies, 252 subjects, 337 source records,
306 episodes), `te_symptomatic_hypoglycemia` v1:

| | |
| --- | --- |
| collection-state accuracy | **1.000** over 4 718 field readings |
| abstention precision / answer precision | 1.000 / 1.000 over 563 model-path decisions |
| provenance violations | **0** |
| episode boundary agreement | 0.990 (1 over-merge, 1 over-split) |
| PPV / sensitivity | 1.000 / 0.890 |
| of 16 missed cases | 16 are episodes the system declined on flagged linkage, 0 are silent |
| sensitivity excluding declined linkage | 1.000 |
| transportability (development → held-out) | 0.889 → 0.890 sensitivity |
| representation invariance, verdict | 0.909 across six renderings |
| representation invariance, evidence state | 0.636 |
| discovery negation FP rate, filter on / off | 0.0000 / 0.1034 |
| reproducibility | manifest id, results hash and normalization all stable |

Three of these are worth reading carefully rather than admiring.

**The Layer 1 figures are bounded by the lexicon, not by the method.** The
extraction config covers the surface forms this generator writes, so the model
path answers 217 times without error and declines 346 times without error. That
says the baseline is not guessing where it lacks evidence — which is the property
being tested. It says nothing about recall on phrasings nobody wrote into
`config/extraction.yaml`, and on real narratives that is the number that would
move.

**Sensitivity is 0.890 because the system declined, not because it missed.**
Every one of the 16 misses is an episode whose linkage the reconciler flagged and
whose definition routes flagged linkage to review. Excluding those, sensitivity is
1.000. Whether declining is the right behaviour is a question about the
definition; the harness's job is to keep the two failures distinguishable.

**Invariance on evidence state is 0.636, and that is the finding.** Every
departure is STUDY-04 (V-D), which collects no coded terms at all: the same
clinical truth that reaches `explicit` in five renderings can only reach
`supported`, `insufficient` or nothing where nobody coded it. The pipeline is not
inconsistent — the renderings genuinely carry different evidence, and a system
that reported identical states across them would be inventing coding that does
not exist.

---

## Interfaces

```bash
aelayer generate --seed 7            # six renderings of one sampled truth
aelayer normalize                    # the deterministic path, by collection state
aelayer extract                      # normalize, enrich, reconcile, index
aelayer definitions --compare te_symptomatic_hypoglycemia:1:2 \
                    --scope "hypoglycemia incidence after escalation"
aelayer evaluate --version 1         # the case table, with the rule behind each row
aelayer retrieve HYPOGLYCEMIA --window 0:14          # precise path
aelayer retrieve HYPOGLYCEMIA --mode lexical         # discovery path
aelayer ask "how many subjects had symptomatic hypoglycemia?"
aelayer trace <manifest-id>          # the number, back to the text
aelayer replay <manifest-id>         # hash for hash
aelayer eval --report reports/eval.md
aelayer knowledge status
```

The HTTP API mirrors the CLI through the same pipeline object, so the two cannot
disagree about which version produced a number. `make serve` also serves a
single-page UI with five panels: source text beside the canonical record and its
collection states; episodes beside the records they were derived from; the
definition, a candidate builder that never mutates a frozen file, and the scoped
executed comparison; the two retrieval paths side by side with the assertion
filter on and off; and the agent, with the trace it produces.

The UI is API-driven, so opening `ui/index.html` from the filesystem shows an
empty shell. Run `make demo` once, then `make serve`, and open the address it
prints.

---

## Layout

```
config/
  concepts.yaml               concepts, lexicons, coded terms by dictionary version
  collection_semantics.yaml   what a blank means, per study, per field
  extraction.yaml             cues, patterns, anchors, confidences
  phenotypes/                 one file per definition version
data/synthetic/               generated tables, narratives, and the answer key
src/aelayer/
  models.py                   Field[T], CanonicalAERecord, CanonicalAEEpisode, ...
  semantics.py                the collection-semantics config, and gates/codelists
  guards.py                   the deterministic/model boundary, in code
  generate.py                 one truth, six renderings
  normalize/                  the deterministic path
  extract/                    the model path: text, concepts, assertion, temporal, values
  episode.py                  record -> episode reconciliation, with rules and flags
  phenotype/                  definition loading, validation, evaluation
  retrieval/                  precise and discovery paths over SQLite FTS5
  knowledge/                  the program knowledge layer and executed comparison
  agent/                      compile, the registered services, run, trace
  eval/                       the three layers, invariance, reproducibility, report
  runs.py                     manifests, the result store, replay
  pipeline.py                 one assembled path shared by CLI, API and agent
  api.py                      FastAPI
  cli.py                      Typer
ui/                           static single-page client
tests/
```

**Stack.** Python 3.11+, Pydantic v2, FastAPI, Typer, SQLite with FTS5, PyYAML,
pytest. No cloud services. Everything runs on a laptop with the network cable
pulled; the model path degrades to deterministic-only and says so.

---

## Out of scope

Regulatory signal detection and disproportionality analysis; genetics and omics
analysis, though `CaseAssignment` is shaped so a case-control export can be handed
to an existing pipeline — and subjects in the review set are exported with a null
status rather than coded either way; multi-user auth and RBAC; real terminology
licences, since coded terms in the config are illustrative placeholders; and model
training of any kind.
