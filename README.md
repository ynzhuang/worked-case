# Adverse event evidence layer

A working prototype of a clinical evidence layer over completed-trial adverse
event data.

The worked example is a **cutaneous event with mucosal involvement within 30
days of first exposure**. That question is hard for one specific reason: the
modifier — *mucosal involvement* — is recorded in a different place by every
study. A structured qualifier here, a linked form there, the investigator's own
words somewhere else, a comment record, or nowhere at all. This layer reads all
five into one provenance-bearing shape and runs one versioned phenotype
definition across them.

Two things in this build are the point of it:

```bash
make ablation   # is reading narrative text worth it? Ends in a DECISION.
make silver     # what the extraction is worth, with both caveats printed.
```

If only those two work, the prototype has done its job.

```bash
make demo       # the whole path end to end, offline, ~10s
make eval       # every harness -> reports/evaluation.md
make serve      # the API and the single-page UI on http://127.0.0.1:8000/
make test       # 299 tests, 89% statement coverage
```

---

## Read this first

**All data is synthetic.** The repository generates its own corpus. No real
patient data, and nothing derived from real patient data, is present anywhere
here. Every table row carries a `SYNTHETIC` column and every narrative carries a
synthetic header.

**The extraction backend is a configurable rule and lexicon baseline.** By
default it is a deterministic system of dictionaries and cue-scoping rules
driven entirely by `config/`. It is not a trained clinical NLP model and must
not be described as one. Its recall is exactly the surface forms somebody wrote
into the catalogue — a property of the configuration, not of the method. An LLM
backend can be swapped in behind the same interface; the manifest records which
one ran, and with the network disconnected the model path degrades to the rules
baseline and says so in its notes.

**Coded terms in the configuration are illustrative placeholders.** This
repository holds no terminology licence and ships no licensed dictionary
content. The terms in `config/concepts.yaml` are stand-ins with the right
*shape* — several legitimate codings of one clinical situation, a code renamed
between dictionary versions, and a code that disappears from the target version
— so that concept-set membership and version reconciliation can be exercised.
Replace them with a licensed extract before any real use.

**The metrics measure signal recovery, not clinical performance.** Gold labels
are the generator's own intent, so the numbers measure whether the pipeline
recovers a signal that was deliberately planted. They are not an estimate of
performance on real clinical text.

**Model training is out of scope.** Nothing here is fitted to data. The
transportability gap therefore measures how much harder the held-out collection
conventions are, not overfitting.

---

## The one idea

Two fields, never merged:

| field | question it answers | values |
|---|---|---|
| `assertion` | what did the source **say**? | `present` · `absent` · `uncertain` |
| `availability` | did the source say **anything**? | `observed` · `not_collected` · `not_applicable` · `pending` · `unresolved` |

A record with `assertion=absent` is a patient somebody examined and found clear.
That is a **`non_case`**, and they belong in the denominator.

A record with `availability=not_collected` is silence. Nobody looked. That
patient belongs in **neither** the numerator nor the denominator.

Collapse those two into one "missing" flag and every rate the system reports is
wrong by however many patients were never examined. The model refuses to
represent the merged state at all:

```python
Attribute[str](availability="not_collected", assertion="absent")
# ValidationError: availability 'not_collected' means the source is silent, so
# it cannot assert 'absent'; assertion and availability are orthogonal and must
# never be merged
```

There is a test that walks every combination of the two fields and asserts each
one is explicitly allowed or explicitly refused.

---

## Four verdicts

| verdict | meaning | denominator? |
|---|---|---|
| `case` | every criterion satisfied | numerator and denominator |
| `non_case` | a criterion was **evaluated** and failed | denominator |
| `review` | the evidence exists but does not settle it | neither |
| `not_ascertainable` | the evidence was never collected in a form this study can answer from | neither |

Without an explicit `non_case` you cannot state a denominator. Incidence is
computed **within the ascertainable population**, and the **ascertainable
fraction** is printed beside every rate as a study characteristic:

```
study       total  case   non   rev   n/a   asc.f  incidence
STUDY-A        32    11    11     4     6   0.688        0.5
STUDY-B        33    14     5     2    12   0.576     0.7368
STUDY-C        32     0     0     0    32   0.000       None   <- collects it nowhere
STUDY-D        32    11    12     6     3   0.719     0.4783
STUDY-E        32    12    10     4     6   0.688     0.5455
STUDY-F        32    11    11     6     4   0.688        0.5
STUDY-G        32    11     6     4    11   0.531     0.6471
```

STUDY-C is the one to get right. Qualifying events, qualifying timing, and a
phenotype that still cannot be evaluated — because the modifier was never
collected. That is `not_ascertainable`, not a run of negatives.

---

## Seven collection profiles

One clinical truth, rendered seven ways. The profile is a **declared collection
decision**, never inferred: a study whose profile is not declared cannot be read
at all, because every silence in it would be guesswork.

| profile | study | modifier lives in | what it demonstrates |
|---|---|---|---|
| `P_structured` | STUDY-A | `AEMUCOS` | the direct route; no model touches it |
| `P_text` | STUDY-B | `AETERM` | survives only in the investigator's own words |
| `P_absent` | STUDY-C | nowhere | `not_ascertainable`, not a `non_case` |
| `P_both` | STUDY-D | `AEMUCOS` **and** `AETERM` | the silver-standard evaluation set |
| `P_negated` | STUDY-E | `AETERM`, stated either way | **an observed negative, told from silence** |
| `P_version` | STUDY-F | `SC.MUCOSAL` | coded under a superseded dictionary version |
| `P_concept_variant` | STUDY-G | `CO.COVAL` | a different legitimate coding of the same situation |

`P_negated` matters most. It is the only way to prove the system distinguishes
an observed negative from silence.

---

## Three normalizations, and only one of them is a model

Three different things get called "normalization". Conflating them is how a
system ends up with a model quietly rewriting coded fields. `guards.py` enforces
the separation; a request naming either of the last two raises rather than
degrading.

**5.1 · Language variation.** The same clinical fact written a dozen ways in
prose. *This is the only mechanism a model is used for.* Cue-scoped assertion
classification over declared lexicons, with a span required for every value.

```
'rash with oral mucosal involvement'          -> present, ORAL,  0.95
'rash without mucosal involvement'            -> absent,  —,     0.90
'rash; mucosal involvement cannot be excluded'-> uncertain, —,   0.55
'rash with involvement of the wet surfaces'   -> (abstained)
```

Abstention is a valid answer and is measured as a rate. A guess is a defect.

**5.2 · Coded-concept variation.** Several legitimate codings of one clinical
situation — `Rash`, `Rash erythematous`, `Rash maculopapular`. Resolved by the
phenotype's **concept set**. Nothing merges them and nothing overwrites either
one.

**5.3 · Terminology-version variation.** The same concept coded under different
dictionary versions. Reconciled by a **mechanical 1:1 map** to a declared target
version. The split is reported, always:

```
dictionary version reconciliation (mechanical, never a model):
  unchanged                  213
  remapped_mechanically       30    Rash erythematous -> Erythematous rash
  flagged_for_review           9    Rash maculopapular: no code under D-21.0
```

A code that does not persist is **flagged for review, never auto-recoded**. The
original code and its source version are preserved on every record; the
reconciliation sits beside them, never on top of them.

---

## Cross-domain derivation

`exposure_relation` exists in no single field. It is `AE.AESTDTC` minus the
anchor exposure date from `EX`, computed by governed code and stamped
`method="derived"`, `source_variable="AE+EX"`:

```
AE.AESTDTC 2022-04-19 - EX.first_exposure 2022-04-09 = 10 days
```

Governed computation, never model reasoning, and a reader can tell the
difference from the record alone.

---

## §8 · The silver-standard harness

STUDY-D records mucosal involvement **twice** — in a structured qualifier and,
independently of that qualifier, in the investigator's own words. Masking the
structured value gives a real evaluation set on data nobody hand-annotated.

What is compared is the **assertion**, not just the value. An extractor that
turned every documented "no" into silence would score perfectly on values while
destroying the denominator.

```
                      n  answered  correct  recall  precision
present              15        12       12   0.800      1.000
absent                8         2        2   0.250      1.000
uncertain             6         6        6   1.000      1.000

Brier score              0.0724
expected calib. error    0.209
bin             n  mean conf  observed     gap
[0.4, 0.6)      6      0.550     1.000  -0.450
[0.6, 0.8)      4      0.780     1.000  -0.220
[0.8, 1.0]     10      0.940     1.000  -0.060
```

Calibration is reported because every phenotype definition here thresholds on
confidence, so an overconfident extractor turns into cases nobody can defend.

The **adjudication queue** contains every disagreement, every low-confidence
prediction, *and a random sample of agreements*. The sample is not optional:
without it you only ever inspect failures and can never estimate the
comparator's own error rate.

### The two caveats, printed verbatim wherever a silver number appears

> **CAVEAT 1 — THE TWO ROUTES ARE NOT INDEPENDENT.** The structured qualifier
> and the narrative were produced by the same investigator, at the same visit,
> on the same form, often in the same minute. They share every upstream error: a
> clinician who did not examine the mucosa records nothing in the qualifier and
> writes nothing in the term. Agreement between them is therefore an **upper
> bound** on agreement with an independent adjudicator, not an estimate of it.
> This is a silver standard, not ground truth, and the comparator has its own
> error rate.

> **CAVEAT 2 — THE EVALUATION SET IS NOT A RANDOM SUBSET.** Only studies that
> collect the modifier **both** structurally and in prose can be scored at all.
> Those studies are, by construction, the ones with the more thorough collection
> conventions and the more detailed narratives. Performance measured here does
> not transfer to a study that keeps the modifier only in free text, which is
> precisely the study the layer is meant to help.

---

## §9 · The value ablation

The experiment that can falsify the proposal. Three cumulative stages, each a
distinct engineering investment:

| stage | what it reads |
|---|---|
| `structured` | coded concepts, structured qualifiers, linked forms, cross-domain derivation. **No model runs at all.** |
| `+ reported_term` | the model path reads `AETERM` where the structured route left the modifier unresolved |
| `+ comments` | the model path also reads linked comment records |

```
stage             eval   asc  asc.f  case   ok  bad   prec  recall
structured         225    67  0.298    33   33    0  1.000   1.000
reported_term      225   108  0.480    59   59    0  1.000   0.967
comments           225   125  0.556    70   70    0  1.000   0.946

structured -> reported_term
  - 26 correctly ascertained cases added (>= the declared floor of 10)
  - a 78.8% relative gain over the 33 correct cases the previous stage found
  - precision on the added cases is 100.0%; 0 of the added cases are wrong

DECISION: ADOPT. Stage 'reported_term' is worth building: it adds 26 correctly
ascertained cases the previous stage could not reach, at 100% precision on
those additions.
```

**The output states a decision, not just numbers.** And "correctly" is
load-bearing: a stage that finds forty more cases of which thirty are wrong has
made the cohort worse while making the headline bigger, so precision on the
*added* cases is a veto rather than a tiebreak. The materiality thresholds live
in `ablation.py` and were written before the numbers were seen.

The machinery can say no. Run it against `cutaneous_mucosal.v1`, which accepts
only structured evidence:

```
DECISION: DO NOT ADOPT. Stage 'reported_term' changed no verdict at all.
Whatever it recovered was already recorded structurally, or the records fail
another criterion regardless.
```

---

## Transportability

Whole studies are held out, **never rows**. A random row split would leak every
profile's collection conventions into both sides and measure nothing about
protocol shift, which is how this layer actually fails when it meets a new
study. Row splits are disallowed, not merely discouraged.

```
                        development     held out
profiles          P_both, P_structured, P_text | P_absent, P_concept_variant,
                                                 P_negated, P_version
records                          97          128
PPV                             1.0          1.0
sensitivity                     1.0       0.8947
not-ascertainable rate       0.2165       0.4141
```

---

## Phenotype definitions

A definition is a versioned scientific artifact with a content hash. Frozen
versions are never edited: a change to what qualifies as a case is a **new
version**, not an edit, and the loader refuses to overwrite a frozen file.

```yaml
concept_set:
  include: [RASH, RASH_ERYTHEMATOUS, RASH_MACULOPAPULAR]
  dictionary_target: "D-21.0"

modifiers:
  - name: mucosal_involvement
    require_assertion: present
    accept_methods: [direct, extracted]      # <- the demonstration
    on_unavailable: not_ascertainable

temporal:
  anchor: first_exposure
  window: {min: 0, max: 30, unit: days}
```

`accept_methods: [direct, extracted]` **is** the demonstration: the rule does
not know or care which study field supplied the evidence, only that it is there
and points at something a reader can check. Nothing in the definition names
`AEMUCOS`, `SC.MUCOSAL`, `AETERM` or `CO.COVAL` — there is a test asserting so.

Three definitions ship. `cutaneous_mucosal.v1` accepts structured evidence only;
`v2` supersedes it and also accepts evidence read out of prose. The executed
difference between them is what the ablation measures.

`graded_toxicity.v1` is structurally different — a grade threshold and a
cumulative-exposure threshold, no modifier requirement at all — and it loads and
runs with **no code changes**.

---

## The agent

The agent chooses *which* rule to run. It never decides what a case is, never
computes a number, and never resolves an ambiguity by picking a default.

**It binds a definition version.** The compiled specification names an id, a
version and a content hash. It carries no window, no threshold and no assertion
of its own.

**It raises a conflict rather than overriding.** A question implying different
parameters does not become a query option:

```
$ aelayer ask "cutaneous events without mucosal involvement"

NOT RUN — the question conflicts with the definition it names,
or leaves a rule underdetermined. Nothing was computed.

  conflict   The question asks about subjects who did not have
             mucosal_involvement, and that phrase has two readings this system
             keeps deliberately apart.
  bound to   cutaneous_mucosal.v2
  effect     A documented negative means somebody examined the patient and
             recorded an absence; that subject is a non_case and belongs in the
             denominator. Silence means nobody recorded anything, and that
             subject belongs in neither the numerator nor the denominator. The
             two readings give different rates, and one of them is not a rate
             at all.
  options:
    - Subjects with an observed absence of mucosal_involvement
      (assertion=absent) — evaluated negatives
    - Subjects for whom mucosal_involvement was never recorded
      (availability=not_collected or unresolved) — not ascertainable
    - Both, as separate counts with the ascertainable fraction
```

**It screens studies on metadata before any patient-level query.** A study that
records the modifier nowhere cannot answer, and finding that out by scanning its
patients first is both slower and worse manners:

```
supportability for 'mucosal_involvement', from declared collection metadata
  no patient record was read to produce this

  STUDY-A  P_structured       supported
  STUDY-B  P_text             supported_via_extraction
  STUDY-C  P_absent           cannot_ascertain
  ...
```

**The tool surface is typed, permissioned, and small.** Eight tools, each with an
input schema, an output schema and a permission. Calls are validated both ways.
**No SQL surface. No tool writes to a source record.**

**There is no approval gate.** Approving a specification you cannot
independently evaluate is ceremony. What makes a number checkable is the trace
returned with it:

```
result     case count = 70
  analysis   d94ef9fbe0c27e89
    cohort     70 source record(s) with verdict 'case' across 6 studies
      definition cutaneous_mucosal.v2 (hash a97e91231505bd4a)
        attribute  mucosal_involvement is present via direct from AEMUCOS
        attribute  onset is 19 days after first_exposure, inside [0, 30] days
          record     STUDY-A:STUDY-A-BG-001-AE-01
            span       AE:STUDY-A-BG-001-AE-01:0-9  AEMUCOS = 'Y'
```

A number that cannot be traced end to end is a failing test, not a caveat.

---

## Grain

The **source record** is the grain. Verdicts, denominators, the silver standard
and the ablation all run over it, because it is the thing the study actually
collected and the only grain every claim traces back to.

Episode grouping is **demoted** to a derived view. It carries no attributes of
its own and nothing is evaluated at that grain — promotion would mean choosing
between two records that disagree, and that choice would sit underneath every
downstream number while being invisible in it.

---

## Retrieval

Two paths, and no parameter turns one into the other.

**Precise.** Cohort membership is decided by normalized values and a
definition's verdict. No embedding is consulted. Assertion and availability are
separate predicates, so "documented negatives" and "never asked" cannot be
requested as one population.

**Discovery.** Free-text search over mentions. Everything it returns is a
`candidate`; calling `as_cohort()` on it raises, because a mention is a place in
a document where something was named, not an adjudicated event.

---

## Reproducibility

Every execution writes a `Manifest` recording what was asked, what was compiled,
which versions of everything produced the answer, and a **pointer** to where the
result lives. It never stores the result payload: copying outputs into the
registry would create a second, uncontrolled result store with its own drift
problem.

```bash
aelayer replay d94ef9fbe0c27e89
# run d94ef9fbe0c27e89 reproduced exactly (results hash ea97499ef51bdb21…)
```

The manifest id is content-derived, so identical inputs yield an identical id.
Each input is checked separately on replay, so a failure says *which* one moved:
the data, the deterministic config, the extraction config, or the definition.

---

## Layout

```
config/
  concepts.yaml               concepts, dictionary versions, modifier catalogues
  study_profiles.yaml         where each modifier lives, per study
  extraction.yaml             readable sources, cue lists, confidence keys
  phenotypes/*.yaml           versioned definitions, frozen
src/aelayer/
  models.py                   Attribute[T], the two-field invariant, verdicts
  guards.py                   the deterministic/model boundary, in code
  catalog.py                  concepts, modifiers, version reconciliation
  profiles.py                 collection conventions and supportability
  generate.py                 the synthetic corpus and its answer key
  normalize/                  the deterministic path
  extract/                    the model path: mentions, assertions, backends
  phenotype/                  definitions and the four-verdict evaluator
  silver.py                   §8 — the silver-standard harness
  ablation.py                 §9 — the value ablation and its decision
  eval/                       harnesses, transportability, the markdown report
  retrieval/                  the precise path and the discovery path
  agent/                      compile, bind, refuse, execute, trace
  runs.py                     manifests, result store, replay
  pipeline.py  api.py  cli.py
ui/                           the single-page UI
tests/                        299 tests
```

---

## Out of scope

Real terminology licences. Model training. Real patient data of any kind. A
graph store. Anything that requires a network connection.
