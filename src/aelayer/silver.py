"""The silver-standard harness.

Some studies collect a modifier **twice**: once in a structured variable and
once, independently of that variable, in the investigator's own words. Where
both exist, the structured value is a comparator the extractor never sees — so
masking it gives a real evaluation set on data nobody hand-annotated.

The method, in four steps:

1. mask the structured variable from the extractor
2. run the model path over the reported term (or comment) alone
3. normalize both sides to the same value space
4. compare the extracted **assertion** against the masked structured assertion

Comparing assertions rather than values is the point. An extractor that turns
every documented "no mucosal involvement" into silence would score perfectly on
values — it never emits a wrong site — while destroying the denominator.

The output has three parts and none of them is optional:

**Agreement**, broken out by profile and by how the site writes.

**Calibration.** Confidence is only useful if it means something: a set of
predictions made at 0.8 should be right about 80% of the time. Reported as a
Brier score plus a reliability table, because a single accuracy number cannot
show a systematically overconfident extractor.

**An adjudication queue**, containing every disagreement, every low-confidence
prediction, *and a random sample of agreements*. The sample is the one teams
skip, and skipping it means never learning what the comparator itself gets
wrong.

Two caveats are printed verbatim wherever this is reported. They are not
hedging; they bound what the numbers can be used for.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .catalog import Configs
from .extract.backends import ExtractionRequest
from .extract.engine import ExtractionEngine
from .ingest import TrialStore
from .models import CanonicalAERecord
from .normalize.values import TRISTATE

#: Printed verbatim in every report that carries a silver number. The first
#: bounds what agreement *means*; the second bounds who it generalizes to.
SILVER_CAVEATS: tuple[str, ...] = (
    "CAVEAT 1 - THE TWO ROUTES ARE NOT INDEPENDENT. The structured qualifier "
    "and the narrative were produced by the same investigator, at the same "
    "visit, on the same form, often in the same minute. They share every "
    "upstream error: a clinician who did not examine the mucosa records "
    "nothing in the qualifier and writes nothing in the term. Agreement "
    "between them is therefore an UPPER BOUND on agreement with an independent "
    "adjudicator, not an estimate of it. This is a silver standard, not ground "
    "truth, and the comparator has its own error rate.",
    "CAVEAT 2 - THE EVALUATION SET IS NOT A RANDOM SUBSET. Only studies that "
    "collect the modifier BOTH structurally and in prose can be scored at all. "
    "Those studies are, by construction, the ones with the more thorough "
    "collection conventions and the more detailed narratives. Performance "
    "measured here does not transfer to a study that keeps the modifier only "
    "in free text, which is precisely the study the layer is meant to help. "
    "See the transportability holdout for what that costs.",
)


@dataclass
class Comparison:
    """One record where both routes could speak."""

    source_record_id: str
    study_id: str
    profile: str
    modifier: str
    structured_assertion: str | None
    structured_value: str | None
    structured_variable: str | None
    extracted_assertion: str | None
    extracted_value: str | None
    extracted_confidence: float | None
    text: str
    span_text: str
    reported_term_style: str
    agreement: str            # agree | disagree | abstained
    note: str = ""

    @property
    def answered(self) -> bool:
        return self.extracted_assertion is not None

    @property
    def correct(self) -> bool:
        return self.agreement == "agree"


def _brier(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """Mean squared error between stated confidence and the outcome.

    0 is perfect, 0.25 is what you get by always saying 0.5, and a confident
    extractor that is often wrong scores worse than an unconfident one.
    """
    if not pairs:
        return None
    return round(
        sum((confidence - (1.0 if hit else 0.0)) ** 2 for confidence, hit in pairs)
        / len(pairs),
        4,
    )


@dataclass
class SilverReport:
    modifier: str
    profiles: list[str]
    comparisons: list[Comparison] = _dc_field(default_factory=list)
    caveats: tuple[str, ...] = SILVER_CAVEATS

    # -- populations --------------------------------------------------------

    @property
    def eligible(self) -> int:
        return len(self.comparisons)

    @property
    def answered(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.answered]

    @property
    def agreements(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "agree"]

    @property
    def disagreements(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "disagree"]

    @property
    def abstentions(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "abstained"]

    # -- headline numbers ---------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        """Precision, recall, F1, coverage, abstention, agreement.

        Precision is over the answers the extractor gave; recall is over the
        records where the structured comparator said something. Coverage and
        abstention sit beside them because a precision of 1.0 reached by
        answering three times in a hundred is not a useful extractor, and the
        pair of numbers is what makes that visible.
        """
        total = self.eligible
        answered = len(self.answered)
        correct = len(self.agreements)
        with_comparator = sum(
            1 for c in self.comparisons if c.structured_assertion is not None
        )
        precision = correct / answered if answered else 0.0
        recall = correct / with_comparator if with_comparator else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        return {
            "eligible_records": total,
            "with_comparator": with_comparator,
            "answered": answered,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "coverage": round(answered / total, 4) if total else 0.0,
            "abstention_rate": round(len(self.abstentions) / total, 4) if total else 0.0,
            "agreements": correct,
            "disagreements": len(self.disagreements),
        }

    def by_assertion(self) -> dict[str, dict[str, Any]]:
        """Agreement per assertion class the comparator recorded.

        Broken out because the classes fail differently, and the one that
        matters most for the denominator — a documented `absent` — is also the
        rarest and the easiest to lose in an average.
        """
        rows: dict[str, dict[str, Any]] = {}
        for assertion in ("present", "absent", "uncertain"):
            subset = [
                c for c in self.comparisons if c.structured_assertion == assertion
            ]
            if not subset:
                continue
            answered = [c for c in subset if c.answered]
            correct = [c for c in subset if c.correct]
            rows[assertion] = {
                "n": len(subset),
                "answered": len(answered),
                "correct": len(correct),
                "recall": round(len(correct) / len(subset), 4),
                "precision": (
                    round(len(correct) / len(answered), 4) if answered else 0.0
                ),
            }
        return rows

    # -- calibration --------------------------------------------------------

    def calibration(self, bins: int = 5) -> dict[str, Any]:
        """Does a stated confidence of 0.8 mean right eight times in ten?

        A reliability table plus a Brier score. Reported because an extractor
        whose confidence carries no information cannot be thresholded, and
        every phenotype definition in this repository thresholds on it.
        """
        pairs = [
            (c.extracted_confidence or 0.0, c.correct)
            for c in self.answered
        ]
        table: list[dict[str, Any]] = []
        for index in range(bins):
            low = index / bins
            high = (index + 1) / bins
            in_bin = [
                (confidence, hit) for confidence, hit in pairs
                if (low <= confidence < high) or (index == bins - 1 and confidence == 1.0)
            ]
            if not in_bin:
                continue
            observed = sum(1 for _c, hit in in_bin if hit) / len(in_bin)
            mean_confidence = sum(c for c, _hit in in_bin) / len(in_bin)
            table.append({
                "bin": f"[{low:.1f}, {high:.1f}{']' if index == bins - 1 else ')'}",
                "n": len(in_bin),
                "mean_confidence": round(mean_confidence, 4),
                "observed_accuracy": round(observed, 4),
                "gap": round(mean_confidence - observed, 4),
            })
        gaps = [abs(row["gap"]) * row["n"] for row in table]
        total = sum(row["n"] for row in table)
        return {
            "brier_score": _brier(pairs),
            "expected_calibration_error": (
                round(sum(gaps) / total, 4) if total else None
            ),
            "reliability": table,
            "note": (
                "A positive gap means the extractor was more confident than it "
                "was right, which is the direction that matters: a phenotype "
                "definition thresholds on this number, so overconfidence turns "
                "into cases nobody can defend."
            ),
        }

    # -- breakdowns ---------------------------------------------------------

    def by(self, key: str) -> dict[str, dict[str, Any]]:
        """The same metrics, broken out by profile or by reported-term style."""
        groups: dict[str, list[Comparison]] = {}
        for comparison in self.comparisons:
            groups.setdefault(getattr(comparison, key), []).append(comparison)
        return {
            name: SilverReport(self.modifier, [name], rows).metrics()
            for name, rows in sorted(groups.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "modifier": self.modifier,
            "profiles": self.profiles,
            "standard": "silver",
            "caveats": list(self.caveats),
            "overall": self.metrics(),
            "by_assertion": self.by_assertion(),
            "by_profile": self.by("profile"),
            "by_reported_term_style": self.by("reported_term_style"),
            "calibration": self.calibration(),
        }

    # -- adjudication -------------------------------------------------------

    def adjudication_queue(
        self, seed: int = 7, agreement_sample: int = 20,
        low_confidence_below: float = 0.8,
    ) -> list[dict[str, Any]]:
        """What a clinician should look at, and why each row is there.

        Three populations, deliberately: disagreements, low-confidence
        predictions, and a random sample of agreements. The third is the one
        teams skip, and skipping it means never learning what the comparator
        itself gets wrong — which, given caveat 1, is the number that decides
        how much any of this is worth.
        """
        rng = random.Random(seed)
        queue: list[dict[str, Any]] = []

        def entry(comparison: Comparison, reason: str) -> dict[str, Any]:
            return {
                **asdict(comparison),
                "queue_reason": reason,
                "standard": "silver",
            }

        for comparison in self.disagreements:
            queue.append(entry(
                comparison,
                "the extractor and the structured qualifier disagree; either "
                "could be the one that is wrong",
            ))
        for comparison in self.answered:
            confidence = comparison.extracted_confidence or 0.0
            if comparison.agreement != "disagree" and confidence < low_confidence_below:
                queue.append(entry(
                    comparison,
                    f"the extraction was accepted at confidence {confidence:.2f}, "
                    f"below the review threshold {low_confidence_below:.2f}",
                ))
        agreements = list(self.agreements)
        rng.shuffle(agreements)
        seen = {q["source_record_id"] for q in queue}
        sampled = 0
        for comparison in agreements:
            if sampled >= agreement_sample:
                break
            if comparison.source_record_id in seen:
                continue
            queue.append(entry(
                comparison,
                "sampled agreement: included so the silver standard's own error "
                "rate can be estimated rather than assumed",
            ))
            sampled += 1
        return queue

    def write_adjudication(
        self, path: str | Path, seed: int = 7, agreement_sample: int = 20,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.adjudication_queue(seed, agreement_sample)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return target


class SilverHarness:
    """Build a silver standard for one modifier over the eligible profiles."""

    def __init__(self, configs: Configs, store: TrialStore, engine: ExtractionEngine):
        self.configs = configs
        self.store = store
        self.engine = engine

    def eligible_profiles(self, modifier: str) -> list[str]:
        """Profiles carrying the modifier both structurally and in text."""
        return self.configs.profiles.evaluation_profiles(modifier)

    def run(
        self, records: Sequence[CanonicalAERecord],
        modifier: str = "mucosal_involvement",
        profiles: Iterable[str] | None = None,
    ) -> SilverReport:
        wanted = list(profiles or self.eligible_profiles(modifier))
        report = SilverReport(modifier=modifier, profiles=wanted)

        for record in records:
            if record.profile not in wanted:
                continue
            profile = self.configs.profiles.profile(record.profile)
            if not profile.carries_both(modifier):
                continue
            comparison = self._compare(record, profile, modifier)
            if comparison is not None:
                report.comparisons.append(comparison)
        return report

    def _compare(self, record, profile, modifier: str) -> Comparison | None:
        """One record: mask the structured value, extract, compare."""
        structured_assertion, structured_value = self._masked_comparator(
            record, profile, modifier
        )
        text_home = profile.text_home(modifier)
        if text_home is None:
            return None

        # Steps 1 and 2: the extractor sees the text and nothing else. The
        # structured value is not in the request, so there is no route by which
        # it could leak into the answer.
        doc_id, text, source_kind, variable = self._text_source(record, text_home)
        if not text:
            return None
        request = ExtractionRequest(
            doc_id=doc_id, text=text, modifiers=(modifier,),
            concept_id=None, source_kind=source_kind, source_variable=variable,
        )
        result = self.engine.backend.extract(request)
        extracted = result.values.get(modifier)

        # Steps 3 and 4. Both sides are already in the same space, because the
        # normalizer and the backend each normalize before returning.
        if extracted is None:
            agreement = "abstained"
        elif structured_assertion is None:
            agreement = "disagree"
        elif extracted.assertion != structured_assertion:
            agreement = "disagree"
        elif (
            structured_value is not None
            and extracted.value is not None
            and extracted.value != structured_value
        ):
            # Same call about whether the modifier was there, different site.
            agreement = "disagree"
        else:
            agreement = "agree"

        home = profile.structured_home(modifier)
        return Comparison(
            source_record_id=record.source_record_id,
            study_id=record.study_id,
            profile=record.profile,
            modifier=modifier,
            structured_assertion=structured_assertion,
            structured_value=structured_value,
            structured_variable=home.variable if home else None,
            extracted_assertion=extracted.assertion if extracted else None,
            extracted_value=extracted.value if extracted else None,
            extracted_confidence=extracted.confidence if extracted else None,
            text=text,
            span_text=(
                extracted.evidence[0].text if extracted and extracted.evidence else ""
            ),
            reported_term_style=profile.reported_term_style,
            agreement=agreement,
            note=(extracted.note if extracted else "the extractor abstained"),
        )

    def _masked_comparator(
        self, record: CanonicalAERecord, profile, modifier: str
    ) -> tuple[str | None, str | None]:
        """The comparator, read from the source row rather than the record.

        Read here and nowhere else, so that the extraction call above cannot
        see it. On a record where the model path already filled the modifier
        from text, the structured value is still what the *study* recorded, and
        that is what a silver standard compares against.
        """
        home = profile.structured_home(modifier)
        if home is None or not home.variable:
            return None, None
        raw = self._structured_raw(record, home)
        if raw in (None, ""):
            # The comparator itself is silent. Nothing to score against, and
            # scoring it as a negative would manufacture agreement out of a
            # blank cell.
            return None, None
        return TRISTATE.get(str(raw).strip().lower()), None

    def _structured_raw(self, record: CanonicalAERecord, home) -> Any:
        if home.kind == "linked_form" and home.variable and "." in home.variable:
            _domain, testcd = home.variable.split(".", 1)
            for row in self.store.linked_form_rows(record.source_record_id):
                if str(row.get("SCTESTCD")) == testcd:
                    return row.get("SCORRES")
            return None
        row = next(
            (r for r in self.store.rows("ae")
             if str(r.get("AESPID")) == record.source_record_id),
            None,
        )
        return None if row is None else row.get(home.variable)

    def _text_source(self, record: CanonicalAERecord, home) -> tuple[str, str, str, str]:
        if home.kind == "comment":
            documents = self.store.documents_of(record.source_record_id)
            if documents:
                return (
                    documents[0].doc_id, documents[0].full_text, "comment",
                    "CO.COVAL",
                )
            return ("", "", "comment", "CO.COVAL")
        return (
            f"AE:{record.source_record_id}:AETERM",
            str(record.reported_term.value or ""),
            "reported_term",
            "AETERM",
        )
