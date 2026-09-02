"""The silver-standard harness.

Some studies collect an attribute **twice**: once in a structured variable and
once, independently, in the investigator's own words. Where both exist, the
structured value is a comparator the extractor never sees — so masking it gives
a genuine evaluation set on data nobody hand-annotated.

The method, in four steps:

1. mask the structured variable from the extractor
2. run the model path over the reported term (or comment) alone
3. normalize both sides to the concept catalogue
4. compare the extracted value against the masked structured value

It is called a **silver** standard, not ground truth, everywhere it is reported.
The structured field has its own error rate — a site can mistype a qualifier as
easily as it can write an ambiguous phrase — and a metric that calls it truth
would be overstating what it knows.

The output includes an **adjudication queue**: every disagreement, every
low-confidence prediction, *and a random sample of agreements*. The sample is
not optional. Without it you only ever inspect failures, and you can never
estimate the silver standard's own error rate.
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

SILVER_CAVEAT = (
    "This is a silver standard, not ground truth. The comparator is the "
    "study's own structured qualifier, which has its own error rate: a site can "
    "mistype a coded qualifier as easily as it can write an ambiguous phrase. "
    "Disagreement therefore means the two disagree, not that the extractor is "
    "wrong, and the adjudication queue exists so that the difference can be "
    "measured rather than assumed."
)


@dataclass
class Comparison:
    """One record where both routes could speak."""

    source_record_id: str
    study_id: str
    profile: str
    attribute: str
    structured_value: str | None
    structured_variable: str | None
    extracted_value: str | None
    extracted_confidence: float | None
    text: str
    span_text: str
    reported_term_style: str
    agreement: str            # agree | disagree | abstained
    note: str = ""

    @property
    def produced_a_value(self) -> bool:
        return self.extracted_value is not None


@dataclass
class SilverReport:
    attribute: str
    profiles: list[str]
    comparisons: list[Comparison] = _dc_field(default_factory=list)
    caveat: str = SILVER_CAVEAT

    # -- headline numbers ---------------------------------------------------

    @property
    def eligible(self) -> int:
        return len(self.comparisons)

    @property
    def answered(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.produced_a_value]

    @property
    def agreements(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "agree"]

    @property
    def disagreements(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "disagree"]

    @property
    def abstentions(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.agreement == "abstained"]

    def metrics(self) -> dict[str, Any]:
        """Precision, recall, F1, coverage, abstention, normalized agreement.

        Precision is over the values the extractor produced; recall is over the
        records where the structured comparator had a value. Coverage and
        abstention are reported alongside because a high precision reached by
        answering three times out of a hundred is not a useful extractor, and
        the pair of numbers is what makes that visible.
        """
        total = self.eligible
        answered = len(self.answered)
        correct = len(self.agreements)
        with_comparator = sum(
            1 for c in self.comparisons if c.structured_value is not None
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
            "normalized_agreement": round(correct / answered, 4) if answered else 0.0,
            "agreements": correct,
            "disagreements": len(self.disagreements),
        }

    def by(self, key: str) -> dict[str, dict[str, Any]]:
        """The same metrics, broken out by profile or by reported-term style."""
        groups: dict[str, list[Comparison]] = {}
        for comparison in self.comparisons:
            groups.setdefault(getattr(comparison, key), []).append(comparison)
        return {
            name: SilverReport(self.attribute, [name], rows).metrics()
            for name, rows in sorted(groups.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "profiles": self.profiles,
            "standard": "silver",
            "caveat": self.caveat,
            "overall": self.metrics(),
            "by_profile": self.by("profile"),
            "by_reported_term_style": self.by("reported_term_style"),
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
        itself gets wrong.
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
        for comparison in agreements:
            if len([q for q in queue if q["queue_reason"].startswith("sampled")]) \
                    >= agreement_sample:
                break
            if comparison.source_record_id in seen:
                continue
            queue.append(entry(
                comparison,
                "sampled agreement: included so the silver standard's own error "
                "rate can be estimated rather than assumed",
            ))
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
    """Build a silver standard for one attribute over the eligible profiles."""

    def __init__(self, configs: Configs, store: TrialStore, engine: ExtractionEngine):
        self.configs = configs
        self.store = store
        self.engine = engine

    def eligible_profiles(self, attribute: str) -> list[str]:
        """Profiles carrying the attribute both structurally and in text."""
        return self.configs.profiles.evaluation_profiles(attribute)

    def run(
        self, records: Sequence[CanonicalAERecord], attribute: str = "location",
        profiles: Iterable[str] | None = None,
    ) -> SilverReport:
        wanted = list(profiles or self.eligible_profiles(attribute))
        report = SilverReport(attribute=attribute, profiles=wanted)

        for record in records:
            if record.profile not in wanted:
                continue
            profile = self.configs.profiles.profile(record.profile)
            if not profile.carries_both(attribute):
                continue
            comparison = self._compare(record, profile, attribute)
            if comparison is not None:
                report.comparisons.append(comparison)
        return report

    def _compare(self, record, profile, attribute: str) -> Comparison | None:
        """One record: mask the structured value, extract, compare."""
        structured = self._masked_structured_value(record, profile, attribute)
        text_home = profile.text_home(attribute)
        if text_home is None:
            return None

        # Step 1 and 2: the extractor sees the text and nothing else. The
        # structured value is not in the request, so there is no route by which
        # it could leak into the answer.
        doc_id, text, source_kind, variable = self._text_source(record, text_home)
        if not text:
            return None
        request = ExtractionRequest(
            doc_id=doc_id, text=text, attributes=(attribute,),
            concept_id=record.standardized_concept,
            source_kind=source_kind, source_variable=variable,
        )
        result = self.engine.backend.extract(request)
        extracted = result.values.get(attribute)

        # Steps 3 and 4: both sides are already catalogue values, because the
        # normalizer and the backend each normalize before returning.
        if extracted is None:
            agreement = "abstained"
        elif structured is None:
            agreement = "disagree"
        else:
            agreement = "agree" if extracted.value == structured else "disagree"

        return Comparison(
            source_record_id=record.source_record_id,
            study_id=record.study_id,
            profile=record.profile,
            attribute=attribute,
            structured_value=structured,
            structured_variable=(
                profile.structured_home(attribute).variable
                if profile.structured_home(attribute) else None
            ),
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

    def _masked_structured_value(
        self, record: CanonicalAERecord, profile, attribute: str
    ) -> str | None:
        """The comparator, read from the source row rather than the record.

        Read here and nowhere else, so that the extraction call above cannot
        see it. On a record where the model path already filled the attribute
        from text, the structured value is still what the *study* recorded, and
        that is what a silver standard compares against.
        """
        home = profile.structured_home(attribute)
        if home is None or not home.variable:
            return None
        row = next(
            (r for r in self.store.rows("ae")
             if str(r.get("AESPID")) == record.source_record_id),
            None,
        )
        if row is None:
            return None
        raw = row.get(home.variable)
        if raw in (None, ""):
            return None
        catalogue = self.configs.catalog.attribute(attribute)
        text = str(raw).strip()
        return text if text in catalogue.values else catalogue.normalize(text)

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
