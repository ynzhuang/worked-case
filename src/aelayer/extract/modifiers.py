"""Finding modifiers in language, and normalizing them to the catalogue.

A modifier is only useful if it belongs to the event in question. "Rash on the
chest, prior eczema on the back" contains two sites and one of them is not this
event's. So a mention is anchored: it must be reachable from a concept mention
in the same sentence, within a declared token distance, and not governed by a
disqualifying cue.

Recall here is exactly the surface forms declared in ``concepts.yaml``. That is a
property of the configuration, not of the method, and the README says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..catalog import AttributeCatalogue, ConceptCatalog, ExtractionConfig
from ..models import Span
from .text import Sentence, normalise, sentence_for, split_sentences, tokenize


@dataclass(frozen=True)
class ModifierHit:
    """One modifier mention, already normalized to a catalogue value."""

    attribute: str
    value: str
    surface: str
    start: int
    end: int
    confidence: float
    rule: str
    anchor_surface: str = ""

    def span(self, doc_id: str, field: str) -> Span:
        return Span(
            doc_id=doc_id, start=self.start, end=self.end, field=field,
            extracted_value=self.value, text=self.surface, kind="text",
        )


#: Words that carry no meaning between an event and its site.
_FILLERS = {"the", "a", "an", "his", "her", "their", "both"}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in phrase.split()]
    return re.compile(r"\b" + r"[\s\-]+".join(parts) + r"\b", re.IGNORECASE)


class ModifierExtractor:
    """The deterministic model-path backend.

    It is a lexicon and a scoping rule, not a trained model, and it is described
    as such everywhere it appears.
    """

    def __init__(self, catalog: ConceptCatalog, config: ExtractionConfig):
        self.catalog = catalog
        self.config = config
        self.attributes: dict[str, AttributeCatalogue] = {
            name: catalog.attribute(name)
            for name in config.extractable_attributes
            if name in catalog.attributes
        }
        modifiers = config.modifiers
        self.connectors = tuple(modifiers.get("connectors") or [])
        self.connector_set = set(self.connectors)
        self.disqualifiers = tuple(modifiers.get("disqualifiers") or [])
        self.max_distance = int(modifiers.get("max_distance_tokens", 8))
        self.quality_lexicon = config.quality_lexicon

        # Concept surface forms, so a modifier can be anchored to the event it
        # belongs to rather than to whatever else the sentence mentions.
        self.concept_forms: list[tuple[str, str]] = []
        for concept in catalog.concepts.values():
            for form in concept.lexicon:
                self.concept_forms.append((form, concept.concept_id))
        self.concept_forms.sort(key=lambda pair: -len(pair[0]))

    # -- anchors ------------------------------------------------------------

    def concept_mentions(self, text: str, concept_id: str | None = None):
        """Every concept mention in the text, longest form first."""
        found: list[tuple[int, int, str, str]] = []
        for form, cid in self.concept_forms:
            if concept_id and cid != concept_id:
                continue
            for match in _phrase_pattern(form).finditer(text):
                if any(
                    start <= match.start() and match.end() <= end
                    for start, end, _s, _c in found
                ):
                    continue
                found.append((match.start(), match.end(), match.group(0), cid))
        return sorted(found)

    # -- modifiers ----------------------------------------------------------

    def find(
        self, text: str, attribute: str, concept_id: str | None = None,
        source_kind: str = "reported_term",
    ) -> list[ModifierHit]:
        """Modifier mentions for one attribute, anchored to a concept mention."""
        catalogue = self.attributes.get(attribute)
        if catalogue is None:
            return []
        sentences = split_sentences(text)
        anchors = self.concept_mentions(text, concept_id)
        hits: list[ModifierHit] = []

        for phrase, value_id in sorted(
            catalogue.surface_index().items(), key=lambda kv: -len(kv[0])
        ):
            for match in _phrase_pattern(phrase).finditer(text):
                if any(h.start <= match.start() and match.end() <= h.end for h in hits):
                    continue
                scored = self._score(
                    text, sentences, anchors, match, attribute, value_id,
                    source_kind,
                )
                if scored is not None:
                    hits.append(scored)

        return sorted(hits, key=lambda h: (h.start, -h.confidence))

    def _score(
        self, text: str, sentences: list[Sentence], anchors, match: re.Match[str],
        attribute: str, value_id: str, source_kind: str = "reported_term",
    ) -> ModifierHit | None:
        """Decide whether this mention belongs to the event, and how sure."""
        sentence = sentence_for(sentences, match.start())
        if sentence is None:
            return None
        window = text[sentence.start:sentence.end].lower()
        in_scope = [a for a in anchors if sentence.contains(a[0])]
        if not in_scope:
            return None

        # A disqualifier between the anchor and the mention detaches them:
        # "rash resolved; history of eczema on the back" is not a truncal rash.
        before = text[: match.start()].lower()
        tail = before[max(0, len(before) - 60):]
        for cue in self.disqualifiers:
            if cue in tail:
                return None

        anchor = min(in_scope, key=lambda a: abs(a[0] - match.start()))
        # The words *between* the two, in whichever order they appear. Including
        # either mention would make the connector test match its own site.
        if match.start() >= anchor[1]:
            between = text[anchor[1]:match.start()]
        else:
            between = text[match.end():anchor[0]]
        gap_tokens = len(tokenize(between))
        if gap_tokens > self.max_distance:
            return None

        connector = normalise(between).strip()
        confidence_key = "same_sentence"
        rule = "same sentence as the event mention"
        # A direct connector means *only* a connector stands between the event
        # and the site. "rash on the chest and the back" does not connect the
        # rash to the back that way, and scoring it as though it did would make
        # a second site look as well supported as the first.
        core = [t for t in connector.split() if t not in _FILLERS]
        if " ".join(core) in self.connector_set:
            confidence_key, rule = "direct_connector", (
                f"joined to the event by {connector!r}"
            )
        elif gap_tokens <= 1:
            confidence_key, rule = "adjacent", "adjacent to the event mention"
        elif source_kind == "comment":
            # A comment is written about this record specifically, so a
            # same-sentence mention in one is better supported than the same
            # mention in general prose would be.
            confidence_key, rule = "comment_record", (
                "same sentence in a comment written about this record"
            )

        return ModifierHit(
            attribute=attribute,
            value=value_id,
            surface=match.group(0),
            start=match.start(),
            end=match.end(),
            confidence=self.config.confidence_for(confidence_key, 0.7),
            rule=rule,
            anchor_surface=anchor[2],
        )

    def best(
        self, text: str, attribute: str, concept_id: str | None = None,
        source_kind: str = "reported_term",
    ) -> ModifierHit | None:
        """The single best mention, or None.

        None is a real answer. Where the text does not support a value, the
        extractor abstains; a guess is a defect.
        """
        hits = self.find(text, attribute, concept_id, source_kind)
        if not hits:
            return None
        best = max(hits, key=lambda h: (h.confidence, -h.start))
        rivals = {h.value for h in hits if h.confidence >= best.confidence}
        if len(rivals) > 1:
            # Two equally supported and different values. Abstaining is the
            # correct answer: picking one would assert something the text does
            # not settle.
            return None
        return best

    def qualities(self, text: str) -> list[ModifierHit]:
        """Descriptors that are not in any catalogue value space.

        Not a phenotype criterion. They exist so the discovery path has
        something un-normalized to surface.
        """
        hits: list[ModifierHit] = []
        for value, forms in sorted(self.quality_lexicon.items()):
            for form in forms:
                for match in _phrase_pattern(form).finditer(text):
                    hits.append(ModifierHit(
                        attribute="quality", value=value, surface=match.group(0),
                        start=match.start(), end=match.end(),
                        confidence=self.config.confidence_for("extraction_default", 0.8),
                        rule="quality lexicon",
                    ))
        return sorted(hits, key=lambda h: h.start)
