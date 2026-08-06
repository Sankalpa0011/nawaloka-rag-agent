"""Symptom -> specialty lookup, backed by LlamaIndex over Qdrant.

Pairs live in data/ontology/symptoms.yaml, validated against the specialties
table on load, and embedded into their own Qdrant collection for semantic
matching. Falls back to substring matching over the YAML when Qdrant is down.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, model_validator

from app.config import DATA_DIR, settings
from app.logging_conf import get_logger

log = get_logger(__name__)

ONTOLOGY_PATH = DATA_DIR / "ontology" / "symptoms.yaml"

# Calibrated thresholds, tunable via ONTOLOGY_* env vars (app/config.py).

# Minimum similarity for a symptom match to count at all (used by match_specialties).
MATCH_FLOOR = settings.ontology_match_floor

# Keep a second specialty only if its score is close behind the top one.
RELATIVE_MARGIN = settings.ontology_relative_margin

# Higher bar used by looks_like_complaint, which guards the doctor-name field.
COMPLAINT_FLOOR = settings.ontology_complaint_floor

# How many symptom nodes to retrieve per query.
TOP_K = settings.ontology_top_k

# Minimum gap between the top specialty and the best node of a different
# specialty, needed to call a match "decisive" rather than noise.
DECISIVE_MARGIN = settings.ontology_decisive_margin


class SymptomMapping(BaseModel):
    specialty: str = Field(min_length=2)
    #: How patients describe the problem. Embedded and matched semantically.
    symptoms: list[str] = Field(default_factory=list)
    #: What patients call the practitioner. Matched literally, never embedded.
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_something(self) -> SymptomMapping:
        # An entry needs at least one of symptoms or aliases to do anything.
        if not self.symptoms and not self.aliases:
            raise ValueError(
                f"{self.specialty!r} declares neither symptoms nor aliases"
            )
        return self


@lru_cache(maxsize=1)
def load_ontology() -> list[SymptomMapping]:
    """Read the YAML and check every specialty referenced actually exists."""
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    mappings = [SymptomMapping(**entry) for entry in raw]

    try:
        from app.tools.db import fetch

        rows, _ = fetch("SELECT name FROM specialties")
        known = {str(r["name"]) for r in rows}
    except Exception as exc:
        # No DB at import time (unit tests, offline tooling) - skip the check.
        log.warning("ontology.validation_skipped", error=str(exc)[:160])
        return mappings

    unknown = sorted({m.specialty for m in mappings} - known)
    if unknown:
        raise ValueError(
            f"symptoms.yaml references specialties that are not in the database: "
            f"{unknown}. Either the table was renamed or the file has a typo - "
            f"known specialties are {sorted(known)}"
        )

    _check_alias_collisions(mappings)

    log.info("ontology.loaded", specialties=len(mappings),
             symptoms=sum(len(m.symptoms) for m in mappings),
             aliases=sum(len(m.aliases) for m in mappings))
    return mappings


def _check_alias_collisions(mappings: list[SymptomMapping]) -> None:
    """Fail loudly if one alias is claimed by more than one specialty."""
    owner: dict[str, str] = {}
    clashes: list[str] = []
    for mapping in mappings:
        for alias in mapping.aliases:
            key = alias.strip().lower()
            if key in owner and owner[key] != mapping.specialty:
                clashes.append(f"{alias!r} claimed by {owner[key]!r} and {mapping.specialty!r}")
            else:
                owner[key] = mapping.specialty
    if clashes:
        raise ValueError("symptoms.yaml has ambiguous aliases: " + "; ".join(clashes))


@lru_cache(maxsize=1)
def specialty_aliases() -> dict[str, tuple[str, ...]]:
    """Canonical specialty name -> what patients call its practitioners."""
    return {m.specialty: tuple(m.aliases) for m in load_ontology() if m.aliases}


def _nodes():
    from llama_index.core.schema import TextNode

    # Symptoms only - aliases name the practitioner, not the problem.
    nodes = []
    for mapping in load_ontology():
        for symptom in mapping.symptoms:
            nodes.append(
                TextNode(
                    text=symptom,
                    metadata={"specialty": mapping.specialty, "symptom": symptom},
                    excluded_embed_metadata_keys=["specialty", "symptom"],
                )
            )
    return nodes


@lru_cache(maxsize=1)
def _index():
    """Build (or attach to) the ontology collection."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    from app.retrieval.embedder import llama_embedding
    from app.retrieval.qdrant_store import get_client

    client = get_client()
    collection = settings.qdrant_ontology_collection
    embed_model = llama_embedding()

    store = QdrantVectorStore(client=client, collection_name=collection)
    storage = StorageContext.from_defaults(vector_store=store)

    existing = 0
    if client.collection_exists(collection):
        existing = client.count(collection).count

    if existing:
        return VectorStoreIndex.from_vector_store(store, embed_model=embed_model)

    log.info("ontology.indexing", collection=collection)
    return VectorStoreIndex(
        _nodes(), storage_context=storage, embed_model=embed_model, show_progress=False
    )


@lru_cache(maxsize=1)
def _retriever():
    return _index().as_retriever(similarity_top_k=settings.ontology_top_k)


def _fallback_match(text: str) -> list[str]:
    """Substring match over the YAML, for when Qdrant is unreachable."""
    lowered = (text or "").lower()
    hits: list[str] = []
    for mapping in load_ontology():
        for symptom in mapping.symptoms:
            if symptom.lower() in lowered or lowered in symptom.lower():
                if mapping.specialty not in hits:
                    hits.append(mapping.specialty)
                break
    return hits


@dataclass(frozen=True)
class OntologyMatch:
    """What the ontology made of a phrase, with enough detail to act on it."""

    specialties: list[str]
    top_score: float = 0.0
    #: Gap from the winning node to the best node of a different specialty.
    #: None when every node retrieved belongs to one specialty.
    cross_gap: float | None = None
    source: str = "semantic"        # "semantic" | "substring" | "none"

    @property
    def decisive(self) -> bool:
        """Whether this phrase is something the ontology genuinely recognises,
        rather than just having a nearest specialty."""
        if not self.specialties:
            return False
        if self.cross_gap is None:
            return True
        return self.cross_gap >= settings.ontology_decisive_margin


def match_complaint(complaint: str) -> OntologyMatch:
    """Match a phrase against the symptom ontology, with the match quality.
    Use `.decisive` rather than truthiness before showing the result to a patient.
    """
    text = (complaint or "").strip()
    if not text:
        return OntologyMatch([], source="none")

    try:
        results = _retriever().retrieve(text)
    except Exception as exc:
        log.warning("ontology.retrieval_unavailable", error=str(exc)[:160])
        hits = _fallback_match(text)
        # A substring hit counts as decisive only when it lands on one specialty.
        return OntologyMatch(
            hits,
            top_score=1.0 if hits else 0.0,
            cross_gap=None if len(hits) == 1 else 0.0,
            source="substring",
        )

    if not results or results[0].score is None:
        return OntologyMatch([], source="none")

    best = results[0].score
    if best < settings.ontology_match_floor:
        # Nothing stood out - not a complaint we recognise.
        return OntologyMatch([], top_score=best)

    top_specialty = results[0].metadata.get("specialty")
    cross_gap = next(
        (best - n.score for n in results[1:]
         if n.score is not None and n.metadata.get("specialty") != top_specialty),
        None,
    )

    specialties: list[str] = []
    for node in results:
        if node.score is None or node.score < best - settings.ontology_relative_margin:
            continue
        specialty = node.metadata.get("specialty")
        if specialty and specialty not in specialties:
            specialties.append(specialty)
    return OntologyMatch(specialties, top_score=best, cross_gap=cross_gap)


def match_specialties(complaint: str) -> list[str]:
    """Specialties that normally see this complaint, best first. Empty unless decisive."""
    match = match_complaint(complaint)
    return match.specialties if match.decisive else []


def _is_known_doctor(value: str) -> bool:
    """Does this match a doctor actually in the database (via the gazetteer)?"""
    try:
        from app.agent.signals import build_gazetteer

        terms = build_gazetteer().doctor_terms
    except Exception as exc:
        log.warning("ontology.gazetteer_unavailable", error=str(exc)[:160])
        return False

    lowered = value.lower().strip()
    if lowered in terms:
        return True
    for term in terms:
        if len(term) >= settings.ontology_doctor_term_min_len and term in lowered:
            return True
    return False


def looks_like_complaint(text: str) -> bool:
    """Is this describing a problem rather than naming a person?
    Guards the doctor-name field, so it errs towards "not a complaint".
    """
    value = (text or "").strip()
    if not value:
        return False

    # A name in the database is a name, whatever it scores against symptoms.
    if _is_known_doctor(value):
        return False

    try:
        results = _retriever().retrieve(value)
    except Exception as exc:
        log.warning("ontology.retrieval_unavailable", error=str(exc)[:160])
        return bool(_fallback_match(value))

    if not results:
        return False
    top = results[0]
    return top.score is not None and top.score >= settings.ontology_complaint_floor


def seed() -> int:
    """Build the collection if it's empty. Called from ensure_seeded."""
    _index()
    from app.retrieval.qdrant_store import get_client

    return get_client().count(settings.qdrant_ontology_collection).count


def reset_cache() -> None:
    load_ontology.cache_clear()
    specialty_aliases.cache_clear()
    _index.cache_clear()
    _retriever.cache_clear()
