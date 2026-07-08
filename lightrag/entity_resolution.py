"""Deterministic core for OceanStack inline entity resolution.

This module decides, for each newly extracted entity name in a batch,
whether it should become a brand-new node (``CREATE_NEW``), be discarded in
favor of an existing node under a different spelling (``DISCARD_AND_REUSE``),
or replace an existing node's canonical name (``PROMOTE``). Every gate here
is deterministic: string residue comparison, vector-store similarity bands,
and graph node-liveness checks. There is no reasoner/LLM call yet - any
entity whose best candidate lands in the "candidate similarity" band (or
whose candidates carry no extractable similarity score at all) is deferred
with ``method="reasoner_band_deferred"`` and resolved as ``CREATE_NEW``. A
later step wires the reasoner tie-breaker for that band; ``PROMOTE`` is also
produced by that later step and is not emitted here.

Deliberately imports only ``lightrag.base`` and ``lightrag.utils`` (plus
stdlib) - no backend implementations and no ``lightrag.operate`` import, to
avoid import cycles with the module that will call into this one.
"""

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lightrag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from lightrag.utils import logger


class Decision(str, Enum):
    """Outcome of resolving one newly extracted entity against the graph."""

    CREATE_NEW = "create_new"
    DISCARD_AND_REUSE = "discard_and_reuse"
    PROMOTE = "promote"


@dataclass
class Candidate:
    """One vector-store hit considered as a match for an extracted entity.

    Populated fully by the (later) reasoner step; the deterministic core
    only needs ``name`` and ``similarity``.
    """

    name: str
    similarity: float | None
    degree: int
    description: str
    shared_neighbors: list[str]


@dataclass
class ResolutionDecision:
    """Record of how one extracted entity name was resolved."""

    decision: Decision
    extracted_name: str
    target_name: str | None
    similarity: float | None
    confidence: float
    rationale: str
    method: str


@dataclass
class PromotePlan:
    """Plan to rename an existing node onto a newly extracted canonical name.

    Not produced by the deterministic core in this step; ``PROMOTE``
    decisions and their plans are wired once the reasoner is in place.
    """

    old_name: str
    new_name: str
    old_node_data: dict
    old_edges: list[tuple[str, str, dict]]


@dataclass
class BatchResolution:
    """Aggregate result of resolving one extraction batch."""

    name_map: dict[str, str]
    promote_plans: list[PromotePlan]
    records: list[ResolutionDecision]
    llm_calls: int


def _namespace(name: str) -> str:
    """Return the dotted namespace prefix of an entity name, or "" if none."""
    return name.rsplit(".", 1)[0] if "." in name else ""


def _extract_similarity(hit: dict) -> float | None:
    """Pull a similarity score out of a vector-store hit, backend-agnostic.

    The PG ``entities`` vdb query returns no similarity score today (only
    ``entity_name``/``created_at``), so this returns ``None`` for those hits
    until a later step adds the score column. Other backends surface it
    under one of "similarity", "score", "distance", or "__metrics__".
    """
    if "similarity" in hit:
        return float(hit["similarity"])
    if "score" in hit:
        return float(hit["score"])
    if "distance" in hit:
        return 1.0 - float(hit["distance"])
    if "__metrics__" in hit:
        return float(hit["__metrics__"])
    return None


def _drift_residue(name: str) -> str:
    """Casefold and strip everything but [a-z0-9.] - namespaces stay distinct."""
    return re.sub(r"[^a-z0-9.]", "", name.casefold())


def _residue_no_dots(name: str) -> str:
    """Casefold and strip everything but [a-z0-9] for auto-merge residue equality."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


async def _resolve_one(
    name: str,
    node_items: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    auto_threshold: float,
    candidate_threshold: float,
    top_k: int,
) -> ResolutionDecision:
    """Run the deterministic gates for a single extracted entity name."""
    # 1. Exact match short-circuit: the extracted name is already a live node.
    if await knowledge_graph_inst.get_node(name) is not None:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=None,
            confidence=1.0,
            rationale="entity name already exists as a live node; existing merge handles it",
            method="exact",
        )

    # 2. Build the vdb query text from the name plus its description(s).
    descriptions = [
        str(item["description"]) for item in node_items if item.get("description")
    ]
    query_text = (name + "\n" + "\n".join(descriptions))[:512]

    # 3. Query the vector store for candidates.
    hits = await entity_vdb.query(query_text, top_k=top_k)
    if not hits:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=None,
            confidence=1.0,
            rationale="no vdb candidates returned",
            method="no_candidates",
        )

    # 4. Namespace guard: only compare within the same dotted namespace.
    own_namespace = _namespace(name)
    candidates = [
        hit
        for hit in hits
        if hit.get("entity_name") != name
        and _namespace(str(hit.get("entity_name", ""))) == own_namespace
    ]
    if not candidates:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=None,
            confidence=1.0,
            rationale="all vdb candidates fall outside this entity's namespace",
            method="namespace_guard",
        )

    # 5. Score candidates and route into a similarity band.
    scored = [(str(hit["entity_name"]), _extract_similarity(hit)) for hit in candidates]
    has_none = any(similarity is None for _, similarity in scored)
    scored_non_none: list[tuple[str, float]] = [
        (cand_name, sim) for cand_name, sim in scored if sim is not None
    ]
    if scored_non_none:
        best_name, best_similarity = max(scored_non_none, key=lambda pair: pair[1])
    else:
        best_name, best_similarity = scored[0][0], None

    own_residue = _residue_no_dots(name)
    if (
        best_similarity is not None
        and best_similarity >= auto_threshold
        and _residue_no_dots(best_name) == own_residue
    ):
        if await knowledge_graph_inst.get_node(best_name) is None:
            return ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=best_similarity,
                confidence=1.0,
                rationale=f"top candidate {best_name!r} scored in vdb but is not a live node",
                method="stale_vdb",
            )
        return ResolutionDecision(
            decision=Decision.DISCARD_AND_REUSE,
            extracted_name=name,
            target_name=best_name,
            similarity=best_similarity,
            confidence=1.0,
            rationale=f"auto-merge: residue-equal to live node {best_name!r} at similarity {best_similarity:.4f}",
            method="auto_threshold",
        )

    if has_none or (
        best_similarity is not None and best_similarity >= candidate_threshold
    ):
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=1.0,
            rationale=(
                f"top candidate {best_name!r} (similarity={best_similarity}) falls in the "
                "reasoner band; reasoner tie-break not wired yet, deferring to CREATE_NEW"
            ),
            method="reasoner_band_deferred",
        )

    return ResolutionDecision(
        decision=Decision.CREATE_NEW,
        extracted_name=name,
        target_name=None,
        similarity=best_similarity,
        confidence=1.0,
        rationale=f"best candidate {best_name!r} similarity {best_similarity} below candidate threshold",
        method="below_threshold",
    )


async def resolve_batch(
    nodes_by_name: dict[str, list[dict]],
    edges_by_pair: dict[tuple[str, str], list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> BatchResolution:
    """Resolve one extraction batch's entity names against the live graph.

    Deterministic gates only - no LLM/reasoner calls in this step
    (``llm_response_cache`` is accepted for signature compatibility with the
    later reasoner-enabled step and is otherwise unused here). Every
    per-entity failure is caught and downgraded to ``CREATE_NEW`` so a
    single bad hit can never abort ingest.
    """
    del edges_by_pair  # not consulted by the deterministic gates themselves
    del llm_response_cache  # reserved for the reasoner step, unused here

    auto_threshold = float(global_config["entity_resolution_auto_merge_similarity"])
    candidate_threshold = float(global_config["entity_resolution_candidate_similarity"])
    top_k = int(global_config.get("entity_resolution_top_k", 5))

    name_map: dict[str, str] = {}
    promote_plans: list[PromotePlan] = []
    records: list[ResolutionDecision] = []
    llm_calls = 0

    for name in sorted(nodes_by_name):
        try:
            record = await _resolve_one(
                name,
                nodes_by_name[name],
                knowledge_graph_inst,
                entity_vdb,
                auto_threshold,
                candidate_threshold,
                top_k,
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe: never abort ingest
            logger.warning(
                "entity_resolution: failed resolving %r, defaulting to CREATE_NEW: %s",
                name,
                exc,
            )
            record = ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=None,
                confidence=0.0,
                rationale=f"resolution raised {type(exc).__name__}: {exc}",
                method="error_fallback",
            )

        records.append(record)
        if record.decision == Decision.DISCARD_AND_REUSE and record.target_name:
            name_map[name] = record.target_name
        else:
            name_map[name] = name

    return BatchResolution(
        name_map=name_map,
        promote_plans=promote_plans,
        records=records,
        llm_calls=llm_calls,
    )


def apply_name_map(
    name_map: dict[str, str],
    nodes_by_name: dict[str, list[dict]],
    edges_by_pair: dict[tuple[str, str], list[dict]],
) -> tuple[dict[str, list[dict]], dict[tuple[str, str], list[dict]]]:
    """Rewrite extraction-batch keys and item fields through a resolved name map.

    Pure function: returns brand-new dicts and never mutates the inputs.
    Nodes or edges that collapse onto the same target after remapping have
    their item lists merged (extended), and self-loop edges created by the
    remapping are dropped.
    """
    new_nodes: dict[str, list[dict]] = {}
    for extracted_name, items in nodes_by_name.items():
        target_name = name_map.get(extracted_name, extracted_name)
        rewritten_items = []
        for item in items:
            new_item = dict(item)
            if "entity_name" in new_item:
                new_item["entity_name"] = target_name
            if "entity_id" in new_item:
                new_item["entity_id"] = target_name
            rewritten_items.append(new_item)
        new_nodes.setdefault(target_name, []).extend(rewritten_items)

    new_edges: dict[tuple[str, str], list[dict]] = {}
    for (src, tgt), items in edges_by_pair.items():
        new_src = name_map.get(src, src)
        new_tgt = name_map.get(tgt, tgt)
        if new_src == new_tgt:
            logger.debug(
                "entity_resolution: dropping self-loop edge (%r, %r) -> %r after name-map rewrite",
                src,
                tgt,
                new_src,
            )
            continue

        target_key = (new_src, new_tgt) if new_src <= new_tgt else (new_tgt, new_src)
        rewritten_items = []
        for item in items:
            new_item = dict(item)
            for field_name in ("src_id", "tgt_id", "source", "target"):
                if field_name in new_item:
                    new_item[field_name] = name_map.get(
                        new_item[field_name], new_item[field_name]
                    )
            rewritten_items.append(new_item)
        new_edges.setdefault(target_key, []).extend(rewritten_items)

    return new_nodes, new_edges


def write_resolution_log(
    records: list[ResolutionDecision], global_config: dict
) -> None:
    """Append one JSONL line per resolution decision for audit/replay.

    Never raises: a logging failure must never take down ingest.
    """
    try:
        working_dir = Path(global_config.get("working_dir", "."))
        log_path = working_dir / "entity_resolution_log.jsonl"
        dry_run = bool(global_config.get("entity_resolution_dry_run", False))
        workspace = global_config.get("workspace", "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with log_path.open("a", encoding="utf-8") as fh:
            for record in records:
                line = {
                    "ts": ts,
                    "workspace": workspace,
                    "dry_run": dry_run,
                    "decision": record.decision.value,
                    "method": record.method,
                    "extracted_name": record.extracted_name,
                    "target_name": record.target_name,
                    "similarity": record.similarity,
                    "confidence": record.confidence,
                    "rationale": record.rationale,
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception as exc:  # noqa: BLE001 - logging must never break ingest
        logger.warning("entity_resolution: failed to write resolution log: %s", exc)
