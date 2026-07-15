"""Deterministic core for OceanStack inline entity resolution.

This module decides, for each newly extracted entity name in a batch,
whether it should become a brand-new node (``CREATE_NEW``), be discarded in
favor of an existing node under a different spelling (``DISCARD_AND_REUSE``),
or replace an existing node's canonical name (``PROMOTE``). Every gate here
is deterministic: string residue comparison, vector-store similarity bands,
and graph node-liveness checks. Entities whose best candidate lands in the
"candidate similarity" band (or whose candidates carry no extractable
similarity score at all) are handed to a relationship-aware LLM reasoner
that returns ``DISCARD_AND_REUSE``/``CREATE_NEW``/``PROMOTE``; the reasoner
is optional (``entity_resolution_use_reasoner``) and capped per batch, and
every failure mode fails safe to ``CREATE_NEW``. ``PROMOTE`` renames an
existing node onto the extracted name (executed by ``apply_promotions``) and
is gated behind ``entity_resolution_allow_promote`` (default off); when that
flag is off a reasoner ``PROMOTE`` downgrades to ``DISCARD_AND_REUSE``.

Deliberately imports only ``lightrag.base``, ``lightrag.prompt`` and
``lightrag.utils`` (plus stdlib) - no backend implementations and no
``lightrag.operate`` import, to avoid import cycles with the module that
will call into this one.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lightrag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    get_llm_cache_identity,
    logger,
    use_llm_func_with_cache,
)

# Character budget for the vector-store candidate query. Characters, not
# tokens. BGE-M3 accepts 8192 tokens, so the previous 512-character cap threw
# away most of a multi-description entity's signal before it ever reached the
# vector store, which silently degraded candidate recall.
_VDB_QUERY_CHAR_BUDGET = 2000

# Deterministic gates only read the graph and the vector store, so they are
# safe to run concurrently. Bounded so a large batch cannot stampede either.
_DEFAULT_GATE_CONCURRENCY = 8


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

    Captured by ``_capture_promote_plan`` when the reasoner returns a
    ``PROMOTE`` and ``entity_resolution_allow_promote`` is set; executed by
    ``apply_promotions``. ``old_node_data``/``old_edges`` are the reversal
    record written to the promote-undo log before any destructive step.
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

    The PG ``entities`` vdb query now projects ``similarity`` (see
    postgres_impl.py), so PG hits carry a score; backends that do not project
    one yield ``None`` here and route through the reasoner band. Recognised
    keys: "similarity", "score", "distance" (1 - d), "__metrics__".
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


def _is_suffix_variant(a: str, b: str) -> bool:
    """True when a and b are suffix/version variants that must stay distinct.

    Enforces the hard rule (e.g. "OceanStack" vs "OceanStack-core", "Model" vs
    "Model-v2") deterministically instead of trusting the reasoner prompt: on
    the no-dot residues, differing residues where one is a strict prefix of the
    other are treated as distinct variants and never merged. Equal residues are
    NOT a conflict (they are the same name up to punctuation/case, handled by
    the auto-merge path); the dotted-namespace case is handled separately.
    """
    ra, rb = _residue_no_dots(a), _residue_no_dots(b)
    if not ra or not rb or ra == rb:
        return False
    return ra.startswith(rb) or rb.startswith(ra)


def _residue_no_dots(name: str) -> str:
    """Casefold and strip everything but [a-z0-9] for auto-merge residue equality."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _select_reasoner_llm_func_and_role(global_config: dict):
    """Pick the reasoner LLM callable and the role name it was taken from.

    LightRAG 1.5.3+ exposes role-scoped functions under ``role_llm_funcs`` and
    its raw ``llm_model_func`` requires a ``hashing_kv`` kwarg that the cache
    wrapper does not forward (KeyError at call time). 1.4.x exposes a
    self-contained ``llm_model_func``. Prefer a role func, fall back to
    ``llm_model_func``, so the reasoner works on both.

    The role is returned so the caller can partition the LLM cache under the
    identity of the model actually used. It is ``None`` when the raw
    ``llm_model_func`` fallback is taken, which carries no role identity.
    """
    roles = global_config.get("role_llm_funcs")
    if isinstance(roles, dict):
        for role in ("extract", "query", "keyword"):
            func = roles.get(role)
            if func is not None:
                return func, role
    return global_config.get("llm_model_func"), None


def _select_reasoner_llm_func(global_config: dict):
    """Pick the LLM callable for the reasoner across LightRAG versions."""
    return _select_reasoner_llm_func_and_role(global_config)[0]


async def _resolve_one(
    name: str,
    node_items: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    auto_threshold: float,
    candidate_threshold: float,
    top_k: int,
) -> tuple[ResolutionDecision, list[tuple[str, float | None]]]:
    """Run the deterministic gates for a single extracted entity name.

    Returns the decision plus the in-band scored candidate list. That list is
    non-empty only when the decision is deferred to the reasoner
    (``method="reasoner_band_deferred"``); every terminal deterministic path
    returns an empty candidate list.
    """
    # 1. Exact match short-circuit: the extracted name is already a live node.
    if await knowledge_graph_inst.get_node(name) is not None:
        return (
            ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=None,
                confidence=1.0,
                rationale="entity name already exists as a live node; existing merge handles it",
                method="exact",
            ),
            [],
        )

    # 2. Build the vdb query text from the name plus its description(s).
    descriptions = [
        str(item["description"]) for item in node_items if item.get("description")
    ]
    query_text = (name + "\n" + "\n".join(descriptions))[:_VDB_QUERY_CHAR_BUDGET]

    # 3. Query the vector store for candidates.
    hits = await entity_vdb.query(query_text, top_k=top_k)
    if not hits:
        return (
            ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=None,
                confidence=1.0,
                rationale="no vdb candidates returned",
                method="no_candidates",
            ),
            [],
        )

    # 4. Namespace guard: only compare within the same dotted namespace.
    own_namespace = _namespace(name)
    ns_candidates = [
        hit
        for hit in hits
        if hit.get("entity_name") != name
        and _namespace(str(hit.get("entity_name", ""))) == own_namespace
    ]
    if not ns_candidates:
        return (
            ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=None,
                confidence=1.0,
                rationale="all vdb candidates fall outside this entity's namespace",
                method="namespace_guard",
            ),
            [],
        )

    # 4b. Suffix/version-variant guard (HARD rule): a candidate that is a
    # suffix/version variant of the extracted name (OceanStack vs
    # OceanStack-core) must never merge - drop it deterministically so it can
    # never reach the reasoner or the auto path.
    candidates = [
        hit
        for hit in ns_candidates
        if not _is_suffix_variant(name, str(hit.get("entity_name", "")))
    ]
    if not candidates:
        return (
            ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=None,
                confidence=1.0,
                rationale="all in-namespace candidates are suffix/version variants; kept distinct",
                method="variant_guard",
            ),
            [],
        )
    # 5. Score candidates and route into a similarity band.
    scored = [
        (str(hit.get("entity_name", "")), _extract_similarity(hit))
        for hit in candidates
    ]
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
            return (
                ResolutionDecision(
                    decision=Decision.CREATE_NEW,
                    extracted_name=name,
                    target_name=None,
                    similarity=best_similarity,
                    confidence=1.0,
                    rationale=f"top candidate {best_name!r} scored in vdb but is not a live node",
                    method="stale_vdb",
                ),
                [],
            )
        return (
            ResolutionDecision(
                decision=Decision.DISCARD_AND_REUSE,
                extracted_name=name,
                target_name=best_name,
                similarity=best_similarity,
                confidence=1.0,
                rationale=f"auto-merge: residue-equal to live node {best_name!r} at similarity {best_similarity:.4f}",
                method="auto_threshold",
            ),
            [],
        )

    if has_none or (
        best_similarity is not None and best_similarity >= candidate_threshold
    ):
        return (
            ResolutionDecision(
                decision=Decision.CREATE_NEW,
                extracted_name=name,
                target_name=None,
                similarity=best_similarity,
                confidence=1.0,
                rationale=(
                    f"top candidate {best_name!r} (similarity={best_similarity}) falls in the "
                    "reasoner band; deferred to the reasoner tie-break"
                ),
                method="reasoner_band_deferred",
            ),
            scored,
        )

    return (
        ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=1.0,
            rationale=f"best candidate {best_name!r} similarity {best_similarity} below candidate threshold",
            method="below_threshold",
        ),
        [],
    )


def _batch_neighbors(
    name: str, edges_by_pair: dict[tuple[str, str], list[dict]]
) -> set[str]:
    """Names the extracted entity is linked to inside this extraction batch."""
    neighbors: set[str] = set()
    for src, tgt in edges_by_pair:
        if src == name:
            neighbors.add(tgt)
        elif tgt == name:
            neighbors.add(src)
    return neighbors


async def _gather_evidence(
    batch_neighbors: set[str],
    band_candidates: list[tuple[str, float | None]],
    knowledge_graph_inst: BaseGraphStorage,
) -> list[Candidate]:
    """Enrich in-band candidates with graph evidence (degree, shared neighbors).

    Stale candidates (scored in the vdb but not live in the graph) are dropped
    so the reasoner only ever sees existing, reusable nodes.
    """
    enriched: list[Candidate] = []
    for cand_name, similarity in band_candidates:
        node = await knowledge_graph_inst.get_node(cand_name)
        if node is None:
            continue
        edges = await knowledge_graph_inst.get_node_edges(cand_name) or []
        neighbors: set[str] = set()
        for edge in edges:
            neighbors.update(edge)
        neighbors.discard(cand_name)
        enriched.append(
            Candidate(
                name=cand_name,
                similarity=similarity,
                degree=len(edges),
                description=str(node.get("description", ""))[:200],
                shared_neighbors=sorted(batch_neighbors & neighbors),
            )
        )
    return enriched


def _parse_reasoner_response(
    text: str, candidate_names: set[str]
) -> tuple[Decision | None, str | None, float]:
    """Parse the reasoner's JSON reply. Returns (decision, target, confidence).

    ``decision`` is ``None`` when the reply is malformed, references an
    off-list target, or carries an out-of-range confidence - the caller then
    fails safe to ``CREATE_NEW``.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None, 0.0
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None, None, 0.0
    raw_decision = str(payload.get("decision", "")).strip().lower()
    try:
        decision = Decision(raw_decision)
    except ValueError:
        return None, None, 0.0
    target = payload.get("target")
    target = str(target) if target not in (None, "", "null") else None
    if target is not None and target not in candidate_names:
        return None, None, 0.0
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (ValueError, TypeError):
        return None, None, 0.0
    if not 0.0 <= confidence <= 1.0:
        return None, None, 0.0
    return decision, target, confidence


async def _run_reasoner(
    name: str,
    node_items: list[dict],
    band_candidates: list[tuple[str, float | None]],
    edges_by_pair: dict[tuple[str, str], list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    llm_func: object,
    llm_response_cache: BaseKVStorage | None,
    min_confidence: float,
    allow_promote: bool,
    llm_cache_identity: object = None,
) -> ResolutionDecision:
    """Break a reasoner-band tie with a relationship-aware LLM call.

    Fail-safe: any malformed reply, off-list target, low confidence, or absent
    live candidate resolves to ``CREATE_NEW``. ``PROMOTE`` is downgraded to
    ``DISCARD_AND_REUSE`` until the promote-plan machinery is wired.
    """
    batch_neighbors = _batch_neighbors(name, edges_by_pair)
    candidates = await _gather_evidence(
        batch_neighbors, band_candidates, knowledge_graph_inst
    )
    if not candidates:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=band_candidates[0][1] if band_candidates else None,
            confidence=1.0,
            rationale="no live candidate remained for the reasoner",
            method="no_live_candidates",
        )

    descriptions = [
        str(item["description"]) for item in node_items if item.get("description")
    ]
    candidate_lines = "\n".join(
        f"- {c.name} (similarity={c.similarity}, degree={c.degree}, "
        f"shared_neighbors={c.shared_neighbors}): {c.description}"
        for c in candidates
    )
    prompt = PROMPTS["entity_resolution"].format(
        entity_name=name,
        entity_description=" ".join(descriptions)[:512] or "(no description)",
        batch_neighbors=sorted(batch_neighbors) or "(none)",
        candidates_block=candidate_lines,
    )

    response, _ = await use_llm_func_with_cache(
        prompt,
        llm_func,
        llm_response_cache=llm_response_cache,
        cache_type="entity_resolution",
        llm_cache_identity=llm_cache_identity,
    )

    candidate_names = {c.name for c in candidates}
    decision, target, confidence = _parse_reasoner_response(response, candidate_names)
    _sims = [c.similarity for c in candidates if c.similarity is not None]
    best_similarity = max(_sims) if _sims else None

    if decision is None:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=0.0,
            rationale="reasoner reply malformed or off-list; failing safe to CREATE_NEW",
            method="malformed_llm",
        )

    if decision == Decision.CREATE_NEW or target is None:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=confidence,
            rationale="reasoner judged the extracted entity distinct",
            method="reasoner_create",
        )

    if _is_suffix_variant(name, target):
        # HARD rule: reasoner must not merge a suffix/version variant even if it
        # tries. Candidates are pre-filtered, so this is defence-in-depth.
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=confidence,
            rationale=f"reasoner target {target!r} is a suffix/version variant; kept distinct",
            method="variant_guard",
        )

    if confidence < min_confidence:
        return ResolutionDecision(
            decision=Decision.CREATE_NEW,
            extracted_name=name,
            target_name=None,
            similarity=best_similarity,
            confidence=confidence,
            rationale=(
                f"reasoner chose {decision.value} -> {target!r} but confidence "
                f"{confidence:.2f} < {min_confidence:.2f}; failing safe to CREATE_NEW"
            ),
            method="low_confidence",
        )

    if decision == Decision.PROMOTE:
        if not allow_promote:
            # PROMOTE is disabled: reuse the existing canonical name instead of
            # renaming it. Still drift-reducing and non-destructive.
            return ResolutionDecision(
                decision=Decision.DISCARD_AND_REUSE,
                extracted_name=name,
                target_name=target,
                similarity=best_similarity,
                confidence=confidence,
                rationale=f"reasoner PROMOTE downgraded to reuse existing {target!r} (promote disabled)",
                method="promote_downgraded",
            )
        return ResolutionDecision(
            decision=Decision.PROMOTE,
            extracted_name=name,
            target_name=target,
            similarity=best_similarity,
            confidence=confidence,
            rationale=f"reasoner promotes extracted name over existing {target!r}",
            method="reasoner_promote",
        )

    return ResolutionDecision(
        decision=Decision.DISCARD_AND_REUSE,
        extracted_name=name,
        target_name=target,
        similarity=best_similarity,
        confidence=confidence,
        rationale=f"reasoner judged the extracted entity a duplicate of {target!r}",
        method="reasoner_discard",
    )


def _error_fallback(name: str, exc: Exception) -> ResolutionDecision:
    """Downgrade any per-entity failure to CREATE_NEW; never abort the batch."""
    logger.warning(
        "entity_resolution: failed resolving %r, defaulting to CREATE_NEW: %s",
        name,
        exc,
    )
    return ResolutionDecision(
        decision=Decision.CREATE_NEW,
        extracted_name=name,
        target_name=None,
        similarity=None,
        confidence=0.0,
        rationale=f"resolution raised {type(exc).__name__}: {exc}",
        method="error_fallback",
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

    Deterministic gates run first; names whose best candidate lands in the
    reasoner band are handed to the LLM reasoner when
    ``entity_resolution_use_reasoner`` is set and the per-batch LLM-call cap
    is not yet reached. Every per-entity failure is caught and downgraded to
    ``CREATE_NEW`` so a single bad hit can never abort ingest.
    """
    auto_threshold = float(
        global_config.get("entity_resolution_auto_merge_similarity", 0.98)
    )
    candidate_threshold = float(
        global_config.get("entity_resolution_candidate_similarity", 0.85)
    )
    top_k = int(global_config.get("entity_resolution_top_k", 5))
    use_reasoner = bool(global_config.get("entity_resolution_use_reasoner", True))
    min_confidence = float(global_config.get("entity_resolution_min_confidence", 0.80))
    allow_promote = bool(global_config.get("entity_resolution_allow_promote", False))
    max_llm_calls = int(
        global_config.get("entity_resolution_max_llm_calls_per_batch", 50)
    )
    llm_func, llm_role = _select_reasoner_llm_func_and_role(global_config)
    llm_cache_identity = (
        get_llm_cache_identity(global_config, llm_role) if llm_role else None
    )

    name_map: dict[str, str] = {}
    promote_plans: list[PromotePlan] = []
    records: list[ResolutionDecision] = []
    llm_calls = 0

    names = sorted(nodes_by_name)
    gate_concurrency = max(
        1,
        int(
            global_config.get(
                "entity_resolution_max_concurrency", _DEFAULT_GATE_CONCURRENCY
            )
        ),
    )
    gate_sem = asyncio.Semaphore(gate_concurrency)

    async def _gate(
        name: str,
    ) -> tuple[ResolutionDecision | None, list | None, Exception | None]:
        """Run the deterministic gates for one name, never raising."""
        async with gate_sem:
            try:
                record, band_candidates = await _resolve_one(
                    name,
                    nodes_by_name[name],
                    knowledge_graph_inst,
                    entity_vdb,
                    auto_threshold,
                    candidate_threshold,
                    top_k,
                )
                return record, band_candidates, None
            except Exception as exc:  # noqa: BLE001 - fail-safe: never abort ingest
                return None, None, exc

    # The gates are independent: nothing mutates the graph until
    # apply_promotions runs, so resolving them concurrently cannot change any
    # outcome. The reasoner stage below stays sequential, because its
    # per-batch call budget must be spent in a deterministic order.
    gated = await asyncio.gather(*(_gate(name) for name in names))

    for name, (record, band_candidates, gate_exc) in zip(names, gated):
        if gate_exc is not None:
            record = _error_fallback(name, gate_exc)
        elif (
            record.method == "reasoner_band_deferred"
            and use_reasoner
            and band_candidates
            and llm_func is not None
            and llm_calls < max_llm_calls
        ):
            llm_calls += 1
            try:
                record = await _run_reasoner(
                    name,
                    nodes_by_name[name],
                    band_candidates,
                    edges_by_pair,
                    knowledge_graph_inst,
                    llm_func,
                    llm_response_cache,
                    min_confidence,
                    allow_promote,
                    llm_cache_identity,
                )
            except Exception as exc:  # noqa: BLE001 - fail-safe: never abort ingest
                record = _error_fallback(name, exc)

        records.append(record)
        if record.decision == Decision.PROMOTE and record.target_name:
            # Extracted name (new) wins; existing target (old) is absorbed by
            # apply_promotions. Capture old's node + edges as the reversal record.
            plan = await _capture_promote_plan(
                record.target_name, name, knowledge_graph_inst
            )
            if plan is not None:
                promote_plans.append(plan)
            name_map[name] = name  # the new node is kept; old is re-homed later
        elif record.decision == Decision.DISCARD_AND_REUSE and record.target_name:
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


async def _capture_promote_plan(
    old_name: str,
    new_name: str,
    knowledge_graph_inst: BaseGraphStorage,
) -> PromotePlan | None:
    """Snapshot the existing node + its edges before a PROMOTE re-homes them.

    Returns None when the target is not a live node (stale vdb / already gone),
    so a PROMOTE against a vanished node degrades to a plain create.
    """
    old_node = await knowledge_graph_inst.get_node(old_name)
    if old_node is None:
        return None
    edge_pairs = await knowledge_graph_inst.get_node_edges(old_name)
    old_edges: list[tuple[str, str, dict]] = []
    for src, tgt in edge_pairs or []:
        data = await knowledge_graph_inst.get_edge(src, tgt)
        old_edges.append((src, tgt, dict(data) if data else {}))
    return PromotePlan(
        old_name=old_name,
        new_name=new_name,
        old_node_data=dict(old_node),
        old_edges=old_edges,
    )


def _inject_old_node_into_batch(
    nodes_by_name: dict[str, list[dict]], new_name: str, old_node_data: dict
) -> None:
    """Add the promoted-away node's content to the batch node for new_name.

    The subsequent merge then absorbs the old node's description/source instead
    of losing it when the old node is deleted.
    """
    item = dict(old_node_data)
    item["entity_name"] = new_name
    item["entity_id"] = new_name
    nodes_by_name.setdefault(new_name, []).append(item)


def _write_promote_undo(plan: PromotePlan, global_config: dict) -> None:
    """Append a full reversal record BEFORE any destructive promote step."""
    try:
        working_dir = Path(global_config.get("working_dir", "."))
        undo_path = working_dir / "entity_resolution_promote_undo.jsonl"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = {
            "ts": ts,
            "workspace": global_config.get("workspace", ""),
            "old_name": plan.old_name,
            "new_name": plan.new_name,
            "old_node_data": plan.old_node_data,
            "old_edges": [[s, t, d] for (s, t, d) in plan.old_edges],
        }
        with undo_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception as exc:  # noqa: BLE001 - logging must never break ingest
        logger.warning("entity_resolution: failed to write promote-undo log: %s", exc)


async def _migrate_entity_chunks(
    entity_chunks_storage: BaseKVStorage | None, old_name: str, new_name: str
) -> None:
    """Move the per-entity chunk-tracking record from old_name to new_name."""
    if entity_chunks_storage is None:
        return
    old_record = await entity_chunks_storage.get_by_id(old_name)
    if not old_record:
        return
    new_record = await entity_chunks_storage.get_by_id(new_name)
    merged_chunks = list((new_record or {}).get("chunk_ids", []))
    for chunk_id in old_record.get("chunk_ids", []):
        if chunk_id not in merged_chunks:
            merged_chunks.append(chunk_id)
    await entity_chunks_storage.upsert(
        {new_name: {"chunk_ids": merged_chunks, "count": len(merged_chunks)}}
    )
    await entity_chunks_storage.delete([old_name])


async def apply_promotions(
    plans: list[PromotePlan],
    nodes_by_name: dict[str, list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
    entity_chunks_storage: BaseKVStorage | None = None,
) -> int:
    """Execute PROMOTE plans: re-home an existing node onto the new name.

    Per plan the order is reversal-log -> re-home edges -> inject content ->
    delete old node -> delete old vector -> migrate chunk key, so the reversal
    record is durable before any destructive step. Idempotent: a plan whose old
    node is already gone (re-run) is skipped. Never raises: a single failed
    promote is logged and skipped, never aborting the merge.
    """
    applied = 0
    for plan in plans:
        try:
            if await knowledge_graph_inst.get_node(plan.old_name) is None:
                continue  # already promoted on a previous run
            _write_promote_undo(plan, global_config)
            new = plan.new_name
            for src, tgt, data in plan.old_edges:
                other = tgt if src == plan.old_name else src
                if other == new or other == plan.old_name:
                    continue  # drop the degenerate/self edge
                await knowledge_graph_inst.upsert_edge(new, other, data)
            _inject_old_node_into_batch(nodes_by_name, new, plan.old_node_data)
            await knowledge_graph_inst.delete_node(plan.old_name)
            if entity_vdb is not None:
                await entity_vdb.delete_entity(plan.old_name)
            await _migrate_entity_chunks(entity_chunks_storage, plan.old_name, new)
            applied += 1
        except Exception as exc:  # noqa: BLE001 - never abort merge on a promote
            logger.warning(
                "entity_resolution: promote %r -> %r skipped: %s",
                plan.old_name,
                plan.new_name,
                exc,
            )
    return applied
