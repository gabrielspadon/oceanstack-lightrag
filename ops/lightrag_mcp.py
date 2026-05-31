"""Comprehensive MCP shim for lightrag-server.
Wraps the LightRAG REST API (OpenAPI 3.1 spec) for Claude Code consumption.
Run via uv with fastmcp+httpx.
"""

import os
from typing import Any

import httpx
from fastmcp import FastMCP

BASE = os.environ["LIGHTRAG_URL"]
KEY = os.environ["LIGHTRAG_API_KEY"]
HEADERS = {"X-API-Key": KEY}

client = httpx.Client(
    base_url=BASE,
    headers=HEADERS,
    timeout=httpx.Timeout(180.0, connect=5.0),
)
mcp = FastMCP("lightrag")


def _post(path: str, payload: dict | None = None) -> Any:
    r = client.post(path, json=payload or {})
    r.raise_for_status()
    return r.json()


def _get(path: str, params: dict | None = None) -> Any:
    r = client.get(path, params=params or {})
    r.raise_for_status()
    return r.json()


def _delete(path: str, payload: dict | None = None) -> Any:
    r = client.request("DELETE", path, json=payload) if payload else client.delete(path)
    r.raise_for_status()
    return r.json()


# =========================================================================
# Documents — ingestion, listing, status, lifecycle
# =========================================================================


@mcp.tool()
def documents_scan() -> dict:
    """Scan INPUT_DIR for new files (.txt/.md/.mdx/.pdf) and ingest them."""
    return _post("/documents/scan")


@mcp.tool()
def insert_text(text: str, file_source: str = "") -> dict:
    """Insert a single text into the KB. file_source labels it for citation."""
    payload: dict = {"text": text}
    if file_source:
        payload["file_source"] = file_source
    return _post("/documents/text", payload)


@mcp.tool()
def insert_texts(texts: list[str], file_sources: list[str] | None = None) -> dict:
    """Insert multiple texts in one batch. file_sources order matches texts."""
    payload: dict = {"texts": texts}
    if file_sources:
        payload["file_sources"] = file_sources
    return _post("/documents/texts", payload)


@mcp.tool()
def list_documents() -> dict:
    """Deprecated: list up to 1000 docs grouped by status. Use list_documents_paginated."""
    return _get("/documents")


@mcp.tool()
def list_documents_paginated(
    page: int = 1,
    page_size: int = 50,
    status_filter: str | None = None,
    sort_field: str = "updated_at",
    sort_direction: str = "desc",
) -> dict:
    """Paginated doc list. status_filter in {PENDING,PROCESSING,PROCESSED,FAILED} or None."""
    payload: dict = {
        "page": page,
        "page_size": page_size,
        "sort_field": sort_field,
        "sort_direction": sort_direction,
    }
    if status_filter:
        payload["status_filter"] = status_filter
    return _post("/documents/paginated", payload)


@mcp.tool()
def status_counts() -> dict:
    """Counts of docs by status: PENDING/PROCESSING/PROCESSED/FAILED."""
    return _get("/documents/status_counts")


@mcp.tool()
def track_status(track_id: str) -> dict:
    """Get docs associated with a track_id (returned from insert/upload)."""
    return _get(f"/documents/track_status/{track_id}")


@mcp.tool()
def pipeline_status() -> dict:
    """Current pipeline status: busy, batches, latest message, history."""
    return _get("/documents/pipeline_status")


@mcp.tool()
def reprocess_failed() -> dict:
    """Re-trigger pipeline for FAILED/PENDING/stuck-PROCESSING docs."""
    return _post("/documents/reprocess_failed")


@mcp.tool()
def cancel_pipeline() -> dict:
    """Request graceful cancellation of running ingest pipeline."""
    return _post("/documents/cancel_pipeline")


@mcp.tool()
def delete_documents(
    doc_ids: list[str],
    delete_file: bool = False,
    delete_llm_cache: bool = False,
) -> dict:
    """Delete documents by IDs. delete_file removes source from INPUT_DIR."""
    return _delete(
        "/documents/delete_document",
        {
            "doc_ids": doc_ids,
            "delete_file": delete_file,
            "delete_llm_cache": delete_llm_cache,
        },
    )


@mcp.tool()
def clear_documents() -> dict:
    """Wipe ALL documents, entities, relations, and INPUT_DIR files. Destructive."""
    return _delete("/documents")


@mcp.tool()
def clear_llm_cache() -> dict:
    """Clear all cached LLM responses (forces re-extraction on retry)."""
    return _post("/documents/clear_cache")


# =========================================================================
# Query — RAG generation + raw retrieval
# =========================================================================


@mcp.tool()
def query(
    text: str,
    mode: str = "mix",
    top_k: int = 40,
    chunk_top_k: int = 10,
    response_type: str = "Multiple Paragraphs",
    include_references: bool = True,
    enable_rerank: bool = False,
    max_total_tokens: int | None = None,
    user_prompt: str | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """RAG query with full LLM generation.
    mode: local|global|hybrid|naive|mix|bypass (mix recommended).
    Returns {response, references}.
    """
    payload: dict = {
        "query": f"task: search result | query: {text}",
        "mode": mode,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "response_type": response_type,
        "include_references": include_references,
        "enable_rerank": enable_rerank,
        "stream": False,
    }
    if max_total_tokens:
        payload["max_total_tokens"] = max_total_tokens
    if user_prompt:
        payload["user_prompt"] = user_prompt
    if conversation_history:
        payload["conversation_history"] = conversation_history
    return _post("/query", payload)


@mcp.tool()
def query_data(
    text: str,
    mode: str = "mix",
    top_k: int = 40,
    chunk_top_k: int = 10,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
) -> dict:
    """Raw retrieval (no LLM). Returns entities, relations, chunks, references.
    Use for debugging KB or building custom pipelines."""
    payload: dict = {
        "query": f"task: search result | query: {text}",
        "mode": mode,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
    }
    for k, v in {
        "max_entity_tokens": max_entity_tokens,
        "max_relation_tokens": max_relation_tokens,
        "max_total_tokens": max_total_tokens,
    }.items():
        if v is not None:
            payload[k] = v
    return _post("/query/data", payload)


@mcp.tool()
def query_with_keywords(
    text: str,
    high_level_keywords: list[str],
    low_level_keywords: list[str],
    mode: str = "mix",
    top_k: int = 40,
) -> dict:
    """Bypass LLM keyword extraction by providing them directly.
    hl_keywords = themes/concepts; ll_keywords = specific entities/terms."""
    payload = {
        "query": f"task: search result | query: {text}",
        "mode": mode,
        "top_k": top_k,
        "hl_keywords": high_level_keywords,
        "ll_keywords": low_level_keywords,
        "stream": False,
    }
    return _post("/query", payload)


# =========================================================================
# Graph — labels, search, subgraph, entity/relation CRUD
# =========================================================================


@mcp.tool()
def graph_labels() -> list:
    """All entity/relation labels in the KG (capped at server max)."""
    return _get("/graph/label/list")


@mcp.tool()
def graph_labels_popular(limit: int = 300) -> list:
    """Top labels by node degree (most connected entities). Max 1000."""
    return _get("/graph/label/popular", {"limit": min(limit, 1000)})


@mcp.tool()
def graph_labels_search(query: str, limit: int = 50) -> list:
    """Fuzzy search labels by name. Max limit=100."""
    return _get("/graph/label/search", {"q": query, "limit": min(limit, 100)})


@mcp.tool()
def graph_subgraph(label: str, max_depth: int = 3, max_nodes: int = 1000) -> dict:
    """Get connected subgraph rooted at the given entity label.
    Prioritizes by hops then degree."""
    return _get(
        "/graphs",
        {"label": label, "max_depth": max_depth, "max_nodes": max_nodes},
    )


@mcp.tool()
def entity_exists(name: str) -> dict:
    """Check if an entity exists in the KG. Returns {exists: bool}."""
    return _get("/graph/entity/exists", {"name": name})


@mcp.tool()
def entity_create(
    entity_name: str,
    description: str,
    entity_type: str = "CONCEPT",
    extra_properties: dict | None = None,
) -> dict:
    """Create a new entity. entity_type e.g. PERSON, ORGANIZATION, LOCATION, CONCEPT."""
    entity_data = {"description": description, "entity_type": entity_type}
    if extra_properties:
        entity_data.update(extra_properties)
    return _post(
        "/graph/entity/create",
        {"entity_name": entity_name, "entity_data": entity_data},
    )


@mcp.tool()
def entity_edit(
    entity_name: str,
    updated_data: dict,
    allow_rename: bool = False,
    allow_merge: bool = False,
) -> dict:
    """Update entity properties. allow_merge=true to fold into existing target on rename collision."""
    return _post(
        "/graph/entity/edit",
        {
            "entity_name": entity_name,
            "updated_data": updated_data,
            "allow_rename": allow_rename,
            "allow_merge": allow_merge,
        },
    )


@mcp.tool()
def entity_delete(entity_name: str) -> dict:
    """Delete entity and all its relationships from KG."""
    return _delete("/documents/delete_entity", {"entity_name": entity_name})


@mcp.tool()
def entities_merge(source_entities: list[str], target_entity: str) -> dict:
    """Merge multiple entities into target. Transfers all relationships, deletes sources."""
    return _post(
        "/graph/entities/merge",
        {
            "entities_to_change": source_entities,
            "entity_to_change_into": target_entity,
        },
    )


@mcp.tool()
def relation_create(
    source_entity: str,
    target_entity: str,
    description: str,
    keywords: str = "",
    weight: float = 1.0,
    extra_properties: dict | None = None,
) -> dict:
    """Create undirected relationship between two existing entities."""
    relation_data = {
        "description": description,
        "keywords": keywords,
        "weight": weight,
    }
    if extra_properties:
        relation_data.update(extra_properties)
    return _post(
        "/graph/relation/create",
        {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relation_data": relation_data,
        },
    )


@mcp.tool()
def relation_edit(source_id: str, target_id: str, updated_data: dict) -> dict:
    """Update relation properties (description, weight, keywords)."""
    return _post(
        "/graph/relation/edit",
        {"source_id": source_id, "target_id": target_id, "updated_data": updated_data},
    )


@mcp.tool()
def relation_delete(source_entity: str, target_entity: str) -> dict:
    """Delete relation between two entities."""
    return _delete(
        "/documents/delete_relation",
        {"source_entity": source_entity, "target_entity": target_entity},
    )


# =========================================================================
# System — health, auth status
# =========================================================================


@mcp.tool()
def health() -> dict:
    """Server health, active LLM/embed config, workspace, pipeline busy state."""
    return _get("/health")


@mcp.tool()
def auth_status() -> dict:
    """Auth mode + guest token if auth disabled."""
    return _get("/auth-status")


if __name__ == "__main__":
    mcp.run()
