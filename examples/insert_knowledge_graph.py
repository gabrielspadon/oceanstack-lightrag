"""Insert a validated typed knowledge graph with caller-owned identities."""

import asyncio

from lightrag import LightRAG
from lightrag.kg.graph_contract import (
    EvidenceRef,
    GraphAssertion,
    GraphChunk,
    GraphEntity,
    KnowledgeGraphBuild,
)
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


build_id = "product-catalog:2026-07-16"
chunk = GraphChunk(
    build_id=build_id,
    chunk_id="product-catalog:company-a-product-x",
    source_key="product-catalog/docs/company-a.md",
    source_revision="2026-07-16",
    content="CompanyA develops ProductX.",
    metadata={"section": "products"},
)
evidence = (
    EvidenceRef(
        chunk_id=chunk.chunk_id,
        source_key=chunk.source_key,
        source_revision=chunk.source_revision,
        metadata={"quote": chunk.content},
    ),
)
build = KnowledgeGraphBuild.create(
    build_id=build_id,
    chunks=(chunk,),
    entities=(
        GraphEntity(
            build_id=build_id,
            entity_id="organization:company-a",
            entity_type="Organization",
            evidence=evidence,
            metadata={"name": "CompanyA"},
        ),
        GraphEntity(
            build_id=build_id,
            entity_id="product:product-x",
            entity_type="Product",
            evidence=evidence,
            metadata={"name": "ProductX"},
        ),
    ),
    assertions=(
        GraphAssertion(
            build_id=build_id,
            assertion_id="assertion:company-a-develops-product-x",
            predicate="develops",
            src_id="organization:company-a",
            dst_id="product:product-x",
            evidence=evidence,
            confidence=1.0,
            method="curated",
        ),
    ),
    metadata={"plane": "product"},
)


async def main() -> None:
    rag = LightRAG(
        working_dir="./typed_graph",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    try:
        await rag.ainsert_knowledge_graph(build)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
