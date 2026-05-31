"""Embed graph nodes that lack a vector-store entry so every node is locatable.

The structural spine and architecture backbone are injected straight into the
graphml, which makes them traversable but not directly vector-searchable: a
vector-seeded query never lands on them and graph expansion only reaches them
when an adjacent node is already embedded. This script finds graph nodes with no
row in the entity vector table, embeds each via the same Ollama model the corpus
was built with, and upserts the vector. A preflight re-embeds an existing row and
checks cosine similarity against its stored vector to confirm the preprocessing
matches the corpus embedding space; it refuses to write if the spaces diverge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
import urllib.request

import asyncpg
import networkx as nx

GRAPH = "/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
WORKSPACE = "oceanstack_code_schema"
EMBED_MODEL = "qwen3-embedding:0.6b"
OLLAMA = "http://127.0.0.1:11434/api/embed"  # nosemgrep
DSN = "postgresql:///lightrag"

# Table name is fixed by the embedding model + dimension; spelled literally so the
# SQL below stays a constant string (no identifier interpolation, no injection surface).
SQL_SAMPLE = "SELECT entity_name, content, content_vector FROM lightrag_vdb_entity_qwen3_embedding_0_6b_1024d WHERE workspace=$1 AND content_vector IS NOT NULL LIMIT 1"  # noqa: E501
SQL_PRESENT = "SELECT entity_name FROM lightrag_vdb_entity_qwen3_embedding_0_6b_1024d WHERE workspace=$1"
SQL_UPSERT = "INSERT INTO lightrag_vdb_entity_qwen3_embedding_0_6b_1024d (id, workspace, entity_name, content, content_vector, file_path) VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (workspace, id) DO UPDATE SET content=EXCLUDED.content, content_vector=EXCLUDED.content_vector, file_path=EXCLUDED.file_path, update_time=CURRENT_TIMESTAMP"  # noqa: E501


def embed(text: str) -> list[float]:
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(  # nosemgrep
        OLLAMA, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:  # nosemgrep
        return json.loads(r.read().decode())["embeddings"][0]


def vid(name: str) -> str:
    return "ent-" + hashlib.md5(name.encode()).hexdigest()  # noqa: S324 - matches LightRAG id scheme


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-12)


async def main() -> int:
    conn = await asyncpg.connect(DSN)
    try:
        row = await conn.fetchrow(SQL_SAMPLE, WORKSPACE)  # nosemgrep
        stored = [float(x) for x in str(row["content_vector"]).strip("[]").split(",")]
        sim = cosine(stored, embed(row["content"]))
        sys.stdout.write(f"preflight cosine(re-embed vs stored)={sim:.4f} for '{row['entity_name']}'\n")
        if sim < 0.98:
            sys.stdout.write("FAIL: embedding preprocessing diverges from corpus — refusing to write.\n")
            return 1

        present = {r["entity_name"] for r in await conn.fetch(SQL_PRESENT, WORKSPACE)}  # nosemgrep
        g = nx.read_graphml(GRAPH)
        missing = [n for n in g.nodes() if n not in present]
        sys.stdout.write(f"vdb entities={len(present)} graph nodes={g.number_of_nodes()} missing={len(missing)}\n")

        ins = 0
        for n in missing:
            nd = g.nodes[n]
            desc = (nd.get("description") or "").strip()
            content = f"{n}\n{desc}" if desc else n
            vec = "[" + ",".join(map(str, embed(content))) + "]"
            # nosemgrep: python.asyncpg.security.asyncpg-string-concat.asyncpg-string-concat
            await conn.execute(SQL_UPSERT, vid(n), WORKSPACE, n[:512], content, vec, nd.get("file_path", "") or "")  # noqa: E501
            ins += 1
            if ins % 250 == 0:
                sys.stdout.write(f"  embedded {ins}/{len(missing)}\n")
                sys.stdout.flush()
        sys.stdout.write(f"done: embedded+upserted {ins} entities\n")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
