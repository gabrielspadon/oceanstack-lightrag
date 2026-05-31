"""Link AIS / code abbreviation nodes to their canonical entities.

Queries naming an abbreviation (sog, cog, mmsi, h3) should reach the canonical
domain entity (sog_knots, cog_degrees, mmsi_identity, h3_cell_index) by one hop.
Adds alias_of edges only when both endpoints already exist as graph nodes.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import networkx as nx
from lightrag.operate import _canonical_entity_name as canon

GRAPH = "/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")

ALIASES = [
    ("sog", "sog_knots"),
    ("cog", "cog_degrees"),
    ("rot", "rot_deg_per_min"),
    ("mmsi", "mmsi_identity"),
    ("haversine", "haversine_meters"),
    ("h3", "h3_cell_index"),
    ("imo", "imo_number"),
    ("heading", "heading_degrees"),
]

g = nx.read_graphml(GRAPH)
names = set(g.nodes())


def resolve(n: str) -> str | None:
    for c in (n, canon(n)):
        if c in names:
            return c
    return None


added = skipped = 0
for a, b in ALIASES:
    ra, rb = resolve(a), resolve(b)
    if ra is None or rb is None or ra == rb:
        skipped += 1
        continue
    if not g.has_edge(ra, rb):
        g.add_edge(
            ra,
            rb,
            weight=8.0,
            keywords="alias_of",
            description=f"{ra} is the common abbreviation for {rb}.",
            source_id="alias-map",
            file_path="aliases",
            created_at=NOW,
        )
        added += 1
    else:
        skipped += 1

nx.write_graphml(g, GRAPH)
sys.stdout.write(f"alias edges added={added} skipped={skipped}\n")
