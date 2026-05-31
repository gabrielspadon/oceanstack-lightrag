"""Normalize edge-predicate sprawl in the code knowledge graph.

LLM extraction leaks two kinds of noise into the relationship keyword field: full
sentences (and tuple-delimiter spillage) where a verb belongs, and a tail of
synonym predicates for relations that already have a canonical verb. This pass
cleans malformed keywords back to a canonical verb (or related_to when no verb is
recoverable) and folds an explicit synonym set into its canonical form. It never
removes an edge and never touches descriptions — only the keyword field is
rewritten. Run with the server stopped.
"""

from __future__ import annotations

import re
import sys
from collections import Counter

import networkx as nx

GRAPH = "/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"

CANON = {
    "calls",
    "defines",
    "has_method",
    "uses",
    "reads_from",
    "depends_on",
    "instantiates",
    "imports",
    "validates_against",
    "tests",
    "has_column",
    "raises",
    "returns_type",
    "contains",
    "writes_to",
    "inherits_from",
    "indexes",
    "typed_as",
    "aggregates_from",
    "variant_of",
    "wraps",
    "provides",
    "materializes",
    "refreshes",
    "partitions",
    "joins",
    "fires_on",
    "triggered_by",
    "bound_to",
    "exports_to",
    "decodes",
    "parses",
    "configures",
    "derived_from",
    "falls_back_to",
    "emits_to",
    "implements",
    "invokes",
    "validates",
    "serialises_to",
    "deserialises_from",
    "chunked_by",
    "precedes",
    "orchestrates",
    "must_sync_with",
    "gated_by",
    "ranks",
    "ranks_first",
    "binds",
    "flows_to",
    "begins_at",
    "alias_of",
    "fk_references",
    "overrides",
    "has_param",
    "has_field",
    "decorates",
    "produces",
    "returns",
    "updates",
    "modifies",
    "processes",
    "handles",
    "transforms",
    "converts",
    "calculates",
    "compares",
    "references",
    "part_of",
    "related_to",
}

# Direction-agnostic synonym folds — only obvious dedup, never a semantics-losing merge.
SYNONYM = {
    "is_part_of": "part_of",
    "belongs_to": "part_of",
    "member_of": "part_of",
    "component_of": "part_of",
    "contained_in": "part_of",
    "is_contained_in": "part_of",
    "is_method_of": "has_method",
    "methods": "has_method",
    "generates": "produces",
    "creates": "produces",
    "constructs": "produces",
    "outputs": "produces",
    "mutates": "updates",
    "sets": "updates",
    "utilizes": "uses",
    "consumes": "uses",
    "invoked_by": "calls",
    "requires": "depends_on",
    "depends_upon": "depends_on",
}

# Canonical verbs that may legitimately prefix a leaked sentence.
VERB_PREFIX = sorted(CANON | set(SYNONYM), key=len, reverse=True)


def normalize_one(kw: str) -> str:
    kw = kw.strip()
    if not kw:
        return ""
    if kw in CANON:
        return kw
    if kw in SYNONYM:
        return SYNONYM[kw]
    # malformed: tuple-delimiter spillage or a full sentence — recover a leading verb.
    head = re.split(r"[<\s]", kw, maxsplit=1)[0].strip().lower()
    if head in SYNONYM:
        return SYNONYM[head]
    if head in CANON:
        return head
    if len(kw) > 30 or kw.count(" ") >= 4 or "<|" in kw:
        return "related_to"
    # short single-token non-canonical verb — keep it (real vocabulary).
    return kw if kw.replace("_", "").isalpha() else "related_to"


g = nx.read_graphml(GRAPH)
before = Counter()
after = Counter()
changed = 0
for _, _, d in g.edges(data=True):
    raw = str(d.get("keywords", ""))
    parts = [p for p in raw.split(",") if p.strip()]
    for p in parts:
        before[p.strip()] += 1
    norm = []
    for p in parts:
        n = normalize_one(p)
        if n and n not in norm:
            norm.append(n)
    new = ",".join(norm)
    for n in norm:
        after[n] += 1
    if new != raw:
        d["keywords"] = new
        changed += 1

nx.write_graphml(g, GRAPH)
sys.stdout.write(
    f"edges_rewritten={changed}\n"
    f"distinct_predicates: {len(before)} -> {len(after)}\n"
    f"related_to fallbacks: {after.get('related_to', 0)}\n"
)
