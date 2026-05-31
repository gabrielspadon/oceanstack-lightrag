#!/usr/bin/env python3
# Reconcile OceanStack code-KG entity types to the 22-type canonical set.
# Off-taxonomy labels (e.g. test_suite from ghost/relation-endpoint entities)
# are remapped to a canonical member or demoted to concept. Format-safe: lxml
# rewrites only the entity_type <data> text in place. The type lives only in the
# graphml (PGVector rows carry no type column). Run with the server stopped.
import sys
from collections import Counter
from pathlib import Path
from lxml import etree

GRAPHML = Path("/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml")
VALID = {"module","function","method","class","dataclass","enum","protocol","macro",
         "ffi_binding","constant","exception","schema","table","column","domain_type",
         "sql_function","cagg","index","gpu_kernel","ais_concept","library","concept"}
REMAP = {"ffi_function":"ffi_binding","pyfunction":"ffi_binding","pyo3_function":"ffi_binding",
  "pymethod":"ffi_binding","ffi":"ffi_binding","rust_macro":"macro","proc_macro":"macro",
  "macro_rules":"macro","instance_method":"method","class_method":"method","staticmethod":"method",
  "async_function":"function","coroutine":"function","decorator":"function","rust_struct":"class",
  "struct":"class","pyclass":"class","pydantic_model":"dataclass","data_class":"dataclass",
  "trait":"protocol","interface":"protocol","abc":"protocol","abstract_class":"protocol",
  "enum_type":"enum","enum_value":"enum","variant":"enum","intflag":"enum","intenum":"enum",
  "literal":"constant","type_alias":"domain_type","domain":"domain_type","type":"domain_type",
  "newtype":"domain_type","stored_procedure":"sql_function","procedure":"sql_function",
  "trigger":"sql_function","trigger_function":"sql_function","continuous_aggregate":"cagg",
  "materialized_view":"cagg","view":"cagg","hypertable":"table","gist_index":"index",
  "brin_index":"index","shader":"gpu_kernel","wgsl_shader":"gpu_kernel","compute_shader":"gpu_kernel",
  "kernel":"gpu_kernel","cuda_kernel":"gpu_kernel","maritime_concept":"ais_concept",
  "ais_message":"ais_concept","domain_concept":"ais_concept","package":"library","crate":"library",
  "dependency":"library","framework":"library","tool":"library","service":"library",
  "test_suite":"concept","test_fixture":"concept","fixture":"concept","test":"concept",
  "test_case":"concept","test_class":"concept","file":"concept","process":"concept",
  "system":"concept","component":"concept","other":"concept","unknown":"concept",
  "object":"concept","entity":"concept","category":"concept","event":"concept",
  "person":"concept","location":"concept","equipment":"concept","organization":"concept","product":"concept"}

def canon(raw):
    t = (raw or "").strip().lower()
    t = REMAP.get(t, t)
    return t if t in VALID else "concept"

def main():
    if not GRAPHML.exists():
        sys.stderr.write("graphml not found\n"); return 1
    tree = etree.parse(str(GRAPHML)); root = tree.getroot()
    ns = root.nsmap.get(None); qn = "{%s}" % ns if ns else ""
    et = None
    for k in root.iter(qn+"key"):
        if k.get("attr.name") == "entity_type": et = k.get("id"); break
    if et is None: sys.stderr.write("entity_type key not found\n"); return 1
    moved = Counter()
    for d in root.iter(qn+"data"):
        if d.get("key") != et: continue
        new = canon(d.text or "")
        if new != (d.text or ""): moved[((d.text or ""), new)] += 1; d.text = new
    if moved: tree.write(str(GRAPHML), encoding="utf-8", xml_declaration=True)
    sys.stdout.write("remapped %d nodes\n" % sum(moved.values()))
    for (s, dst), n in moved.most_common(): sys.stdout.write("  %5d  %r -> %s\n" % (n, s, dst))
    return 0

raise SystemExit(main())
