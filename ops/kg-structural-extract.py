#!/usr/bin/env python3
# Deterministic structural-edge extractor for the OceanStack code KG.
# Parses .py (ast) + .sql (sqlglot) and emits ground-truth nodes+edges keyed by
# the SAME canonical names LightRAG uses, so edges bind to existing graph nodes
# instead of creating duplicates. Output: a custom_kg JSON {entities,relationships}.
import sys
import ast
import json
import re
import pathlib
import subprocess
sys.path.insert(0, "/fast-array/lightrag/.venv/lib/python3.13/site-packages")
from lightrag.operate import _canonical_entity_name as canon  # match existing node names

REPO = pathlib.Path("/home/spadon/Codebases/OceanStack")
def rg_files(ext, extra=()):
    args = ["rg","--files","-t" if ext in ("py","sql") else "-g", ext if ext in ("py","sql") else f"*.{ext}",
            "-g","!external/**","-g","!.venv/**","-g","!target/**"]
    if ext == "py": args += ["-g","!tests/**"]
    out = subprocess.run(["bash","-lc", f"cd {REPO} && rg --files {'-tpy' if ext=='py' else '-tsql' if ext=='sql' else ''} "
        + ("-g '*.rs'" if ext=='rs' else "-g '*.wgsl'" if ext=='wgsl' else "")
        + " -g '!external/**' -g '!.venv/**' -g '!target/**' " + ("-g '!tests/**'" if ext=='py' else "")],
        capture_output=True, text=True).stdout.split()
    return [REPO/p for p in out]

ENT = {}   # canon_name -> {entity_type, description, file}
REL = []   # (src_canon, tgt_canon, keyword, description, file)
def add_ent(name, etype, desc, f):
    c = canon(name)
    if not c: return None
    ENT.setdefault(c, {"entity_type": etype, "description": desc, "file": f})
    return c
def add_rel(s, t, kw, desc, f):
    if s and t and s != t: REL.append((s, t, kw, desc, f))

# module dotted path for a repo-relative .py file
def py_module(rel):
    p = rel.with_suffix("")
    parts = list(p.parts)
    if parts and parts[0] == "src": parts = parts[1:]
    if parts and parts[-1] == "__init__": parts = parts[:-1]
    return ".".join(parts)

# ---------- PYTHON ----------
pyfiles = rg_files("py")
# pass 1: global symbol index (qualified -> canon) for call/import resolution
defined = {}   # simple_name -> canon (last wins; good enough for repo-unique names)
mod_canon = {}
for f in pyfiles:
    rel = f.relative_to(REPO); mod = py_module(rel)
    mc = add_ent(mod or rel.stem, "module", f"Python module {mod}", str(rel))
    mod_canon[str(rel)] = mc
    try: tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.setdefault(n.name, set()).add(canon(n.name))

for f in pyfiles:
    rel = f.relative_to(REPO); mod = py_module(rel); mc = mod_canon[str(rel)]
    try: tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
    except Exception: continue
    # imports -> internal module deps
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and ("oceanstack" in (n.module or "")):
            tc = add_ent(n.module, "module", f"Python module {n.module}", str(rel))
            add_rel(mc, tc, "imports", f"{mod} imports from {n.module}", str(rel))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if "oceanstack" in a.name:
                    tc = add_ent(a.name, "module", f"Python module {a.name}", str(rel))
                    add_rel(mc, tc, "imports", f"{mod} imports {a.name}", str(rel))
    # classes -> methods, bases, decorators; module defines
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cc = add_ent(node.name, "dataclass" if any('dataclass' in ast.dump(d) for d in node.decorator_list) else "class",
                         f"Class {node.name} in {mod}", str(rel))
            add_rel(mc, cc, "defines", f"{mod} defines {node.name}", str(rel))
            for b in node.bases:
                bn = getattr(b, "id", getattr(b, "attr", None))
                if bn: add_rel(cc, canon(bn), "inherits_from", f"{node.name} inherits {bn}", str(rel))
            for it in node.body:
                if isinstance(it, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mname = f"{node.name}.{it.name}"
                    mcm = add_ent(mname, "method", f"Method {it.name} of {node.name}", str(rel))
                    add_rel(cc, mcm, "has_method", f"{node.name} has method {it.name}", str(rel))
                    if it.returns is not None:
                        rn = getattr(it.returns, "id", getattr(it.returns, "attr", None))
                        if rn and rn in defined: add_rel(mcm, next(iter(defined[rn])), "returns_type", f"{it.name} returns {rn}", str(rel))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fc = add_ent(node.name, "function", f"Function {node.name} in {mod}", str(rel))
            add_rel(mc, fc, "defines", f"{mod} defines {node.name}", str(rel))
            if node.returns is not None:
                rn = getattr(node.returns, "id", getattr(node.returns, "attr", None))
                if rn and rn in defined: add_rel(fc, next(iter(defined[rn])), "returns_type", f"{node.name} returns {rn}", str(rel))
    # calls (resolve callee simple-name to a defined symbol)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            caller = canon(n.name)
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    callee = getattr(fn, "id", getattr(fn, "attr", None))
                    if callee and callee in defined and len(defined[callee])==1:
                        tgt=next(iter(defined[callee]))
                        if tgt!=caller: add_rel(caller, tgt, "calls", f"{n.name} calls {callee}", str(rel))

# ---------- SQL (regex — robust on PG/TimescaleDB DDL) ----------
sqlfiles = rg_files("sql")
alltxt = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in sqlfiles)
# domain names (CREATE DOMAIN x ...)
domains = set(re.findall(r"CREATE\s+DOMAIN\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][\w\.]*)", alltxt, re.I))
for d in domains: add_ent(d.split(".")[-1], "domain_type", f"SQL domain {d}", "db/schemas/types")
BUILTIN = ("int","integer","text","bool","boolean","double","timestamp","timestamptz","numeric","real","bigint","smallint","jsonb","json","bytea","date","char","varchar","geometry","geography","uuid","serial","bigserial","smallserial","float","decimal","interval","inet","cidr","tsvector","oid","name","void","record","trigger")
for f in sqlfiles:
    rel = f.relative_to(REPO); txt = f.read_text(encoding="utf-8", errors="ignore")
    # CREATE TABLE schema.name ( ... )
    for m in re.finditer(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?((?:[a-zA-Z_]\w*)\.)?([a-zA-Z_]\w*)\s*\((.*?)\n\)\s*;", txt, re.I|re.S):
        schema = (m.group(1) or "").rstrip("."); tname = m.group(2); body = m.group(3)
        full = f"{schema}.{tname}" if schema else tname
        tc = add_ent(full, "table", f"Table {full}", str(rel))
        if schema: add_rel(add_ent(schema,"schema",f"Schema {schema}",str(rel)), tc, "contains", f"{schema} contains {tname}", str(rel))
        for line in body.split(","):
            cm = re.match(r"\s*([a-z_]\w*)\s+([a-zA-Z_][\w\.]*)", line)
            if not cm: continue
            cn, ctype = cm.group(1), cm.group(2).lower()
            if cn.upper() in ("CONSTRAINT","PRIMARY","FOREIGN","UNIQUE","CHECK","EXCLUDE","LIKE"): continue
            colfull = f"{full}.{cn}"; cc = add_ent(colfull, "column", f"Column {cn} of {full}", str(rel))
            add_rel(tc, cc, "has_column", f"{full} has column {cn}", str(rel))
            base = ctype.split(".")[-1]
            if base in domains or base in {d.split('.')[-1] for d in domains}:
                add_rel(cc, add_ent(base,"domain_type",f"Domain {base}",str(rel)), "typed_as", f"{cn} typed as {base}", str(rel))
    # CREATE INDEX name ON schema.table
    for m in re.finditer(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_]\w*)\s+ON\s+(?:ONLY\s+)?((?:[a-zA-Z_]\w*\.)?[a-zA-Z_]\w*)", txt, re.I):
        iname, tname = m.group(1), m.group(2)
        ic = add_ent(iname,"index",f"Index {iname}",str(rel))
        add_rel(ic, add_ent(tname,"table",f"Table {tname}",str(rel)), "indexes", f"{iname} indexes {tname}", str(rel))
    # CREATE MATERIALIZED VIEW (cagg) ... FROM tables
    for m in re.finditer(r"CREATE\s+MATERIALIZED\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?((?:[a-zA-Z_]\w*\.)?[a-zA-Z_]\w*)(.*?)(?:;|\Z)", txt, re.I|re.S):
        vfull = m.group(1); vc = add_ent(vfull,"cagg",f"Continuous aggregate {vfull}",str(rel))
        for tm in re.finditer(r"FROM\s+((?:[a-zA-Z_]\w*\.)?[a-zA-Z_]\w*)", m.group(2), re.I):
            tn = tm.group(1)
            if tn != vfull: add_rel(vc, add_ent(tn,"table",f"Table {tn}",str(rel)), "aggregates_from", f"{vfull} aggregates from {tn}", str(rel))


# ---------- RUST (regex, best-effort) ----------
rsfiles = rg_files("rs")
def rust_mod(rel):
    parts=list(rel.with_suffix("").parts)
    if "src" in parts: parts=parts[parts.index("src")+1:]
    if parts and parts[-1] in ("mod","lib","main"): parts=parts[:-1]
    return "::".join(parts)
rdefs={}
for f in rsfiles:
    txt=f.read_text(errors="ignore")
    for m in re.finditer(r"\b(?:pub\s+)?(?:async\s+)?fn\s+([a-z_]\w*)", txt): rdefs.setdefault(m.group(1),set()).add(canon(m.group(1)))
    for m in re.finditer(r"\b(?:pub\s+)?(?:struct|enum)\s+([A-Z]\w*)", txt): rdefs.setdefault(m.group(1),set()).add(canon(m.group(1)))
    for m in re.finditer(r"macro_rules!\s+([a-z_]\w*)", txt): rdefs.setdefault(m.group(1)+"!",set()).add(canon(m.group(1)+"!"))
for f in rsfiles:
    rel=f.relative_to(REPO); txt=f.read_text(errors="ignore"); mod=rust_mod(rel)
    mc=add_ent(mod or rel.stem,"module",f"Rust module {mod}",str(rel))
    for m in re.finditer(r"\b(?:pub\s+)?struct\s+([A-Z]\w*)", txt):
        c=add_ent(m.group(1),"class",f"Rust struct {m.group(1)}",str(rel)); add_rel(mc,c,"defines",f"{mod} defines {m.group(1)}",str(rel))
    for m in re.finditer(r"\b(?:pub\s+)?enum\s+([A-Z]\w*)", txt):
        c=add_ent(m.group(1),"enum",f"Rust enum {m.group(1)}",str(rel)); add_rel(mc,c,"defines",f"{mod} defines {m.group(1)}",str(rel))
    for m in re.finditer(r"\b(?:pub\s+)?trait\s+([A-Z]\w*)", txt):
        c=add_ent(m.group(1),"protocol",f"Rust trait {m.group(1)}",str(rel)); add_rel(mc,c,"defines",f"{mod} defines {m.group(1)}",str(rel))
    for m in re.finditer(r"macro_rules!\s+([a-z_]\w*)", txt):
        c=add_ent(m.group(1)+"!","macro",f"Rust macro {m.group(1)}!",str(rel)); add_rel(mc,c,"defines",f"{mod} defines macro",str(rel))
    for m in re.finditer(r"#\[pyfunction[^\]]*\][\s\S]{0,80}?fn\s+([a-z_]\w*)", txt):
        c=add_ent(m.group(1),"ffi_binding",f"PyO3 binding {m.group(1)}",str(rel)); add_rel(mc,c,"defines",f"{mod} exports {m.group(1)}",str(rel))
    for m in re.finditer(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-z_]\w*)", txt, re.M):
        c=add_ent(m.group(1),"function",f"Rust fn {m.group(1)}",str(rel)); add_rel(mc,c,"defines",f"{mod} defines {m.group(1)}",str(rel))
    for im in re.finditer(r"impl(?:<[^>]*>)?\s+(?:[\w:<>]+\s+for\s+)?([A-Z]\w*)([\s\S]*?)\n\}", txt):
        ty=im.group(1); body=im.group(2); tc=canon(ty)
        for fm in re.finditer(r"\bfn\s+([a-z_]\w*)", body):
            mm=add_ent(f"{ty}.{fm.group(1)}","method",f"Method {fm.group(1)} of {ty}",str(rel)); add_rel(tc,mm,"has_method",f"{ty} has {fm.group(1)}",str(rel))
    for m in re.finditer(r"(?:pub\s+)?(?:async\s+)?fn\s+([a-z_]\w*)[^{]*\{([\s\S]*?)\n\}", txt):
        caller=canon(m.group(1)); body=m.group(2)
        for cm in set(re.findall(r"\b([a-z_]\w*)\s*\(", body)):
            if cm in rdefs and len(rdefs[cm])==1:
                tgt=next(iter(rdefs[cm]))
                if tgt!=caller: add_rel(caller,tgt,"calls",f"{m.group(1)} calls {cm}",str(rel))

# ---------- Python constant usage (function -> constant) ----------
consts={}
for f in pyfiles:
    try: tree=ast.parse(f.read_text(encoding="utf-8",errors="ignore"))
    except Exception: continue
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t,ast.Name) and t.id.isupper() and len(t.id)>2: consts[t.id]=canon(t.id)
for f in pyfiles:
    rel=f.relative_to(REPO)
    try: tree=ast.parse(f.read_text(encoding="utf-8",errors="ignore"))
    except Exception: continue
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            caller=canon(n.name); used=set()
            for sub in ast.walk(n):
                if isinstance(sub,ast.Name) and sub.id in consts: used.add(sub.id)
            for c in used:
                if consts[c]!=caller: add_rel(caller,consts[c],"uses",f"{n.name} uses {c}",str(rel))



# ---------- enum variant -> enum  +  Rust const usage  +  broader Py const usage ----------
# enum/class variants named "Base::Variant" -> link variant to Base
for nm in list(ENT.keys()):
    if "::" in nm:
        base=nm.split("::")[0]
        if base in ENT and base!=nm:
            add_rel(nm, base, "variant_of", f"{nm} is a variant of {base}", ENT[nm]["file"])
# Rust const usage: const NAME refs inside fn bodies -> uses
rconsts={}
for f in rsfiles:
    for m in re.finditer(r"\b(?:pub\s+)?const\s+([A-Z][A-Z0-9_]{2,})", f.read_text(errors="ignore")):
        rconsts[m.group(1)]=canon(m.group(1))
for f in rsfiles:
    rel=f.relative_to(REPO); txt=f.read_text(errors="ignore")
    for m in re.finditer(r"(?:pub\s+)?(?:async\s+)?fn\s+([a-z_]\w*)[^{]*\{([\s\S]*?)\n\}", txt):
        caller=canon(m.group(1)); body=m.group(2)
        for cn in set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", body)):
            if cn in rconsts and rconsts[cn]!=caller:
                add_rel(caller, rconsts[cn], "uses", f"{m.group(1)} uses {cn}", str(rel))
# Broader Py const usage: module-level + class-body references (not just inside functions)
for f in pyfiles:
    rel=f.relative_to(REPO); mc=mod_canon.get(str(rel))
    try: tree=ast.parse(f.read_text(encoding="utf-8",errors="ignore"))
    except Exception: continue
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cc=canon(node.name); used={sub.id for sub in ast.walk(node) if isinstance(sub,ast.Name) and sub.id in consts}
            for c in used:
                if consts[c]!=cc: add_rel(cc, consts[c], "uses", f"{node.name} uses {c}", str(rel))

# ---------- emit custom_kg ----------
SRC = "structural::oceanstack"
chunks = [{"content": f"Structural facts for OceanStack ({len(ENT)} symbols).", "source_id": SRC, "file_path": "structural"}]
entities = [{"entity_name": n, "entity_type": d["entity_type"], "description": d["description"], "source_id": SRC, "file_path": d["file"]}
            for n, d in ENT.items()]
# dedup relationships
seen=set(); rels=[]
for s,t,kw,desc,f in REL:
    k=(s,t,kw)
    if k in seen: continue
    seen.add(k)
    rels.append({"src_id": s, "tgt_id": t, "keywords": kw, "description": desc, "weight": 1.0, "source_id": SRC, "file_path": f})
ck = {"chunks": chunks, "entities": entities, "relationships": rels}
out = "/fast-array/lightrag/structural_kg.json"
json.dump(ck, open(out,"w"))
sys.stdout.write(f"entities={len(entities)} relationships={len(rels)}\n")
from collections import Counter
sys.stdout.write("edge predicates: " + str(dict(Counter(r['keywords'] for r in rels).most_common())) + "\n")
sys.stdout.write("entity types: " + str(dict(Counter(e['entity_type'] for e in entities).most_common())) + "\n")
sys.stdout.write(f"written: {out}\n")
