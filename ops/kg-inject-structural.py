import json
import time
import sys
import networkx as nx
G="/fast-array/lightrag/rag-storage/oceanstack_code_schema/graph_chunk_entity_relation.graphml"
ck=json.load(open("/fast-array/lightrag/structural_kg.json"))
g=nx.read_graphml(G)
n0,e0=g.number_of_nodes(),g.number_of_edges()
ts=int(time.time()); SRC="structural-spine"
added_n=added_e=merged_e=0
for ent in ck["entities"]:
    nm=ent["entity_name"]
    if nm in g: continue
    g.add_node(nm, entity_id=nm, entity_type=ent["entity_type"], description=ent["description"],
               source_id=SRC, file_path=ent.get("file_path","structural"), created_at=ts, truncate="")
    added_n+=1
for r in ck["relationships"]:
    s,t,kw=r["src_id"],r["tgt_id"],r["keywords"]
    if s not in g or t not in g: continue
    if g.has_edge(s,t):
        ex=g[s][t].get("keywords","") or ""
        toks=[x for x in ex.replace(";",",").split(",") if x.strip()]
        if kw not in toks:
            g[s][t]["keywords"]=",".join(toks+[kw]); merged_e+=1
    else:
        g.add_edge(s,t, weight=1.0, keywords=kw, description=r.get("description",""),
                   source_id=SRC, file_path=r.get("file_path","structural"), created_at=ts, truncate="")
        added_e+=1
nx.write_graphml(g, G)
sys.stdout.write(f"nodes {n0}->{g.number_of_nodes()} (+{added_n})  edges {e0}->{g.number_of_edges()} (+{added_e}, merged_kw {merged_e})\n")
sys.stdout.write(f"density {g.number_of_edges()/g.number_of_nodes():.2f}\n")
orph=sum(1 for _,d in g.degree() if d==0)
sys.stdout.write(f"orphans {orph} ({100*orph/g.number_of_nodes():.1f}%)\n")
comps=list(nx.connected_components(g)); big=max(len(c) for c in comps)
sys.stdout.write(f"components {len(comps)}  largest {100*big/g.number_of_nodes():.1f}%\n")
