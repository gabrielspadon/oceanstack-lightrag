#!/usr/bin/env python3
"""Deep entity dedup for LightRAG.

Collapses variants that differ only in case AND/OR word separators
(underscore, hyphen, space) within each namespace segment. Preserves
dot-namespaces (signals.X != derived.X).

Examples merged:
  - OceanStack | oceanstack | Ocean Stack | ocean_stack | Oceanstack
  - oceanstack-core | oceanstack_core | OceanStackCore | OceanstackCore
  - signals.ais_position_reports | Signals.Ais_Position_Reports
                                 | signals.AIS_position_reports

Preserved (different namespace):
  - signals.ais_position_reports vs derived.ais_position_reports
  - OceanStack vs OceanStack-core

Canonical pick: most-popular variant by node degree. Falls back to longest,
then lexicographic.

Run after ingest. Pacing + backoff designed to live under Gemini Flash-Lite
RPM quota.
"""

import os
import sys
import time
import argparse
from collections import defaultdict

import httpx

API = os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621")
KEY = os.environ["LIGHTRAG_API_KEY"]
HEADERS = {"X-API-Key": KEY}

# Use the SAME canonicalizer that ingest uses (lightrag/operate.py patch).
# Single source of truth; impossible to drift.
from lightrag.operate import _canonical_entity_name as _canonical


def normalize(name: str) -> str:
    return _canonical(name)


def list_all_labels(client: httpx.Client) -> list[str]:
    # /graph/label/list defaults to 1000 (server max 10000). Request the max so
    # deep-dedup sees the whole label set, not just the alphabetical first 1000.
    r = client.get("/graph/label/list", params={"limit": 10000})
    r.raise_for_status()
    return r.json()


def list_popular(client: httpx.Client, limit: int = 1000) -> list[str]:
    r = client.get("/graph/label/popular", params={"limit": min(limit, 1000)})
    r.raise_for_status()
    return r.json()


def pick_canonical(variants: list[str], popularity_rank: dict[str, int], canon_key: str | None = None) -> str:
    """Winner selection. If any variant already equals the canonicalized key
    (i.e. it is the fixpoint of _canonical_entity_name), prefer that one — the
    canonicalizer is the source of truth, including bare→schema-qualified
    table mapping. Among non-fixpoint variants, fall back to most-connected,
    longest, then lex order."""

    def key(v: str) -> tuple[int, int, int, str]:
        is_fixpoint = 0 if (canon_key is not None and v == canon_key) else 1
        return (is_fixpoint, popularity_rank.get(v, 10**9), -len(v), v)

    return sorted(variants, key=key)[0]


def merge_group(
    client: httpx.Client,
    target: str,
    sources: list[str],
    max_retries: int = 4,
) -> tuple[bool, str]:
    """Lightrag 1.4.15 merge endpoint has a response-serialization bug that
    returns HTTP 500 while the merge IS applied server-side. We verify
    post-hoc by checking that none of the source labels still exist."""
    payload = {"entities_to_change": sources, "entity_to_change_into": target}
    delay = 30
    for attempt in range(max_retries + 1):
        try:
            r = client.post("/graph/entities/merge", json=payload, timeout=180.0)
            if r.status_code == 200:
                return True, r.json().get("message", "ok")
            body = r.text[:300]
            if r.status_code == 429 or "RESOURCE_EXHAUSTED" in body or "quota" in body.lower():
                if attempt < max_retries:
                    print(f"    rate-limited, sleeping {delay}s (retry {attempt + 1})")
                    time.sleep(delay)
                    delay *= 2
                    continue
            # Known lightrag 1.4.15 bug: server merges successfully but
            # response serialization fails with vars/__dict__ TypeError →
            # HTTP 500. Verify by polling whether source labels are gone.
            if r.status_code == 500 and (
                "cannot convert dictionary" in body or "__dict__" in body or "Internal Server Error" in body
            ):
                still = []
                for s in sources:
                    try:
                        rr = client.get("/graph/entity/exists", params={"name": s}, timeout=15.0)
                        if rr.status_code == 200 and rr.json().get("exists", False):
                            still.append(s)
                    except Exception:
                        pass
                if not still:
                    return True, "ok (verified gone after HTTP 500 — known serializer bug)"
                return False, f"HTTP 500 + sources still present: {still}"
            return False, f"HTTP {r.status_code}: {body}"
        except httpx.ReadTimeout:
            return False, "timeout"
        except Exception as e:
            return False, f"exception: {e}"
    return False, "exhausted retries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap groups (0=all)")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between merges")
    ap.add_argument("--max-failures", type=int, default=10, help="abort after N consecutive failures")
    ap.add_argument("--min-variants", type=int, default=2, help="only merge groups with >= N variants")
    args = ap.parse_args()

    with httpx.Client(base_url=API, headers=HEADERS, timeout=httpx.Timeout(60.0)) as client:
        print(f"fetching labels from {API}...")
        labels = list_all_labels(client)
        popular = list_popular(client, limit=1000)
        rank = {label: i for i, label in enumerate(popular)}
        print(f"  total labels: {len(labels)}")
        print(f"  popularity ranked (top): {len(popular)}")

        groups: dict[str, list[str]] = defaultdict(list)
        for label in labels:
            norm = normalize(label)
            if len(norm) >= args.min_len:
                groups[norm].append(label)

        dups = {k: v for k, v in groups.items() if len(v) >= args.min_variants}
        print(f"  duplicate groups (case+separator collapsed): {len(dups)}")
        total_variants = sum(len(v) for v in dups.values())
        savings = total_variants - len(dups)
        print(f"  total dup-affected labels: {total_variants}")
        print(f"  potential entity reduction: {savings} ({savings * 100 // max(total_variants, 1)}%)")

        if args.limit:
            dups = dict(list(dups.items())[: args.limit])

        merged = 0
        failed = 0
        consecutive_fails = 0

        # Sort by size desc — handle worst dups first (biggest impact)
        for canon_lc, variants in sorted(dups.items(), key=lambda x: -len(x[1])):
            target = pick_canonical(variants, rank, canon_key=canon_lc)
            sources = [v for v in variants if v != target]
            print(f"  {canon_lc!r} ({len(variants)} variants): {variants}  →  {target!r}")
            if args.dry_run:
                continue
            ok, msg = merge_group(client, target, sources)
            if ok:
                merged += 1
                consecutive_fails = 0
                print(f"    ok: {msg[:80]}")
            else:
                failed += 1
                consecutive_fails += 1
                print(f"    FAIL: {msg}")
                if consecutive_fails >= args.max_failures:
                    print(f"\nABORT: {consecutive_fails} consecutive failures")
                    break
            time.sleep(args.sleep)

        print(f"\nDONE. merged={merged} failed={failed} total_groups_targeted={len(dups)}")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
