#!/usr/bin/env python3
"""Batch ingest text/code/sql files via /documents/text endpoint.
Uses thread pool for parallelism. Reads API key from sops-decrypted env."""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

API = os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9621")
KEY = os.environ["LIGHTRAG_API_KEY"]
HEADERS = {"X-API-Key": KEY, "Content-Type": "application/json"}


def post_one(client: httpx.Client, path: Path, file_source: str) -> tuple[str, int, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return file_source, 0, "empty"
        r = client.post(
            "/documents/text",
            json={"text": text, "file_source": file_source},
        )
        if r.status_code == 200:
            return file_source, 200, r.json().get("track_id", "ok")
        return file_source, r.status_code, r.text[:200]
    except Exception as e:
        return file_source, -1, str(e)[:200]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="Dirs containing files to ingest")
    ap.add_argument("--workers", type=int, default=8, help="Parallel POSTs")
    ap.add_argument("--limit", type=int, default=0, help="Cap files (0=all)")
    ap.add_argument("--root", default="", help="Root path to compute relative file_source against (preserves slashes)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else None
    files: list[tuple[Path, str]] = []
    for d in args.dirs:
        for p in Path(d).rglob("*"):
            if not p.is_file():
                continue
            if root:
                try:
                    rel = str(p.resolve().relative_to(root))
                except ValueError:
                    rel = str(p.resolve().relative_to(Path(d).resolve()))
            else:
                rel = str(p.resolve().relative_to(Path(d).resolve()))
            files.append((p, rel))
    files.sort(key=lambda t: t[1])
    if args.limit:
        files = files[: args.limit]

    print(f"Posting {len(files)} files with {args.workers} workers to {API}")
    start = time.time()
    ok = 0
    fail = 0
    with httpx.Client(base_url=API, headers=HEADERS, timeout=60.0) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(post_one, client, p, src): (p, src) for p, src in files}
            for i, fut in enumerate(as_completed(futures), 1):
                file_source, code, msg = fut.result()
                if code == 200:
                    ok += 1
                else:
                    fail += 1
                    print(f"  FAIL [{code}] {file_source}: {msg}")
                if i % 50 == 0 or i == len(files):
                    rate = i / max(time.time() - start, 1)
                    print(f"[{i}/{len(files)}] ok={ok} fail={fail} rate={rate:.1f}/s")
    print(f"\nDONE in {time.time() - start:.1f}s. ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
