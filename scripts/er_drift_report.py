#!/usr/bin/env python3
"""Entity-resolution drift metric + resolution-log audit.

Backend-agnostic and GPU-free: operates on plain entity-name lists and on the
JSONL resolution log that ``lightrag.entity_resolution.write_resolution_log``
emits. Two jobs:

1. ``drift`` - given a newline-delimited entity-name dump, report the variant
   groups (names that collapse to the same drift key), the surplus
   (sum of group_size - 1), and the drift ratio (surplus / total). Pass a
   ``--before`` and ``--after`` dump to report the reduction.

2. ``audit`` - given a resolution-log JSONL, assert the hard invariant that no
   DISCARD_AND_REUSE / PROMOTE record ever crosses a dotted namespace
   (``signals.x`` must never merge onto ``derived.x``). Exits non-zero on any
   violation so it can gate a validation run.

The drift key mirrors the resolver's residue exactly: casefold, keep only
``[a-z0-9.]`` so dotted namespaces stay distinct.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def drift_key(name: str) -> str:
    """Casefold and strip to [a-z0-9.] - dots (namespaces) are preserved."""
    return re.sub(r"[^a-z0-9.]", "", name.casefold())


def namespace(name: str) -> str:
    """Dotted namespace prefix, or '' when the name has no dot."""
    return name.rsplit(".", 1)[0] if "." in name else ""


def load_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped:
            names.append(stripped)
    return names


def compute_drift(names: list[str]) -> dict:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[drift_key(name)].append(name)
    variant_groups = {k: v for k, v in groups.items() if len(v) > 1}
    surplus = sum(len(v) - 1 for v in variant_groups.values())
    total = len(names)
    return {
        "total": total,
        "distinct_keys": len(groups),
        "variant_groups": len(variant_groups),
        "surplus": surplus,
        "drift_ratio": (surplus / total) if total else 0.0,
        "groups": variant_groups,
    }


def cmd_drift(args: argparse.Namespace) -> int:
    after = compute_drift(load_names(Path(args.after)))
    print(
        f"[after ] total={after['total']} variant_groups={after['variant_groups']} "
        f"surplus={after['surplus']} drift_ratio={after['drift_ratio']:.4f}"
    )
    if args.show_groups:
        for key, members in sorted(after["groups"].items()):
            print(f"  {key!r}: {members}")
    if args.before:
        before = compute_drift(load_names(Path(args.before)))
        print(
            f"[before] total={before['total']} variant_groups={before['variant_groups']} "
            f"surplus={before['surplus']} drift_ratio={before['drift_ratio']:.4f}"
        )
        delta = before["surplus"] - after["surplus"]
        print(
            f"[delta ] surplus reduced by {delta} "
            f"({'PASS' if delta > 0 else 'NO REDUCTION'})"
        )
        return 0 if delta > 0 else 1
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    path = Path(args.log)
    if not path.exists():
        print(f"resolution log not found: {path}", file=sys.stderr)
        return 2
    violations = []
    merge_count = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        decision = str(record.get("decision", ""))
        target = record.get("target_name")
        extracted = record.get("extracted_name", "")
        if decision in ("discard_and_reuse", "promote") and target:
            merge_count += 1
            if namespace(extracted) != namespace(str(target)):
                violations.append((extracted, target))
    print(
        f"[audit ] merge records={merge_count} cross_namespace_violations={len(violations)}"
    )
    for extracted, target in violations:
        print(f"  VIOLATION: {extracted!r} -> {target!r}", file=sys.stderr)
    return 1 if violations else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_drift = sub.add_parser(
        "drift", help="report variant-group drift over a name dump"
    )
    p_drift.add_argument(
        "--after", required=True, help="entity-name dump (one per line)"
    )
    p_drift.add_argument("--before", help="baseline dump to compare against")
    p_drift.add_argument(
        "--show-groups", action="store_true", help="list variant groups"
    )
    p_drift.set_defaults(func=cmd_drift)

    p_audit = sub.add_parser("audit", help="assert no cross-namespace merges in a log")
    p_audit.add_argument(
        "--log", required=True, help="entity_resolution_log.jsonl path"
    )
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
