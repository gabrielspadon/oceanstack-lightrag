"""Shared value-resolution helpers for the storage-backend workspace ritual.

Nearly every ``kg/<backend>_impl.py`` constructor repeats the same
"``<BACKEND>_WORKSPACE`` env override, else the passed ``workspace``
argument, then build a workspace-prefixed namespace" sequence. The exact
semantics (whether the env value is stripped, whether an empty result falls
back to a backend-specific default, and how the log line reads) differ
per call site, so this module only resolves *values* — it never logs.
Callers keep their own `logger.info`/`logger.debug` lines so wording stays
backend-specific, and pass in whatever parameters reproduce their site's
existing behavior exactly.
"""

from __future__ import annotations

import os


def resolve_workspace_override(
    workspace: str | None,
    env_var: str,
    *,
    default: str | None = None,
    strip_env_value: bool = True,
    strip_default_check: bool = False,
) -> tuple[str, bool]:
    """Resolve the effective workspace value for a storage backend constructor.

    Priority: 1) ``env_var`` (only when non-blank after stripping) 2) the
    passed ``workspace`` argument (``None`` normalizes to ``""``) 3)
    ``default``, applied only when the resolved value is still blank and
    ``default is not None``.

    Args:
        workspace: The workspace argument/attribute passed to the
            constructor. ``None`` is treated as ``""``.
        env_var: Name of the backend's `*_WORKSPACE` environment variable.
        default: Fallback applied when the resolved value is blank. Some
            call sites (e.g. graph backends defaulting to ``"base"``) apply
            this unconditionally; others apply no default at all and instead
            fold an equivalent `or <default>` into their own post-processing
            (see qdrant_impl.py), so this stays opt-in.
        strip_env_value: Whether the env var's value is stripped before use
            once it is determined to be non-blank. Most backends strip;
            memgraph/neo4j historically do not (kept as-is here).
        strip_default_check: Whether "is the resolved value blank" (for the
            purpose of applying ``default``) is decided on the stripped
            value (memgraph/neo4j: whitespace-only counts as blank) or on
            plain truthiness (qdrant-style `or` fallback, done inline by
            callers that need this — see note above).

    Returns:
        ``(effective_workspace, overridden)`` where ``overridden`` is True
        iff the environment variable fired. Callers use it to pick their own
        log line/level.
    """
    env_value = os.environ.get(env_var)
    if env_value and env_value.strip():
        effective = env_value.strip() if strip_env_value else env_value
        overridden = True
    else:
        effective = workspace or ""
        overridden = False

    if default is not None:
        blank = not effective.strip() if strip_default_check else not effective
        if blank:
            effective = default

    return effective, overridden


def build_namespace_prefix(
    effective_workspace: str,
    namespace: str,
    *,
    separator: str = "_",
) -> str:
    """Build the workspace-prefixed namespace shared by every backend.

    Mirrors the identical tail every KV/DocStatus/Graph/VectorDB backend
    applies: prefix ``namespace`` with ``effective_workspace`` when
    non-empty, otherwise fall back to the bare ``namespace``.
    """
    if effective_workspace:
        return f"{effective_workspace}{separator}{namespace}"
    return namespace


def resolve_effective_workspace(
    workspace: str | None,
    namespace: str,
    env_var: str,
    *,
    separator: str = "_",
) -> tuple[str, str, bool]:
    """Combine `resolve_workspace_override` + `build_namespace_prefix`.

    This is the common case (redis, mongo, opensearch): env override, else
    the passed workspace, no default fallback, then a workspace-prefixed
    final namespace. Backends needing a default fallback (qdrant) or
    additional suffixing before the final namespace is built (milvus) call
    the two lower-level helpers directly instead.

    Returns ``(effective_workspace, final_namespace, overridden)``.
    """
    effective_workspace, overridden = resolve_workspace_override(workspace, env_var)
    final_namespace = build_namespace_prefix(
        effective_workspace, namespace, separator=separator
    )
    return effective_workspace, final_namespace, overridden
