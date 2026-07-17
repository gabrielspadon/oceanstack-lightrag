"""Internal document ingestion machinery for the LightRAG API.

This module holds NO HTTP surface. Greenfield serving exposes only the
immutable plane-qualified query/graph routes (``plane_routes``); the document
router factory that once lived here has been removed, and
``lightrag_server.py`` never mounts anything from this module (asserted by
``tests/test_greenfield_repository_surface.py``).

What remains is the internal ingestion machinery the pipeline and any embedded
caller drive directly: the ``DocumentManager`` (input-dir scan + supported
extensions), file/text enqueue and indexing entry points
(``pipeline_enqueue_file``, ``pipeline_index_file(s)``, ``pipeline_index_texts``),
the scan/dedup task (``run_scanning_process``), the destructive-delete task
(``background_delete_documents``), the pipeline-status concurrency primitives
(``_reserve_enqueue_slot`` / ``_release_enqueue_slot`` /
``_acquire_destructive_busy`` / ``_release_destructive_busy``), file-variant
cleanup, the chunking-config models, and the retained ``DocStatusResponse``
serialization model.
"""

import asyncio
import re
import shutil
from lightrag.utils import (
    logger,
    get_pinyin_sort_key,
    validate_workspace,
)
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from fastapi import (
    HTTPException,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import (
    FILE_EXTRACTION_SUMMARY_PREFIX,
    FULL_DOCS_FORMAT_PENDING_PARSE,
    PARSED_ARTIFACT_DIR_SUFFIXES,
    PARSED_DIR_NAME,
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_PARAGRAH,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
)
from lightrag.parser.routing import (
    FilenameParserHintError,
    chunk_strategy_key,
    encode_parse_engine,
    filename_parser_hint,
    parse_process_options,
    resolve_chunk_options,
    resolve_parser_directives,
)
from lightrag.utils import (
    generate_track_id,
    move_file_to_parsed_dir,
)
from lightrag.utils_pipeline import doc_status_value, normalize_document_file_path


# Temporary file prefix
temp_prefix = "__tmp__"
UNKNOWN_FILE_SOURCE = "unknown_source"
ARCHIVED_FILE_SUFFIX_RE = re.compile(r"_(?:\d{3}|\d{10,})$")


def normalize_file_path(file_path: str | None) -> str:
    """Normalize a document source to its canonical stored identity.

    Delegates to :func:`normalize_document_file_path` so the router and the
    pipeline agree on document identity: directory components are preserved
    (``pkg/mod.rs`` stays distinct from ``other/mod.rs``), parser hints are
    stripped, and missing sources collapse to a single non-null sentinel.
    """
    return normalize_document_file_path(file_path)


TextChunkingStrategy = Literal[
    "fixed_token",
    "recursive_character",
    "semantic_vector",
    "paragraph_semantic",
]


class _StrictChunkParams(BaseModel):
    """Base for per-strategy chunking params.

    ``strict=True`` rejects the Pydantic-v2 lax coercions that would
    otherwise let malformed requests through and fail later in the
    background chunker: bool-as-int (``true`` -> 1), numeric strings
    (``"5"`` -> 5), float-as-int.  ``extra="forbid"`` turns unknown keys
    into a 422 (replacing a hand-rolled allow-list).  ``chunk_token_size``
    is shared by every strategy; ``None`` means "not supplied — fall back
    to ``addon_params``/env default at process time".
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_token_size: Optional[int] = Field(default=None, ge=1)


class _OverlapChunkParams(_StrictChunkParams):
    chunk_overlap_token_size: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _overlap_lt_size(self) -> "_OverlapChunkParams":
        # Only enforceable when BOTH are explicit; when chunk_token_size
        # is None the effective size is resolved from addon_params/env at
        # process time and can't be compared against here.
        if (
            self.chunk_token_size is not None
            and self.chunk_overlap_token_size is not None
            and self.chunk_overlap_token_size >= self.chunk_token_size
        ):
            raise ValueError("chunk_overlap_token_size must be < chunk_token_size")
        return self


class FixedTokenChunkParams(_OverlapChunkParams):
    split_by_character: Optional[str] = None
    split_by_character_only: Optional[bool] = None


class RecursiveCharacterChunkParams(_OverlapChunkParams):
    separators: Optional[list[str]] = None


class ParagraphSemanticChunkParams(_OverlapChunkParams):
    # Drop the trailing reference section before chunking. ``None`` means
    # "not supplied — inherit the addon_params/env default at process time".
    # Detection-tuning knobs (tail window / heading prefixes) are env-only and
    # read live by the chunker, so they are intentionally not exposed here.
    drop_references: Optional[bool] = None


class SemanticVectorChunkParams(_StrictChunkParams):
    # Enum verified against the installed langchain_experimental
    # (text_splitter.py ``BreakpointThresholdType``), not from memory.
    breakpoint_threshold_type: Optional[
        Literal["percentile", "standard_deviation", "interquartile", "gradient"]
    ] = None
    # A strict ``float`` field still accepts an ``int`` (e.g. JSON ``95``) and
    # widens it losslessly to ``95.0`` — strict only rejects ``str`` / ``bool``
    # here, which is exactly what we want. Do NOT relax strict (that would let
    # numeric strings through) or switch to ``int | float`` (that would stop
    # normalizing ints to float). Locked by tests in test_document_routes_chunking.
    breakpoint_threshold_amount: Optional[float] = None
    buffer_size: Optional[int] = Field(default=None, ge=1)
    sentence_split_regex: Optional[str] = None

    @field_validator("sentence_split_regex")
    @classmethod
    def _valid_sentence_split_regex(cls, v: Optional[str]) -> Optional[str]:
        # The value is fed to LangChain's SemanticChunker and compiled during
        # split_text. A malformed pattern (e.g. "(") would only blow up in the
        # background, so compile it here to reject synchronously (HTTP 422).
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(
                f"sentence_split_regex is not a valid regular expression: {exc}"
            ) from exc
        return v

    @model_validator(mode="after")
    def _amount_in_range(self) -> "SemanticVectorChunkParams":
        amt = self.breakpoint_threshold_amount
        if amt is None:
            return self
        # ``> 0`` is type-independent (every threshold type wants a positive
        # magnitude), so it is safe to enforce at parse time.
        if amt <= 0:
            raise ValueError("breakpoint_threshold_amount must be > 0")
        # The ``(0, 100]`` ceiling is percentile/gradient-specific (those feed
        # np.percentile, which requires q in [0, 100]). It depends on the
        # threshold TYPE, so only enforce it here when the type is supplied in
        # the SAME request. When the type is omitted, the effective type is
        # resolved from addon_params/env later — assuming "percentile" here
        # would wrongly 422 a partial override that inherits
        # standard_deviation/interquartile (which allow amounts > 100). The
        # ceiling against the merged type is applied by
        # ``_validate_effective_semantic_amount`` in ``_resolve_text_chunking``.
        if self.breakpoint_threshold_type in ("percentile", "gradient") and amt > 100:
            raise ValueError(
                "breakpoint_threshold_amount must be within (0, 100] "
                "for percentile/gradient"
            )
        return self


_CHUNKING_PARAMS_MODEL: dict[str, type[_StrictChunkParams]] = {
    "fixed_token": FixedTokenChunkParams,
    "recursive_character": RecursiveCharacterChunkParams,
    "semantic_vector": SemanticVectorChunkParams,
    "paragraph_semantic": ParagraphSemanticChunkParams,
}


class TextChunkingConfig(BaseModel):
    """Chunking strategy + strategy-specific params for a text insert.

    Validation is delegated to the per-strategy typed model so unknown
    keys, wrong types, and out-of-range values all raise synchronously
    during request parsing (HTTP 422) — never later in the background
    indexing task, where the HTTP response has already been sent.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: TextChunkingStrategy = "fixed_token"
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "TextChunkingConfig":
        typed = _CHUNKING_PARAMS_MODEL[self.strategy].model_validate(self.params)
        # Normalize down to exactly the keys the caller supplied with a real
        # value (validated + coerced) so the enqueue-time merge overrides only
        # what was set. ``exclude_none`` additionally drops explicit nulls:
        # every param field means "inherit the addon_params/env default" when
        # None, so an explicit ``"chunk_token_size": null`` must NOT be merged
        # over the resolved default — otherwise the route would 200 and the
        # background chunker would do ``int(None)`` and fail the document.
        self.params = typed.model_dump(exclude_unset=True, exclude_none=True)
        return self


# doc_status.metadata keys that are internal pipeline bookkeeping and must
# never reach the frontend. smartheading_llm_cache_ids is a deletion-time
# LLM-cache purge list (written by the parse pipeline at pipeline.py:1748,
# consumed only by adelete_by_doc_id).
_INTERNAL_METADATA_KEYS = frozenset({"smartheading_llm_cache_ids"})


class DocStatusResponse(BaseModel):
    id: str = Field(description="Document identifier")
    content_summary: str = Field(description="Summary of document content")
    content_length: int = Field(description="Length of document content in characters")
    status: DocStatus = Field(description="Current processing status")
    created_at: str = Field(description="Creation timestamp (ISO format string)")
    updated_at: str = Field(description="Last update timestamp (ISO format string)")
    track_id: Optional[str] = Field(
        default=None, description="Tracking ID for monitoring progress"
    )
    chunks_count: Optional[int] = Field(
        default=None, description="Number of chunks the document was split into"
    )
    error_msg: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata about the document"
    )
    file_path: str = Field(description="Path to the document file")

    @field_validator("metadata", mode="after")
    @classmethod
    def _strip_internal_metadata(
        cls, metadata: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Never expose internal pipeline bookkeeping to API clients."""
        if not isinstance(metadata, dict):
            return metadata  # None / (defensively) non-dict pass through
        if not _INTERNAL_METADATA_KEYS.intersection(metadata):
            return metadata  # common case: nothing to strip, no copy
        # Copy-then-strip — the source DocProcessingStatus.metadata is shared
        # with the deletion path and carry-over; mutating it in place would
        # remove the field adelete_by_doc_id needs.
        return {k: v for k, v in metadata.items() if k not in _INTERNAL_METADATA_KEYS}

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123456",
                "content_summary": "Research paper on machine learning",
                "content_length": 15240,
                "status": "processed",
                "created_at": "2025-03-31T12:34:56",
                "updated_at": "2025-03-31T12:35:30",
                "track_id": "upload_20250729_170612_abc123",
                "chunks_count": 12,
                "error": None,
                "metadata": {"author": "John Doe", "year": 2025},
                "file_path": "research_paper.pdf",
            }
        }
    )


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        workspace: str = "",  # New parameter for workspace isolation
    ):
        # Reject path traversal before using workspace in the upload path
        validate_workspace(workspace)
        # Store the base input directory and workspace
        self.base_input_dir = Path(input_dir)
        self.workspace = workspace
        self.indexed_files = set()

        # Create workspace-specific input directory
        # If workspace is provided, create a subdirectory for data isolation
        if workspace:
            self.input_dir = self.base_input_dir / workspace
        else:
            self.input_dir = self.base_input_dir

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supported_extensions(self) -> tuple:
        """Suffixes accepted for an unhinted filename, derived live.

        A suffix is advertised only when it is *routable without extra
        directives*: the engine that ``resolve_file_parser_engine`` picks for
        a bare ``x.<suffix>`` (filename hint absent; ``LIGHTRAG_PARSER``
        rules + default apply) must itself support the suffix. This keeps
        "uploadable" aligned with "will actually parse": e.g. mineru's
        ``png`` joins only when its endpoint is configured AND a routing
        rule (or per-file hint, see ``is_supported_file``) sends pngs to it
        — otherwise the default ``legacy`` engine would fail the suffix gate
        at the parse stage. A default deployment equals the local engines'
        (legacy ∪ native) types; no hardcoded list to keep in sync.
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            resolve_file_parser_engine,
        )

        out = []
        for s in sorted(available_engine_suffixes()):
            engine = resolve_file_parser_engine(f"x.{s}")
            if parser_engine_supports_suffix(engine, s):
                out.append(f".{s}")
        return tuple(out)

    def scan_directory_for_new_files(self) -> List[Path]:
        """Scan input directory for new, routable files.

        Globs over every *available* engine suffix (capability surface, so a
        hint-carrying file like ``img.[mineru].png`` is discoverable even
        when bare ``.png`` is not advertised), then keeps only files whose
        resolved engine actually supports them (``is_supported_file``).
        """
        from lightrag.parser.registry import available_engine_suffixes
        from lightrag.parser.routing import FilenameParserHintError

        new_files = []
        for s in sorted(available_engine_suffixes()):
            ext = f".{s}"
            logger.debug(f"Scanning for {ext} files in {self.input_dir}")
            for file_path in self.input_dir.glob(f"*{ext}"):
                if file_path in self.indexed_files:
                    continue
                try:
                    if not self.is_supported_file(file_path.name):
                        continue
                except FilenameParserHintError:
                    # Malformed hint: pass the file through — the enqueue
                    # path reports a detailed error document, instead of the
                    # scan silently ignoring the user's file.
                    pass
                new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """True when THIS filename routes to an engine that can parse it.

        Resolves the engine for the concrete name — so a per-file hint
        (``img.[mineru].png``) is honoured — and checks the resolved engine
        supports the suffix. A bare suffix that would fall through to the
        default ``legacy`` engine is rejected here instead of failing later
        at the parse worker's suffix gate.

        Raises :class:`FilenameParserHintError` for a malformed hint —
        callers surface it (upload → HTTP 400 with the detailed message;
        scan passes the file through so enqueue emits an error document).
        """
        from lightrag.parser.routing import (
            parser_engine_supports_suffix,
            parser_suffix,
            resolve_file_parser_engine,
        )

        engine = resolve_file_parser_engine(filename)
        return parser_engine_supports_suffix(engine, parser_suffix(filename))


def validate_file_path_security(file_path_str: str, base_dir: Path) -> Optional[Path]:
    """
    Validate file path security to prevent Path Traversal attacks.

    Args:
        file_path_str: The file path string to validate
        base_dir: The base directory that the file must be within

    Returns:
        Path: Safe file path if valid, None if unsafe or invalid
    """
    if not file_path_str or not file_path_str.strip():
        return None

    try:
        # Clean the file path string
        clean_path_str = file_path_str.strip()

        # Check for obvious path traversal patterns before processing
        # This catches both Unix (..) and Windows (..\) style traversals
        if ".." in clean_path_str:
            # Additional check for Windows-style backslash traversal
            if (
                "\\..\\" in clean_path_str
                or clean_path_str.startswith("..\\")
                or clean_path_str.endswith("\\..")
            ):
                # logger.warning(
                #     f"Security violation: Windows path traversal attempt detected - {file_path_str}"
                # )
                return None

        # Normalize path separators (convert backslashes to forward slashes)
        # This helps handle Windows-style paths on Unix systems
        normalized_path = clean_path_str.replace("\\", "/")

        # Create path object and resolve it (handles symlinks and relative paths)
        candidate_path = (base_dir / normalized_path).resolve()
        base_dir_resolved = base_dir.resolve()

        # Check if the resolved path is within the base directory
        if not candidate_path.is_relative_to(base_dir_resolved):
            # logger.warning(
            #     f"Security violation: Path traversal attempt detected - {file_path_str}"
            # )
            return None

        return candidate_path

    except (OSError, ValueError, Exception) as e:
        logger.warning(f"Invalid file path detected: {file_path_str} - {str(e)}")
        return None


async def get_existing_doc_by_file_path_candidates(
    doc_status: Any, file_path: Path | str
) -> dict[str, Any] | None:
    """Find an existing document by canonical basename."""
    basename = normalize_file_path(str(file_path))
    if basename == UNKNOWN_FILE_SOURCE:
        return None
    match = await doc_status.get_doc_by_file_basename(basename)
    if not match:
        return None
    _, existing_doc_data = match
    return existing_doc_data


async def _reserve_enqueue_slot(rag: LightRAG) -> bool:
    """Atomically check exclusive-writer state and reserve a
    pending-enqueue slot.

    Concurrent enqueues are permitted while the processing loop is
    running — the loop is notified via ``request_pending`` and picks up
    newly-enqueued docs after its current batch.  This includes the
    scan task's processing phase: once classification is done, the
    scan transitions to driving the processing pipeline like any
    other enqueuer, and uploads can land alongside it.

    Two states block new uploads/inserts:

    - ``scanning_exclusive``: scan task is in its CLASSIFICATION
      phase — reading doc_status to classify files (PROCESSED →
      archive, FAILED-without-full_docs → retry-as-new, etc.) and
      possibly deleting stale stubs.  Concurrent enqueue would race
      against scan's reads / stub deletions.  ``scanning`` alone
      (the processing phase) does NOT block uploads.
    - ``destructive_busy``: a /documents/clear or per-doc delete is in
      flight.  These DROP storages and remove input files; an enqueue
      accepted in this window would write to a storage that is being
      torn down and silently lose the document after the client saw
      success.

    ``pending_enqueues`` is incremented so the scan endpoint can refuse
    while bg tasks are mid-enqueue.  The counter does NOT gate
    ``apipeline_process_enqueue_documents`` — concurrent processing is
    explicitly allowed and is what makes "upload while pipeline is
    busy" possible.

    A workspace whose ``pipeline_status`` has never been initialised
    (mocked test rigs) is treated as idle; no slot is reserved.

    Returns:
        True when a slot was reserved (caller MUST pair with
        ``_release_enqueue_slot``); False when pipeline_status is not
        bootstrapped.

    Raises:
        HTTPException(409): when
            ``pipeline_status['scanning_exclusive']`` or
            ``pipeline_status['destructive_busy']`` is set.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return False
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    async with pipeline_status_lock:
        if pipeline_status.get("scanning_exclusive"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Document scan is classifying files. "
                    "Wait for the classification phase to finish before "
                    "submitting new work."
                ),
            )
        if pipeline_status.get("destructive_busy"):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Pipeline is clearing or deleting documents. "
                    "Wait for the running job to finish before submitting "
                    "new work."
                ),
            )
        pipeline_status["pending_enqueues"] = (
            pipeline_status.get("pending_enqueues", 0) + 1
        )
    return True


async def _acquire_destructive_busy(rag: LightRAG) -> tuple[bool, str | None]:
    """Atomically reserve the destructive busy slot for ``/documents/clear``
    or ``/documents/delete_document``.

    Both jobs DROP storages and (for clear) remove input files.  They
    must serialise against:

    - any other ``busy`` work (processing loop, another destructive job),
    - an in-flight ``scanning`` task that reads/writes doc_status and
      INPUT/, and
    - any ``pending_enqueues`` reservation whose bg task has not yet
      written to doc_status — accepting the destructive job in that
      window would drop storages while the enqueue is mid-write,
      losing a document the client already saw success for.

    All three checks happen inside a single ``pipeline_status_lock``
    critical section together with the flag write, so a concurrent
    enqueue/scan reservation cannot squeeze past us.

    Caller is responsible for clearing both flags in its finally block.

    Returns:
        (acquired, reason).  ``acquired=True`` and ``reason=None`` on
        success.  ``acquired=False`` with a human-readable ``reason``
        when another writer has the lock; the caller surfaces this to
        the client (HTTP 200 with status="busy" for these endpoints).

        For test rigs where ``pipeline_status`` was never bootstrapped,
        returns (True, None) — there is nothing to coordinate against.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return True, None
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    async with pipeline_status_lock:
        if pipeline_status.get("busy"):
            return False, "Pipeline is busy with another operation."
        if pipeline_status.get("scanning"):
            return False, (
                "Document scan is in progress. "
                "Wait for the scan to complete before clearing or deleting."
            )
        if pipeline_status.get("pending_enqueues", 0) > 0:
            return False, (
                "Document upload/insert is being enqueued. "
                "Wait for in-flight work to complete before clearing or "
                "deleting."
            )
        pipeline_status["busy"] = True
        pipeline_status["destructive_busy"] = True
    return True, None


async def _release_destructive_busy(rag: LightRAG) -> None:
    """Release the destructive busy slot acquired by
    ``_acquire_destructive_busy``.  Never raises.

    Distinct from ``_release_enqueue_slot``: that helper clears
    ``pending_enqueues`` (the upload/insert reservation), this one
    clears ``busy + destructive_busy`` (the clear/delete reservation).
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    async with pipeline_status_lock:
        pipeline_status["busy"] = False
        pipeline_status["destructive_busy"] = False


async def _release_enqueue_slot(rag: LightRAG) -> None:
    """Release a slot reserved by ``_reserve_enqueue_slot``.

    Pure decrement; the bg task itself drives processing by calling
    ``apipeline_process_enqueue_documents`` after enqueue (the call is
    a cheap no-op when the loop is already busy — it just sets
    ``request_pending``).  Drain coordination across sibling bg tasks
    is unnecessary in the new contract: each task triggers processing
    independently and the loop's request_pending mechanism collapses
    duplicate triggers safely.

    Decrement is clamped at 0 so a stray release (e.g. from a workspace
    whose reservation returned False but whose bg task wrapper still
    calls release) is harmless.  Never raises.
    """
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        return
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )
    async with pipeline_status_lock:
        current = pipeline_status.get("pending_enqueues", 0)
        if current > 0:
            pipeline_status["pending_enqueues"] = current - 1


def find_existing_file_by_file_path(input_dir: Path, file_path: str) -> Path | None:
    """Find an input-dir file whose canonical basename matches ``file_path``.

    Callers pass the stored canonical ``file_path`` (already hint-stripped);
    on-disk filenames are normalized before comparison so a hint-bearing
    variant on disk still matches a canonical stored ``file_path``.
    """
    if not file_path or file_path == UNKNOWN_FILE_SOURCE:
        return None
    try:
        for candidate in input_dir.iterdir():
            if not candidate.is_file():
                continue
            if normalize_file_path(candidate.name) == file_path:
                return candidate
    except FileNotFoundError:
        return None
    return None


def canonicalize_archived_file_variant_basename(
    file_path: Path | str, *, strip_archive_suffix: bool = False
) -> str:
    """Canonical basename for original files and numbered archive variants."""
    name = Path(file_path).name
    path = Path(name)
    stem = (
        ARCHIVED_FILE_SUFFIX_RE.sub("", path.stem)
        if strip_archive_suffix
        else path.stem
    )
    return normalize_file_path(f"{stem}{path.suffix}")


def _file_path_for_parsed_artifact_dir(dir_name: str) -> str | None:
    """Return the canonical source basename for a parser artifact dir.

    Recognized layouts (suffix list in
    :data:`lightrag.constants.PARSED_ARTIFACT_DIR_SUFFIXES`):

    - ``<basename>.parsed[_NNN]/``        — sidecar output (every engine)
    - ``<basename>.mineru_raw[_NNN]/``    — MinerU preserved raw bundle
    - ``<basename>.docling_raw[_NNN]/``   — Docling preserved raw bundle

    Raw bundles are preserved across re-parses for cache reuse and on-demand
    diagnostics; they are cleaned only when the user deletes the document
    with ``delete_file=True`` so the raw artifacts and source file go away
    together.
    """
    stripped = ARCHIVED_FILE_SUFFIX_RE.sub("", dir_name)
    for suffix in PARSED_ARTIFACT_DIR_SUFFIXES:
        if stripped.endswith(suffix):
            basename = stripped[: -len(suffix)]
            if basename:
                return normalize_file_path(basename)
    return None


def delete_file_variants_by_file_path(
    input_dir: Path,
    file_path: str | None,
) -> tuple[list[str], list[str]]:
    """Delete input/__parsed__ source files matching a canonical ``file_path``."""
    if not file_path:
        return [], []
    canonical = normalize_file_path(file_path)
    if canonical == UNKNOWN_FILE_SOURCE:
        return [], []
    canonical_names = {canonical}

    deleted_files: list[str] = []
    errors: list[str] = []
    candidate_dirs = [input_dir, input_dir / PARSED_DIR_NAME]
    input_dir_resolved = input_dir.resolve()

    for candidate_dir in candidate_dirs:
        try:
            candidates = list(candidate_dir.iterdir())
        except FileNotFoundError:
            continue
        except Exception as e:
            errors.append(f"Failed to scan {candidate_dir}: {e}")
            continue

        in_parsed_dir = candidate_dir.name == PARSED_DIR_NAME
        for candidate in candidates:
            if candidate.is_file():
                if (
                    canonicalize_archived_file_variant_basename(
                        candidate.name,
                        strip_archive_suffix=in_parsed_dir,
                    )
                    not in canonical_names
                ):
                    continue

                safe_candidate = validate_file_path_security(
                    candidate.name, candidate_dir
                )
                if safe_candidate is None:
                    errors.append(f"Unsafe file path skipped: {candidate.name}")
                    continue

                try:
                    safe_candidate.unlink()
                    deleted_files.append(
                        str(safe_candidate.relative_to(input_dir_resolved))
                    )
                except Exception as e:
                    errors.append(f"Failed to delete {candidate.name}: {e}")
                continue

            if in_parsed_dir and candidate.is_dir():
                canonical_for_dir = _file_path_for_parsed_artifact_dir(candidate.name)
                if (
                    canonical_for_dir is None
                    or canonical_for_dir not in canonical_names
                ):
                    continue

                safe_candidate = validate_file_path_security(
                    candidate.name, candidate_dir
                )
                if safe_candidate is None:
                    errors.append(f"Unsafe artifact dir skipped: {candidate.name}")
                    continue

                try:
                    shutil.rmtree(safe_candidate)
                    deleted_files.append(
                        str(safe_candidate.relative_to(input_dir_resolved))
                    )
                except Exception as e:
                    errors.append(
                        f"Failed to delete artifact dir {candidate.name}: {e}"
                    )

    return deleted_files, errors


async def record_scan_warning(rag: LightRAG, message: str) -> None:
    logger.warning(message)
    try:
        from lightrag.kg import shared_storage

        if not getattr(shared_storage, "_initialized", False):
            return

        workspace = getattr(rag, "workspace", "")
        pipeline_status = await shared_storage.get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = shared_storage.get_namespace_lock(
            "pipeline_status", workspace=workspace
        )
        async with pipeline_status_lock:
            pipeline_status["latest_message"] = message
            pipeline_status["history_messages"].append(message)
    except Exception:
        pass


# Legacy text extractors moved to lightrag.parser.legacy.extractors; the
# legacy engine now extracts at the worker stage (LegacyParser), not here.


async def pipeline_enqueue_file(
    rag: LightRAG,
    file_path: Path,
    track_id: str = None,
    from_scan: bool = False,
) -> tuple[bool, str]:
    """Add a file to the queue for processing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID, if not provided will be generated
        from_scan: True only when invoked by the scan-owned background task,
            which already holds ``pipeline_status["scanning"]``.  Forwarded to
            ``apipeline_enqueue_documents`` so the scan can enqueue the files
            it just discovered without tripping the scanning guard there.
    Returns:
        tuple: (success: bool, track_id: str)
    """

    # Generate track_id if not provided
    if track_id is None:
        track_id = generate_track_id("unknown")

    try:
        file_size = 0

        # Get file size for error reporting
        try:
            stat = await asyncio.to_thread(file_path.stat)
            file_size = stat.st_size
        except Exception:
            file_size = 0

        try:
            directives = resolve_parser_directives(file_path)
        except FilenameParserHintError as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                    + "Filename hint error",
                    "original_error": str(e),
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Invalid filename hint in {file_path.name}: {e}"
            )
            return False, track_id

        extraction_engine = directives.engine
        process_options = directives.process_options
        api_process_options = process_options or PROCESS_OPTION_CHUNK_FIXED

        # Overlay any per-file chunk parameters (from the filename hint or a
        # LIGHTRAG_PARSER rule) onto the active strategy's chunk_options so the
        # parse worker chunks this document with them. Absent params keep the
        # legacy path (chunk_options built at enqueue time from addon_params).
        hint_chunk_options = None
        active_strategy = parse_process_options(api_process_options).chunking
        hint_chunk_params = directives.chunk_params.get(active_strategy)
        if hint_chunk_params:
            try:
                strategy_key = chunk_strategy_key(api_process_options)
                hint_chunk_options = resolve_chunk_options(
                    rag.addon_params, process_options=api_process_options
                )
                hint_chunk_options[strategy_key].update(hint_chunk_params)
                _validate_effective_chunk_overlap(
                    hint_chunk_options, strategy_key, strategy_key
                )
            except ValueError as e:
                error_files = [
                    {
                        "file_path": str(file_path.name),
                        "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                        + "Chunk parameter error",
                        "original_error": str(e),
                        "file_size": file_size,
                    }
                ]
                await rag.apipeline_enqueue_error_documents(error_files, track_id)
                logger.error(
                    f"[File Extraction]Invalid chunk parameters in "
                    f"{file_path.name}: {e}"
                )
                return False, track_id
        # All engines defer parsing to the worker stage: the file is already
        # saved on disk, so we enqueue PENDING_PARSE with the chosen engine.
        # Legacy now extracts at the worker (LegacyParser) instead of eagerly
        # here, so every engine shares one ingestion path.
        # Encode any per-file engine params into the parse_engine field
        # (e.g. "mineru(page_range=1-3,language=en)") so they ride the existing
        # persisted column to the parse worker. Bare engine when there are none.
        parse_engine_field = encode_parse_engine(
            extraction_engine, directives.engine_params
        )
        try:
            # Uploaded files land flat in the input dir, so document identity
            # is the file's own name; library callers of
            # apipeline_enqueue_documents keep their caller-owned relative
            # paths instead.
            enqueue_kwargs = {
                "file_paths": file_path.name,
                "track_id": track_id,
                "docs_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                "parse_engine": parse_engine_field,
                "process_options": api_process_options,
                "from_scan": from_scan,
            }
            if hint_chunk_options is not None:
                enqueue_kwargs["chunk_options"] = hint_chunk_options
            enqueue_result = await rag.apipeline_enqueue_documents("", **enqueue_kwargs)
            if enqueue_result is None:
                try:
                    await move_file_to_parsed_dir(file_path)
                except Exception as move_error:
                    logger.error(
                        f"Failed to move duplicate file {file_path.name} to {PARSED_DIR_NAME} directory: {move_error}"
                    )
                return False, track_id
            logger.info(
                f"[File Extraction]Deferred {file_path.name} to {extraction_engine} parser"
            )
            return True, track_id
        except Exception as e:
            error_files = [
                {
                    "file_path": str(file_path.name),
                    "error_description": FILE_EXTRACTION_SUMMARY_PREFIX
                    + "Parser enqueue error",
                    "original_error": f"Failed to enqueue file for parser: {str(e)}",
                    "file_size": file_size,
                }
            ]
            await rag.apipeline_enqueue_error_documents(error_files, track_id)
            logger.error(
                f"[File Extraction]Error enqueuing {file_path.name} for {extraction_engine}: {str(e)}"
            )
            return False, track_id

    except Exception as e:
        # Catch-all for any unexpected errors
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            file_size = 0

        error_files = [
            {
                "file_path": str(file_path.name),
                "error_description": "Unexpected processing error",
                "original_error": f"Unexpected error: {str(e)}",
                "file_size": file_size,
            }
        ]
        await rag.apipeline_enqueue_error_documents(error_files, track_id)
        logger.error(f"Enqueuing file {file_path.name} error: {str(e)}")
        logger.error(traceback.format_exc())
        return False, track_id
    finally:
        if file_path.name.startswith(temp_prefix):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")


async def pipeline_index_file(rag: LightRAG, file_path: Path, track_id: str = None):
    """Index a file with track_id

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID
    """
    try:
        success, _ = await pipeline_enqueue_file(rag, file_path, track_id)
        if success:
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error indexing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_index_files(
    rag: LightRAG,
    file_paths: List[Path],
    track_id: str = None,
    from_scan: bool = False,
):
    """Index multiple files sequentially to avoid high CPU load

    Args:
        rag: LightRAG instance
        file_paths: Paths to the files to index
        track_id: Optional tracking ID to pass to all files
        from_scan: True only when invoked by the scan-owned background task.
            Forwarded to ``pipeline_enqueue_file`` so the per-file enqueue
            calls bypass the scanning guard inside
            ``apipeline_enqueue_documents`` (whose ``scanning`` flag the
            scan task itself owns).
    """
    if not file_paths:
        return
    try:
        enqueued = False

        # Use get_pinyin_sort_key for Chinese pinyin sorting
        sorted_file_paths = sorted(
            file_paths, key=lambda p: get_pinyin_sort_key(str(p))
        )

        # Process files sequentially with track_id
        for file_path in sorted_file_paths:
            success, _ = await pipeline_enqueue_file(
                rag,
                file_path,
                track_id,
                from_scan=from_scan,
            )
            if success:
                enqueued = True

        # Process the queue only if at least one file was successfully enqueued
        if enqueued:
            await rag.apipeline_process_enqueue_documents()
    except Exception as e:
        logger.error(f"Error indexing files: {str(e)}")
        logger.error(traceback.format_exc())


_STRATEGY_TO_PROCESS_OPTION: Dict[str, str] = {
    "fixed_token": PROCESS_OPTION_CHUNK_FIXED,
    "recursive_character": PROCESS_OPTION_CHUNK_RECURSIVE,
    "semantic_vector": PROCESS_OPTION_CHUNK_VECTOR,
    "paragraph_semantic": PROCESS_OPTION_CHUNK_PARAGRAH,
}


def _resolve_text_chunking(
    chunking: Optional[TextChunkingConfig], rag: LightRAG
) -> tuple[str, dict]:
    """Freeze a ``chunking`` request into ``(process_options, chunk_options)``.

    When ``chunking`` is ``None`` this reproduces today's behavior exactly:
    fixed-token strategy with the snapshot built from
    ``rag.addon_params['chunker']``.

    Otherwise the validated, strategy-specific params are merged into the
    selected strategy's sub-dict. ``chunk_token_size`` rides along inside
    ``params`` like any other key — every strategy (F included, after the
    ``process_single_document`` cleanup) reads its size from its own
    sub-dict, with the top-level snapshot value as the shared fallback.

    Raises:
        ValueError: when the request lowers ``chunk_token_size`` below the
            *effective* ``chunk_overlap_token_size``.  The overlap is often
            inherited from ``addon_params``/env (the overlay fills
            ``fixed_token``/``recursive_character``/``paragraph_semantic``
            overlap with ``CHUNK_*_OVERLAP_SIZE`` / ``CHUNK_OVERLAP_SIZE``),
            so this can only be checked here against the resolved snapshot,
            not in the request model.  Callers on the request path invoke
            this synchronously so the failure surfaces as HTTP 422 before any
            background work is scheduled.
    """
    if chunking is None:
        # No request-driven config: reproduce today's behavior verbatim,
        # including not introducing new validation on the default path.
        process_options = PROCESS_OPTION_CHUNK_FIXED
        return process_options, resolve_chunk_options(
            rag.addon_params, process_options=process_options
        )

    process_options = _STRATEGY_TO_PROCESS_OPTION[chunking.strategy]
    chunk_options = resolve_chunk_options(
        rag.addon_params, process_options=process_options
    )
    strategy_key = chunk_strategy_key(process_options)
    chunk_options[strategy_key].update(chunking.params)
    _validate_effective_chunk_overlap(chunk_options, strategy_key, chunking.strategy)
    _validate_effective_semantic_amount(chunk_options, strategy_key)
    return process_options, chunk_options


def _validate_effective_chunk_overlap(
    chunk_options: dict, strategy_key: str, strategy_name: str
) -> None:
    """Reject a resolved snapshot whose overlap is >= its chunk size.

    Operates on the fully-resolved ``chunk_options`` so it catches the case
    the request model cannot: ``chunk_token_size`` supplied in the request
    while ``chunk_overlap_token_size`` is inherited from addon_params/env
    (e.g. ``chunk_token_size=50`` with the default overlap ``100``).  The
    effective size is the strategy sub-dict value, falling back to the
    top-level snapshot size; the effective overlap is the sub-dict value
    (``semantic_vector`` carries none, so it is skipped).
    """
    sub = chunk_options.get(strategy_key) or {}
    # Fixed-token delimiter-only mode (split_by_character set AND
    # split_by_character_only=True) never applies overlap:
    # chunking_by_token_size only validates each delimiter segment against
    # chunk_token_size and raises on an oversize segment — the overlap field
    # is unused. Enforcing overlap < size there would wrongly 422 a valid
    # request such as paragraph splitting with a small chunk_token_size.
    # (split_by_character_only is itself a no-op when split_by_character is
    # falsy, so both must be effective for overlap to be skipped.)
    if (
        strategy_key == "fixed_token"
        and sub.get("split_by_character")
        and sub.get("split_by_character_only")
    ):
        return
    overlap = sub.get("chunk_overlap_token_size")
    if overlap is None:
        return
    size = sub.get("chunk_token_size")
    if size is None:
        size = chunk_options.get("chunk_token_size")
    if size is not None and overlap >= size:
        raise ValueError(
            f"chunking for strategy '{strategy_name}': effective "
            f"chunk_overlap_token_size ({overlap}) must be < chunk_token_size "
            f"({size}). The overlap is inherited from addon_params/env when "
            f"not set in the request; raise chunk_token_size or lower "
            f"chunk_overlap_token_size."
        )


def _validate_effective_semantic_amount(chunk_options: dict, strategy_key: str) -> None:
    """Reject a resolved semantic_vector snapshot whose breakpoint amount
    exceeds the percentile/gradient ceiling.

    Uses the *effective* ``breakpoint_threshold_type`` from the merged
    snapshot — the request model cannot, because the type may be inherited
    from ``addon_params``/``CHUNK_V_BREAKPOINT_THRESHOLD_TYPE`` while the
    request overrides only ``breakpoint_threshold_amount``. ``percentile`` /
    ``gradient`` feed ``np.percentile`` (q must be in ``[0, 100]``);
    ``standard_deviation`` / ``interquartile`` are multipliers with no upper
    bound, so a request amount > 100 is valid for them.
    """
    if strategy_key != "semantic_vector":
        return
    sub = chunk_options.get(strategy_key) or {}
    amt = sub.get("breakpoint_threshold_amount")
    if amt is None:
        return
    kind = sub.get("breakpoint_threshold_type") or "percentile"
    if kind in ("percentile", "gradient") and amt > 100:
        raise ValueError(
            f"chunking for strategy 'semantic_vector': "
            f"breakpoint_threshold_amount ({amt}) must be within (0, 100] for "
            f"breakpoint_threshold_type '{kind}'. The type is inherited from "
            f"addon_params/env when not set in the request."
        )


async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    file_sources: List[str] = None,
    track_id: str = None,
    chunking: Optional[TextChunkingConfig] = None,
):
    """Index a list of texts with track_id

    Args:
        rag: LightRAG instance
        texts: The texts to index
        file_sources: Sources of the texts
        track_id: Optional tracking ID
        chunking: Optional chunking strategy + params (already validated by
            the request model); when None, default fixed-token chunking is used
    """
    if not texts:
        return

    if not file_sources or len(file_sources) != len(texts):
        raise ValueError("A valid file source is required for each text")

    normalized_file_sources = [normalize_file_path(source) for source in file_sources]
    if any(source == UNKNOWN_FILE_SOURCE for source in normalized_file_sources):
        raise ValueError("A valid file source is required for each text")
    if len(set(normalized_file_sources)) != len(normalized_file_sources):
        raise ValueError("File sources must be unique by filename")

    process_options, chunk_options = _resolve_text_chunking(chunking, rag)
    await rag.apipeline_enqueue_documents(
        input=texts,
        file_paths=normalized_file_sources,
        track_id=track_id,
        process_options=process_options,
        chunk_options=chunk_options,
    )
    await rag.apipeline_process_enqueue_documents()


async def run_scanning_process(
    rag: LightRAG, doc_manager: DocumentManager, track_id: str = None
):
    """Background task to scan and index documents

    Args:
        rag: LightRAG instance
        doc_manager: DocumentManager instance
        track_id: Optional tracking ID to pass to all scanned files
    """
    # The scan endpoint set ``scanning=True`` AND
    # ``scanning_exclusive=True`` synchronously before scheduling this
    # task.  ``scanning`` covers the whole lifecycle (refuses
    # overlapping scans); ``scanning_exclusive`` covers only the
    # classification phase below — we clear it before invoking
    # pipeline_index_files so concurrent uploads can land while the
    # scan-driven processing finishes.  Both MUST be cleared in
    # finally so subsequent uploads / scans can proceed even if the
    # body raises.  When pipeline_status is not initialised (mocked
    # test rigs), the flags were never set so there's nothing to
    # clear — track that here to skip the namespace fetch.
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    pipeline_status = None
    pipeline_status_lock = None
    try:
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )
    except PipelineNotInitializedError:
        pass

    try:
        new_files = doc_manager.scan_directory_for_new_files()
        total_files = len(new_files)
        logger.info(f"Found {total_files} files to index.")

        if new_files:
            # Group canonical-equivalent files so we can prefer hint-bearing
            # variants over plain ones. Within each group sort order is
            # preserved as a deterministic tiebreaker.
            files_by_canonical_name: dict[str, list[Path]] = {}
            for file_path in sorted(
                new_files, key=lambda p: get_pinyin_sort_key(str(p))
            ):
                # Scanned files enqueue under their basename identity
                # (pipeline_enqueue_file passes ``file_path.name``), so
                # canonical grouping keys on the name, not the scan path.
                canonical_name = normalize_file_path(file_path.name)
                files_by_canonical_name.setdefault(canonical_name, []).append(file_path)

            unique_files: list[Path] = []
            for canonical_name, group in files_by_canonical_name.items():
                # Prefer the first file carrying a supported parser hint so
                # the user's explicit engine choice wins over plain variants;
                # otherwise fall back to the first sorted entry.
                chosen = next(
                    (f for f in group if filename_parser_hint(f.name) is not None),
                    group[0],
                )
                unique_files.append(chosen)
                for duplicate in group:
                    if duplicate is chosen:
                        continue
                    warning = (
                        "Skipping duplicate file in scan batch: "
                        f"{duplicate.name} duplicates {chosen.name} "
                        f"(canonical: {canonical_name})"
                    )
                    await record_scan_warning(rag, warning)
                    try:
                        await move_file_to_parsed_dir(duplicate)
                    except Exception as move_error:
                        logger.error(
                            f"Failed to move duplicate scan file {duplicate.name} to {PARSED_DIR_NAME}: {move_error}"
                        )

            # Partition unique_files into:
            #   * processed_files — already PROCESSED, archived and skipped.
            #   * resume_files    — same canonical basename matches an existing
            #                       non-PROCESSED doc_status row (PARSING /
            #                       FAILED / PROCESSING / ANALYZING / PENDING).
            #                       These must NOT go through pipeline_enqueue_file
            #                       because apipeline_enqueue_documents would
            #                       treat the same canonical name as a duplicate
            #                       (returning None) and pipeline_enqueue_file
            #                       would then archive the source as if it were
            #                       a duplicate — corrupting pending-parse cases
            #                       that still need the source on disk.  The
            #                       pipeline's resume logic, triggered via
            #                       apipeline_process_enqueue_documents, will
            #                       advance them based on their existing
            #                       doc_status row.
            #   * new_files       — no existing record; standard enqueue path.
            new_files: list[Path] = []
            resume_files: list[Path] = []
            processed_files: list[str] = []

            for file_path in unique_files:
                filename = file_path.name
                # Inline the canonical-basename lookup so we keep both the
                # doc_id and the data: the FAILED-without-full_docs sub-case
                # below needs the doc_id to delete the stale stub. Scanned
                # files enqueue under their basename identity, so the lookup
                # keys on the name, not the scan path.
                basename = normalize_file_path(filename)
                existing_match = (
                    await rag.doc_status.get_doc_by_file_basename(basename)
                    if basename != UNKNOWN_FILE_SOURCE
                    else None
                )
                existing_doc_id, existing_doc_data = (
                    existing_match if existing_match else (None, None)
                )

                if (
                    existing_doc_data
                    and doc_status_value(existing_doc_data) == DocStatus.PROCESSED.value
                ):
                    # File is already PROCESSED, skip it with warning and archive it.
                    processed_files.append(filename)
                    warning = f"Skipping already processed file: {filename}"
                    await record_scan_warning(rag, warning)
                    try:
                        await move_file_to_parsed_dir(file_path)
                    except Exception as move_error:
                        logger.error(
                            f"Failed to move already processed file {filename} to {PARSED_DIR_NAME}: {move_error}"
                        )
                elif existing_doc_data:
                    # FAILED rows recorded by apipeline_enqueue_error_documents
                    # never write a full_docs entry — extraction blew up before
                    # any content was stored.  _validate_and_fix_document_consistency
                    # preserves them for manual review and removes them from the
                    # processing list, so the resume path can never advance them.
                    # When the user fixes the file and re-scans we want a real
                    # retry: drop the stale stub and treat the file as new so
                    # the standard enqueue path re-extracts content.
                    status_value = doc_status_value(existing_doc_data)
                    if status_value == DocStatus.FAILED.value:
                        full_doc = await rag.full_docs.get_by_id(existing_doc_id)
                        if full_doc is None:
                            try:
                                await rag.doc_status.delete([existing_doc_id])
                            except Exception as delete_error:
                                logger.error(
                                    "Failed to delete stale failed-extraction "
                                    f"doc_status stub {existing_doc_id} "
                                    f"({filename}): {delete_error}"
                                )
                                # Fall through to resume — at worst the row
                                # remains preserved (current behaviour) rather
                                # than re-enqueued.
                                resume_files.append(file_path)
                                continue
                            logger.info(
                                "Retrying previously failed extraction; "
                                f"removed stale doc_status stub: {filename} "
                                f"(doc_id: {existing_doc_id})"
                            )
                            new_files.append(file_path)
                            continue
                    logger.info(
                        "Resuming previously unfinished file from scan: "
                        f"{filename} (Status: {status_value})"
                    )
                    resume_files.append(file_path)
                else:
                    new_files.append(file_path)

            # Classification phase complete — release ``scanning_exclusive``
            # so concurrent uploads/inserts can land in doc_status while
            # the scan-driven processing finishes.  ``scanning`` stays
            # True for the rest of the task lifecycle (releases in
            # finally) so the /scan endpoint still refuses overlapping
            # scans.  Any per-file enqueue or duplicate detected during
            # the processing phase is handled by
            # apipeline_enqueue_documents' in-batch dedup, identical to
            # the upload-during-busy case.
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["scanning_exclusive"] = False

            # New files take the standard enqueue + process path.  When at
            # least one new file is successfully enqueued, pipeline_index_files
            # internally invokes apipeline_process_enqueue_documents, which
            # selects work by doc_status state and so will also pick up any
            # resume_files in the same run.
            if new_files:
                await pipeline_index_files(
                    rag,
                    new_files,
                    track_id,
                    from_scan=True,
                )

            # Resume targets must always trigger the pipeline explicitly:
            # pipeline_index_files only runs apipeline_process_enqueue_documents
            # after at least one new file successfully enqueues, so when every
            # new file is rejected (unsupported extension, empty body, content
            # / filename duplicate, ...) the resume rows would otherwise stay
            # stuck until an unrelated indexing run.  When new files DID
            # enqueue, the inner call already drained the queue and this is a
            # cheap no-op that returns "No documents to process".
            if resume_files:
                await rag.apipeline_process_enqueue_documents()

            total_active = len(new_files) + len(resume_files)
            if total_active or processed_files:
                summary_parts: list[str] = []
                if total_active:
                    summary_parts.append(f"{total_active} files Processed")
                if processed_files:
                    summary_parts.append(f"{len(processed_files)} skipped")
                logger.info(f"Scanning process completed: {' '.join(summary_parts)}.")
            else:
                logger.info(
                    "No files to process after filtering already processed files."
                )
        else:
            # No new files to index — classification is trivially done;
            # release ``scanning_exclusive`` before driving the queue so
            # concurrent uploads can land while process_enqueue runs.
            if pipeline_status is not None and pipeline_status_lock is not None:
                async with pipeline_status_lock:
                    pipeline_status["scanning_exclusive"] = False
            logger.info(
                "No upload file found, check if there are any documents in the queue..."
            )
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error during scanning process: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Always release both scanning flags so future uploads / scans
        # are not blocked by a crashed task.  Skip when pipeline_status
        # was never initialised for this workspace (test rigs).
        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["scanning"] = False
                pipeline_status["scanning_exclusive"] = False


async def background_delete_documents(
    rag: LightRAG,
    doc_manager: DocumentManager,
    doc_ids: List[str],
    delete_file: bool = False,
    delete_llm_cache: bool = False,
):
    """Background task to delete multiple documents"""
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
    )

    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )

    total_docs = len(doc_ids)
    successful_deletions = []
    failed_deletions = []

    # The /documents/delete_document endpoint has already reserved the
    # destructive slot synchronously: ``busy=True`` and
    # ``destructive_busy=True`` were set before the client got
    # ``deletion_started``, after checking busy + scanning +
    # pending_enqueues>0 atomically.  Here we only update the
    # job-info fields; the busy reservation was acquired by the
    # endpoint and is released in the finally block below.
    async with pipeline_status_lock:
        pipeline_status.update(
            {
                # Job name can not be changed, it's verified in adelete_by_doc_id()
                "job_name": f"Deleting {total_docs} Documents",
                "job_start": datetime.now().isoformat(),
                "docs": total_docs,
                "batchs": total_docs,
                "cur_batch": 0,
                "latest_message": "Starting document deletion process",
            }
        )
        # Use slice assignment to clear the list in place
        pipeline_status["history_messages"][:] = ["Starting document deletion process"]
        if delete_llm_cache:
            pipeline_status["history_messages"].append(
                "LLM cache cleanup requested for this deletion job"
            )

    try:
        # Loop through each document ID and delete them one by one
        for i, doc_id in enumerate(doc_ids, 1):
            # Check for cancellation at the start of each document deletion
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    cancel_msg = f"Deletion cancelled by user at document {i}/{total_docs}. {len(successful_deletions)} deleted, {total_docs - i + 1} remaining."
                    logger.info(cancel_msg)
                    pipeline_status["latest_message"] = cancel_msg
                    pipeline_status["history_messages"].append(cancel_msg)
                    # Add remaining documents to failed list with cancellation reason
                    failed_deletions.extend(
                        doc_ids[i - 1 :]
                    )  # i-1 because enumerate starts at 1
                    break  # Exit the loop, remaining documents unchanged

                start_msg = f"Deleting document {i}/{total_docs}: {doc_id}"
                logger.info(start_msg)
                pipeline_status["cur_batch"] = i
                pipeline_status["latest_message"] = start_msg
                pipeline_status["history_messages"].append(start_msg)

            file_path = "#"
            try:
                result = await rag.adelete_by_doc_id(
                    doc_id, delete_llm_cache=delete_llm_cache
                )
                file_path = (
                    getattr(result, "file_path", "-") if "result" in locals() else "-"
                )
                if result.status == "success":
                    successful_deletions.append(doc_id)
                    success_msg = (
                        f"Document deleted {i}/{total_docs}: {doc_id}[{file_path}]"
                    )
                    logger.info(success_msg)
                    async with pipeline_status_lock:
                        pipeline_status["history_messages"].append(success_msg)

                    # Handle file deletion if requested and source information is available
                    if (
                        delete_file
                        and result.file_path
                        and result.file_path != UNKNOWN_FILE_SOURCE
                    ):
                        try:
                            deleted_files, file_delete_errors = (
                                delete_file_variants_by_file_path(
                                    doc_manager.input_dir,
                                    result.file_path,
                                )
                            )
                            for file_delete_error in file_delete_errors:
                                logger.warning(file_delete_error)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = (
                                        file_delete_error
                                    )
                                    pipeline_status["history_messages"].append(
                                        file_delete_error
                                    )

                            if deleted_files:
                                file_delete_msg = (
                                    "Successfully deleted source files: "
                                    + ", ".join(deleted_files)
                                )
                                logger.info(file_delete_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_delete_msg
                                    pipeline_status["history_messages"].append(
                                        file_delete_msg
                                    )
                            else:
                                file_error_msg = (
                                    "File deletion skipped, missing or unsafe file: "
                                    f"{result.file_path}"
                                )
                                logger.warning(file_error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_error_msg
                                    pipeline_status["history_messages"].append(
                                        file_error_msg
                                    )

                        except Exception as file_error:
                            file_error_msg = f"Failed to delete file {result.file_path}: {str(file_error)}"
                            logger.error(file_error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = file_error_msg
                                pipeline_status["history_messages"].append(
                                    file_error_msg
                                )
                    elif delete_file:
                        no_file_msg = (
                            f"File deletion skipped, missing file path: {doc_id}"
                        )
                        logger.warning(no_file_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = no_file_msg
                            pipeline_status["history_messages"].append(no_file_msg)
                else:
                    failed_deletions.append(doc_id)
                    error_msg = f"Failed to delete {i}/{total_docs}: {doc_id}[{file_path}] - {result.message}"
                    logger.error(error_msg)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)

            except Exception as e:
                failed_deletions.append(doc_id)
                error_msg = f"Error deleting document {i}/{total_docs}: {doc_id}[{file_path}] - {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(error_msg)

    except Exception as e:
        error_msg = f"Critical error during batch deletion: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        async with pipeline_status_lock:
            pipeline_status["history_messages"].append(error_msg)
    finally:
        # Final summary and check for pending requests
        async with pipeline_status_lock:
            pipeline_status["busy"] = False
            pipeline_status["destructive_busy"] = False
            pipeline_status["pending_requests"] = False  # Reset pending requests flag
            pipeline_status["cancellation_requested"] = (
                False  # Always reset cancellation flag
            )
            completion_msg = f"Deletion completed: {len(successful_deletions)} successful, {len(failed_deletions)} failed"
            pipeline_status["latest_message"] = completion_msg
            pipeline_status["history_messages"].append(completion_msg)

            # Check if there are pending document indexing requests
            has_pending_request = pipeline_status.get("request_pending", False)

        # If there are pending requests, start document processing pipeline
        if has_pending_request:
            try:
                logger.info(
                    "Processing pending document indexing requests after deletion"
                )
                await rag.apipeline_process_enqueue_documents()
            except Exception as e:
                logger.error(f"Error processing pending documents after deletion: {e}")
