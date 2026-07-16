import base64
import json
import os
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, final

from lightrag.file_atomic import atomic_write, reap_orphan_tmp_files
from lightrag.kg.graph_contract import EvidenceRef, GraphAssertion, GraphEntity
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger, validate_workspace
from lightrag.base import BaseGraphStorage
import networkx as nx
from .shared_storage import (
    get_namespace_lock,
    get_update_flag,
    set_all_update_flags,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


_TYPED_RECORD_KIND = "_lightrag_record_kind"
_CONTRACT_DIGEST = "contract_digest"
_LEGACY_EDGE_KEY = "_lightrag_legacy_edge"
_INTERVAL_FIELDS = ("observed_from", "observed_to", "valid_from", "valid_to")
_ENTITY_PROPERTY_NAMES = {
    _TYPED_RECORD_KIND,
    _CONTRACT_DIGEST,
    *(field.name for field in fields(GraphEntity)),
}


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _json_property(value: object) -> str:
    payload = json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def _decode_json_property(value: object) -> object:
    if not isinstance(value, str):
        raise ValueError("stored typed graph JSON property is not a string")
    payload = base64.b64decode(value, altchars=b"-_", validate=True)
    return json.loads(payload.decode("utf-8"))


def _evidence_property(evidence: tuple[EvidenceRef, ...]) -> str:
    return _json_property(
        [
            {
                "chunk_id": item.chunk_id,
                "source_key": item.source_key,
                "source_revision": item.source_revision,
                "metadata": item.metadata,
            }
            for item in evidence
        ]
    )


def _record_properties(
    record: GraphEntity | GraphAssertion,
    contract_digest: str | None,
) -> dict[str, str | float]:
    properties: dict[str, str | float] = {
        _TYPED_RECORD_KIND: type(record).__name__,
    }
    for item in fields(record):
        value = getattr(record, item.name)
        if value is None:
            continue
        if item.name == "evidence":
            properties[item.name] = _evidence_property(value)
        elif item.name == "metadata":
            properties[item.name] = _json_property(value)
        elif isinstance(value, datetime):
            properties[item.name] = value.isoformat()
        else:
            properties[item.name] = value
    if contract_digest is not None:
        properties[_CONTRACT_DIGEST] = contract_digest
    return properties


def _decode_evidence(value: object) -> tuple[EvidenceRef, ...]:
    decoded = _decode_json_property(value)
    if not isinstance(decoded, list):
        raise ValueError("stored typed graph evidence must be a JSON array")
    return tuple(EvidenceRef(**item) for item in decoded)


def _decode_metadata(value: object) -> dict[str, Any]:
    decoded = _decode_json_property(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored typed graph metadata must be a JSON object")
    return decoded


def _decode_interval(properties: Mapping[str, object], field_name: str) -> datetime | None:
    value = properties.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"stored {field_name} must be an ISO 8601 string")
    return datetime.fromisoformat(value)


def _validate_contract_digest(contract_digest: str | None) -> None:
    if contract_digest is None:
        return
    if len(contract_digest) != 64 or any(
        character not in "0123456789abcdef" for character in contract_digest
    ):
        raise ValueError("contract_digest must be a lowercase SHA-256 digest")


def _stored_record_data(
    record: GraphEntity | GraphAssertion,
    properties: Mapping[str, object],
) -> dict[str, Any]:
    data = {item.name: getattr(record, item.name) for item in fields(record)}
    data[_CONTRACT_DIGEST] = properties.get(_CONTRACT_DIGEST)
    return data


@final
@dataclass
class NetworkXStorage(BaseGraphStorage):
    """File-backed knowledge-graph storage built on ``networkx.MultiDiGraph``.

    Storage model:
        A single ``networkx.MultiDiGraph`` instance lives in process memory; its
        full state is serialized to one GraphML file at
        ``working_dir/[workspace/]graph_<namespace>.graphml``. That GraphML
        file is the **only** cross-process synchronization surface — there
        is no shared memory, no message bus, and no network channel
        between processes. Cross-process visibility is mediated by (a) an
        atomic file write at commit time and (b) a per-namespace
        ``storage_updated`` flag distributed through
        ``lightrag.kg.shared_storage``.

    Concurrency invariants (the code in this file is correct *only* while
    all three hold):
        1. **Single writer per workspace.** The document pipeline's
           ``busy`` / ``destructive_busy`` flags (see ``AGENTS.md``
           *Pipeline concurrency contract*) guarantee at most one process
           performs ``upsert_*`` / ``delete_*`` / ``remove_*`` /
           ``index_done_callback`` at any time. Every other process is
           read-only.
        2. **Eventual consistency is sufficient.** Read-only processes
           only need to observe the writer's data *after* the writer's
           ``index_done_callback`` completes. Reads landing in the gap
           between a writer's in-memory mutation and its commit may
           legitimately return the pre-update snapshot.
        3. **networkx operations are fully synchronous.** Under a
           single-threaded asyncio event loop, ``graph.add_node`` /
           ``graph.remove_node`` / ``graph.degree`` / etc. cannot be
           preempted by another coroutine, which gives them implicit
           mutual exclusion over ``self._graph``. This is why the methods
           below don't have to hold ``_storage_lock`` while calling into
           ``graph``.

    Cross-process sync protocol (identical in shape to
    ``NanoVectorDBStorage`` — see that class's docstring for the canonical
    description):
        Writer side (``index_done_callback``):
            1. ``write_nx_graph`` atomically writes the GraphML file
               (``atomic_write`` lays a tmp file beside the target and
               renames it into place — readers either see the previous
               file in full or the new file in full, never a torn write).
            2. ``set_all_update_flags`` flips every process's
               ``storage_updated`` flag (including the writer's own).
            3. Immediately reset the writer's own flag to ``False`` so
               the next call to ``_get_graph`` does not trigger a
               self-reload of the data this process just wrote.
        Reader side (any method that goes through ``_get_graph``):
            1. Inside ``_storage_lock``, observe
               ``storage_updated.value is True``.
            2. **Fully reload** ``self._graph`` from disk via
               ``load_nx_graph``. networkx GraphML has no incremental
               sync API, so the entire file is re-parsed.
            3. Reset the reader's own flag.

    Lock scope:
        ``_storage_lock`` is a per-``(namespace, workspace)`` keyed lock
        spanning both intra-process coroutines and inter-process workers.
        It wraps only the *reload* and *commit* critical sections, not
        every ``graph.xxx`` call. Operating on ``graph`` outside the lock
        is safe today *because of invariant (3)* — if either premise is
        ever broken (e.g. ``graph.xxx`` is moved to a thread pool, or
        networkx is swapped for an async graph library), the lock scope
        must be widened to cover the mutation/read itself.

    Implementation differences from ``NanoVectorDBStorage`` (same design,
    different surface):
        * No ``client_storage`` property — there is no equivalent live
          reference being exposed to callers, so NanoVectorDB's
          "do-not-retain-across-await" caveat does not apply here.
        * ``write_nx_graph`` passes the tmp path directly to
          ``nx.write_graphml``, so the writer needs no equivalent of
          NanoVectorDB's "temporarily reassign ``storage_file``" trick.
        * Mutation surface is finer-grained (``upsert_node`` /
          ``upsert_edge`` / ``upsert_nodes_batch`` /
          ``upsert_edges_batch`` / ``delete_node`` / ``remove_nodes`` /
          ``remove_edges``); each goes through ``_get_graph`` once and
          then operates synchronously on ``self._graph``.

    Non-pipeline write paths:
        The pipeline's ``busy`` gate serializes mutation calls reached
        through the document ingestion and purge flows. The following
        entry points are **not** serialized by the pipeline gate and
        must be guarded externally:
            * ``drop`` — currently gated by the API layer (the
              ``/documents/clear`` endpoint takes the pipeline busy
              reservation before invoking it).
            * ``delete_node`` / ``remove_nodes`` / ``remove_edges`` /
              ``upsert_node`` / ``upsert_edge`` when invoked from
              ``utils_graph.py`` admin flows (``adelete_by_entity`` /
              ``adelete_by_relation`` / entity-edit flows). These flows
              are currently not exposed in the WebUI; any future caller
              must arrange single-writer serialization the same way the
              pipeline does.
    """

    @staticmethod
    def load_nx_graph(file_name) -> nx.MultiDiGraph | None:
        if os.path.exists(file_name):
            graph = nx.read_graphml(file_name, force_multigraph=True)
            if not graph.is_directed():
                raise ValueError(
                    "Legacy undirected NetworkX GraphML is unsupported by the "
                    "typed directed multigraph protocol; clean startup is required"
                )
            if not isinstance(graph, nx.MultiDiGraph):
                raise ValueError("GraphML did not load as a directed multigraph")
            return graph
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name, workspace="_"):
        logger.info(
            f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        atomic_write(
            file_name,
            lambda tmp: nx.write_graphml(graph, tmp),
            workspace,
        )

    def __post_init__(self):
        # Reject path traversal before using workspace in a file path
        validate_workspace(self.workspace)
        working_dir = self.global_config["working_dir"]
        if self.workspace:
            # Include workspace in the file path for data isolation
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            # Default behavior when workspace is empty
            workspace_dir = working_dir
            self.workspace = ""

        os.makedirs(workspace_dir, exist_ok=True)
        self._graphml_xml_file = os.path.join(
            workspace_dir, f"graph_{self.namespace}.graphml"
        )
        self._storage_lock: Any = None
        self.storage_updated: Any = None

        reap_orphan_tmp_files(self._graphml_xml_file, workspace=self.workspace or "_")

        # Load initial graph
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"[{self.workspace}] Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        else:
            logger.info(
                f"[{self.workspace}] Created new empty graph file: {self._graphml_xml_file}"
            )
        self._graph: nx.MultiDiGraph = preloaded_graph or nx.MultiDiGraph()

    async def initialize(self):
        """Initialize storage data"""
        # Get the update flag for cross-process update notification
        self.storage_updated = await get_update_flag(
            self.namespace, workspace=self.workspace
        )
        # Get the storage lock for use in other methods
        self._storage_lock = get_namespace_lock(
            self.namespace, workspace=self.workspace
        )

    async def _get_graph(self) -> nx.MultiDiGraph:
        """Return the live ``networkx.MultiDiGraph``, reloading if needed.

        This is the **single entry point** every public method funnels
        through to obtain ``self._graph``. It is also the **only place
        readers transition to a fresher on-disk snapshot**: when another
        process has committed (via ``index_done_callback``) and flipped
        this process's ``storage_updated`` flag, the next call here
        rebuilds ``self._graph`` by re-parsing the entire GraphML file.
        networkx has no incremental sync API — the reload is
        unconditionally a full file reload.

        Under the *Single writer* invariant (see class docstring), the
        reload branch never fires in the writer process: the writer
        resets its own flag at the end of every ``index_done_callback``.
        The branch exists for readers.

        ``_storage_lock`` is held during the check-and-reload to (a)
        serialize concurrent reload attempts by sibling coroutines in
        the same process and (b) interlock with ``index_done_callback``
        so a reader cannot observe a partially-saved file.
        """
        async with self._storage_lock:
            # Check if data needs to be reloaded
            if self.storage_updated.value:
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} reloading graph {self._graphml_xml_file} due to modifications by another process"
                )
                # Reload data
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file)
                    or nx.MultiDiGraph()
                )
                # Reset update flag
                self.storage_updated.value = False

            return self._graph

    async def has_node(self, node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        graph = await self._get_graph()
        return graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        graph = await self._get_graph()
        return graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        graph = await self._get_graph()
        if graph.has_node(node_id):
            return graph.degree(node_id)
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        graph = await self._get_graph()
        src_degree = graph.degree(src_id) if graph.has_node(src_id) else 0
        tgt_degree = graph.degree(tgt_id) if graph.has_node(tgt_id) else 0
        return src_degree + tgt_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        graph = await self._get_graph()
        edge_data = graph.get_edge_data(source_node_id, target_node_id)
        if edge_data is None:
            return None
        if _LEGACY_EDGE_KEY in edge_data:
            return edge_data[_LEGACY_EDGE_KEY]
        first_key = min(edge_data, key=str)
        return edge_data[first_key]

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        graph = await self._get_graph()
        if graph.has_node(source_node_id):
            return list(graph.edges(source_node_id))
        return None

    async def upsert_graph_entity(
        self,
        entity: GraphEntity,
        *,
        contract_digest: str | None = None,
    ) -> None:
        await self.upsert_graph_entities(
            [entity],
            contract_digest=contract_digest,
        )

    async def upsert_graph_entities(
        self,
        entities: list[GraphEntity],
        *,
        contract_digest: str | None = None,
    ) -> None:
        _validate_contract_digest(contract_digest)
        by_id: dict[str, GraphEntity] = {}
        for entity in entities:
            if not isinstance(entity, GraphEntity):
                raise TypeError("entities must contain only GraphEntity records")
            by_id[entity.entity_id] = entity

        graph = await self._get_graph()
        for entity_id in sorted(by_id):
            entity = by_id[entity_id]
            if graph.has_node(entity_id):
                for property_name in _ENTITY_PROPERTY_NAMES:
                    graph.nodes[entity_id].pop(property_name, None)
            graph.add_node(
                entity_id,
                **_record_properties(entity, contract_digest),
            )

    async def get_graph_entity(self, entity_id: str) -> dict[str, Any] | None:
        graph = await self._get_graph()
        properties = graph.nodes.get(entity_id)
        if properties is None or properties.get(_TYPED_RECORD_KIND) != "GraphEntity":
            return None
        entity = GraphEntity(
            build_id=properties["build_id"],
            entity_id=properties["entity_id"],
            entity_type=properties["entity_type"],
            evidence=_decode_evidence(properties["evidence"]),
            metadata=_decode_metadata(properties["metadata"]),
            **{
                field_name: _decode_interval(properties, field_name)
                for field_name in _INTERVAL_FIELDS
            },
        )
        return _stored_record_data(entity, properties)

    @staticmethod
    def _assertion_edges(
        graph: nx.MultiDiGraph,
        assertion_id: str,
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        return [
            (src_id, dst_id, key, properties)
            for src_id, dst_id, key, properties in graph.edges(
                keys=True, data=True
            )
            if key == assertion_id
        ]

    async def upsert_graph_assertion(
        self,
        assertion: GraphAssertion,
        *,
        contract_digest: str | None = None,
    ) -> None:
        await self.upsert_graph_assertions(
            [assertion],
            contract_digest=contract_digest,
        )

    async def upsert_graph_assertions(
        self,
        assertions: list[GraphAssertion],
        *,
        contract_digest: str | None = None,
    ) -> None:
        _validate_contract_digest(contract_digest)
        by_id: dict[str, GraphAssertion] = {}
        for assertion in assertions:
            if not isinstance(assertion, GraphAssertion):
                raise TypeError("assertions must contain only GraphAssertion records")
            by_id[assertion.assertion_id] = assertion

        graph = await self._get_graph()
        for assertion in by_id.values():
            missing = sorted(
                endpoint
                for endpoint in (assertion.src_id, assertion.dst_id)
                if not graph.has_node(endpoint)
            )
            if missing:
                raise ValueError(
                    f"assertion {assertion.assertion_id!r} has a missing endpoint: "
                    f"{missing!r}"
                )

        for assertion_id in sorted(by_id):
            assertion = by_id[assertion_id]
            for src_id, dst_id, key, _ in self._assertion_edges(
                graph, assertion_id
            ):
                graph.remove_edge(src_id, dst_id, key=key)
            graph.add_edge(
                assertion.src_id,
                assertion.dst_id,
                key=assertion.assertion_id,
                **_record_properties(assertion, contract_digest),
            )

    async def get_graph_assertion(self, assertion_id: str) -> dict[str, Any] | None:
        graph = await self._get_graph()
        matches = self._assertion_edges(graph, assertion_id)
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"graph contains duplicate assertion_id {assertion_id!r}"
            )
        _, _, _, properties = matches[0]
        if properties.get(_TYPED_RECORD_KIND) != "GraphAssertion":
            return None
        confidence = properties.get("confidence")
        assertion = GraphAssertion(
            build_id=properties["build_id"],
            assertion_id=properties["assertion_id"],
            predicate=properties["predicate"],
            src_id=properties["src_id"],
            dst_id=properties["dst_id"],
            evidence=_decode_evidence(properties["evidence"]),
            metadata=_decode_metadata(properties["metadata"]),
            confidence=float(confidence) if confidence is not None else None,
            method=properties.get("method"),
            **{
                field_name: _decode_interval(properties, field_name)
                for field_name in _INTERVAL_FIELDS
            },
        )
        return _stored_record_data(assertion, properties)

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a single node; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. In ``lightrag.py`` this
            is handled by ``_insert_done()`` at the end of the document
            batch. Callers outside the pipeline must persist explicitly.

        Correctness relies on the class docstring *Lock scope* invariant
        (synchronous networkx ops + single-writer pipeline gate).
        """
        graph = await self._get_graph()
        graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert or update a single edge; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Correctness relies on the class docstring *Lock scope* invariant.
        """
        graph = await self._get_graph()
        graph.add_edge(
            source_node_id,
            target_node_id,
            key=_LEGACY_EDGE_KEY,
            **edge_data,
        )

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict[str, str]]]) -> None:
        """Batch insert/update multiple nodes in a single call.

        Much faster than calling upsert_node() in a loop for large imports
        because it avoids per-call async event loop overhead.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Args:
            nodes: List of (node_id, node_data) tuples.
        """
        graph = await self._get_graph()
        for node_id, node_data in nodes:
            graph.add_node(node_id, **node_data)

    async def has_nodes_batch(self, node_ids: list[str]) -> set[str]:
        """Check existence of multiple nodes in a single call.

        Returns:
            Set of node_ids that exist in the graph.
        """
        graph = await self._get_graph()
        return {nid for nid in node_ids if graph.has_node(nid)}

    async def upsert_edges_batch(
        self, edges: list[tuple[str, str, dict[str, str]]]
    ) -> None:
        """Batch insert/update multiple edges in a single call.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Args:
            edges: List of (source_id, target_id, edge_data) tuples.
        """
        graph = await self._get_graph()
        for src, tgt, edge_data in edges:
            graph.add_edge(src, tgt, key=_LEGACY_EDGE_KEY, **edge_data)

    async def delete_node(self, node_id: str) -> None:
        """Remove a single node from the graph; persistence is deferred.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller: invocations from the
        document purge flow are serialized by ``pipeline busy``;
        invocations from ``utils_graph.py`` admin flows are **not** —
        see class docstring *Non-pipeline write paths*.
        """
        graph = await self._get_graph()
        if graph.has_node(node_id):
            graph.remove_node(node_id)
            logger.debug(f"[{self.workspace}] Node {node_id} deleted from the graph")
        else:
            logger.warning(
                f"[{self.workspace}] Node {node_id} not found in the graph for deletion"
            )

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes from the graph.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller — see ``delete_node`` and
        class docstring *Non-pipeline write paths*.

        Args:
            nodes: List of node IDs to be deleted
        """
        graph = await self._get_graph()
        for node in nodes:
            if graph.has_node(node):
                graph.remove_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges from the graph.

        Persistence:
            Changes are in-memory only; cross-process visibility requires
            a subsequent ``index_done_callback``. Callers outside the
            pipeline must persist explicitly.

        Pipeline-gating depends on the caller — see ``delete_node`` and
        class docstring *Non-pipeline write paths*.

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        graph = await self._get_graph()
        for source, target in edges:
            if graph.has_edge(source, target, key=_LEGACY_EDGE_KEY):
                graph.remove_edge(source, target, key=_LEGACY_EDGE_KEY)

    async def get_all_labels(self) -> list[str]:
        """
        Get all node labels(entity names) in the graph
        Returns:
            [label1, label2, ...]  # Alphabetically sorted label list
        """
        graph = await self._get_graph()
        labels = set()
        for node in graph.nodes():
            labels.add(str(node))  # Add node id as a label

        # Return sorted list
        return sorted(list(labels))

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """
        Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """
        graph = await self._get_graph()

        # Get degrees of all nodes and sort by degree descending
        degrees = dict(graph.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

        # Return top labels limited by the specified limit
        popular_labels = [str(node) for node, _ in sorted_nodes[:limit]]

        logger.debug(
            f"[{self.workspace}] Retrieved {len(popular_labels)} popular labels (limit: {limit})"
        )

        return popular_labels

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """
        Search labels(entity names) with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
        graph = await self._get_graph()
        query_lower = query.lower().strip()

        if not query_lower:
            return []

        # Collect matching nodes with relevance scores
        matches = []
        for node in graph.nodes():
            node_str = str(node)
            node_lower = node_str.lower()

            # Skip if no match
            if query_lower not in node_lower:
                continue

            # Calculate relevance score
            # Exact match gets highest score
            if node_lower == query_lower:
                score = 1000
            # Prefix match gets high score
            elif node_lower.startswith(query_lower):
                score = 500
            # Contains match gets base score, with bonus for shorter strings
            else:
                # Shorter strings with matches are more relevant
                score = 100 - len(node_str)
                # Bonus for word boundary matches
                if f" {query_lower}" in node_lower or f"_{query_lower}" in node_lower:
                    score += 50

            matches.append((node_str, score))

        # Sort by relevance score (desc) then alphabetically
        matches.sort(key=lambda x: (-x[1], x[0]))

        # Return top matches limited by the specified limit
        search_results = [match[0] for match in matches[:limit]]

        logger.debug(
            f"[{self.workspace}] Search query '{query}' returned {len(search_results)} results (limit: {limit})"
        )

        return search_results

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        graph = await self._get_graph()

        result = KnowledgeGraph()

        # Handle special case for "*" label
        if node_label == "*":
            # Get degrees of all nodes
            degrees = dict(graph.degree())
            # Sort nodes by degree in descending order and take top max_nodes
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

            # Check if graph is truncated
            if len(sorted_nodes) > max_nodes:
                result.is_truncated = True
                logger.info(
                    f"[{self.workspace}] Graph truncated: {len(sorted_nodes)} nodes found, limited to {max_nodes}"
                )

            limited_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
            # Create subgraph with the highest degree nodes
            subgraph = graph.subgraph(limited_nodes)
        else:
            # Check if node exists
            if node_label not in graph:
                logger.warning(
                    f"[{self.workspace}] Node {node_label} not found in the graph"
                )
                return KnowledgeGraph()  # Return empty graph

            # Use modified BFS to get nodes, prioritizing high-degree nodes at the same depth
            bfs_nodes = []
            visited = set()
            # Store (node, depth, degree) in the queue
            queue = deque([(node_label, 0, graph.degree(node_label))])

            # Flag to track if there are unexplored neighbors due to depth limit
            has_unexplored_neighbors = False

            # Modified breadth-first search with degree-based prioritization
            while queue and len(bfs_nodes) < max_nodes:
                # Get the current depth from the first node in queue
                current_depth = queue[0][1]

                # Collect all nodes at the current depth
                current_level_nodes = []
                while queue and queue[0][1] == current_depth:
                    current_level_nodes.append(queue.popleft())

                # Sort nodes at current depth by degree (highest first)
                current_level_nodes.sort(key=lambda x: x[2], reverse=True)

                # Process all nodes at current depth in order of degree
                for current_node, depth, degree in current_level_nodes:
                    if current_node not in visited:
                        visited.add(current_node)
                        bfs_nodes.append(current_node)

                        # Only explore neighbors if we haven't reached max_depth
                        if depth < max_depth:
                            # Add neighbor nodes to queue with incremented depth
                            neighbors = list(graph.neighbors(current_node))
                            # Filter out already visited neighbors
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            # Add neighbors to the queue with their degrees
                            for neighbor in unvisited_neighbors:
                                neighbor_degree = graph.degree(neighbor)
                                queue.append((neighbor, depth + 1, neighbor_degree))
                        else:
                            # Check if there are unexplored neighbors (skipped due to depth limit)
                            neighbors = list(graph.neighbors(current_node))
                            unvisited_neighbors = [
                                n for n in neighbors if n not in visited
                            ]
                            if unvisited_neighbors:
                                has_unexplored_neighbors = True

                    # Check if we've reached max_nodes
                    if len(bfs_nodes) >= max_nodes:
                        break

            # Check if graph is truncated - either due to max_nodes limit or depth limit
            if (queue and len(bfs_nodes) >= max_nodes) or has_unexplored_neighbors:
                if len(bfs_nodes) >= max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"[{self.workspace}] Graph truncated: max_nodes limit {max_nodes} reached"
                    )
                else:
                    logger.info(
                        f"[{self.workspace}] Graph truncated: found {len(bfs_nodes)} nodes within max_depth {max_depth}"
                    )

            # Create subgraph with BFS discovered nodes
            subgraph = graph.subgraph(bfs_nodes)

        # Add nodes to result
        seen_nodes = set()
        seen_edges = set()
        for node in subgraph.nodes():
            if str(node) in seen_nodes:
                continue

            node_data = dict(subgraph.nodes[node])
            # Get entity_type as labels
            labels = []
            if "entity_type" in node_data:
                if isinstance(node_data["entity_type"], list):
                    labels.extend(node_data["entity_type"])
                else:
                    labels.append(node_data["entity_type"])

            # Create node with properties
            node_properties = {k: v for k, v in node_data.items()}

            result.nodes.append(
                KnowledgeGraphNode(
                    id=str(node), labels=[str(node)], properties=node_properties
                )
            )
            seen_nodes.add(str(node))

        # Add edges to result
        for source, target, key, edge_data in subgraph.edges(
            keys=True, data=True
        ):
            edge_id = (
                f"{source}-{target}"
                if key == _LEGACY_EDGE_KEY
                else str(key)
            )
            if edge_id in seen_edges:
                continue

            # Create edge with complete information
            result.edges.append(
                KnowledgeGraphEdge(
                    id=edge_id,
                    type="DIRECTED",
                    source=str(source),
                    target=str(target),
                    properties=dict(edge_data),
                )
            )
            seen_edges.add(edge_id)

        logger.info(
            f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_data_with_id = node_data.copy()
            node_data_with_id["id"] = node_id
            all_nodes.append(node_data_with_id)
        return all_nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        graph = await self._get_graph()
        all_edges = []
        for u, v, key, edge_data in graph.edges(keys=True, data=True):
            edge_data_with_nodes = edge_data.copy()
            edge_data_with_nodes["id"] = key
            edge_data_with_nodes["source"] = u
            edge_data_with_nodes["target"] = v
            all_edges.append(edge_data_with_nodes)
        return all_edges

    async def index_done_callback(self) -> bool:
        """Commit in-memory graph to disk and notify other processes.

        This is the writer's **commit point** in the cross-process sync
        protocol (see class docstring). Two effects, in order:
            1. ``write_nx_graph`` atomically writes the GraphML file
               (``atomic_write`` swaps a tmp file into place).
            2. ``set_all_update_flags`` flips every registered process's
               ``storage_updated`` flag, then we immediately reset our
               own flag to ``False`` so the writer does not self-reload
               on the next call to ``_get_graph``.

        Two-block structure (intentional, do not collapse):
            * **First ``async with``** — early-return path for a
              hypothetical second writer. Under the current single-writer
              pipeline contract (class docstring, invariant 1) the
              ``storage_updated.value`` check is permanently ``False`` in
              the writer, so this branch is **dead code in production**.
              It is kept as defensive scaffolding for any future
              relaxation of the single-writer invariant; removing it
              would silently re-enable lost-write bugs the moment a
              second writer is introduced.
            * **Second ``async with``** — the actual save + notify.
        """
        async with self._storage_lock:
            # Check if storage was updated by another process
            if self.storage_updated.value:
                # Storage was updated by another process, reload data instead of saving
                logger.info(
                    f"[{self.workspace}] Graph was updated by another process, reloading..."
                )
                self._graph = (
                    NetworkXStorage.load_nx_graph(self._graphml_xml_file)
                    or nx.MultiDiGraph()
                )
                # Reset update flag
                self.storage_updated.value = False
                return False  # Return error

        # Acquire lock and perform persistence
        async with self._storage_lock:
            try:
                # Save data to disk
                NetworkXStorage.write_nx_graph(
                    self._graph, self._graphml_xml_file, self.workspace
                )
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                return True  # Return success
            except Exception as e:
                # Raise (do NOT swallow + return False): _insert_done's
                # _flush_one only detects failures via exceptions, so a
                # swallowed graph-save error would let the document be marked
                # PROCESSED with the graph changes unpersisted. Surfacing it
                # aligns this backend with the others (faiss/nano raise too).
                logger.error(f"[{self.workspace}] Error saving graph: {e}")
                raise

        return True

    async def drop(self) -> dict[str, str]:
        """Drop all graph data from storage and reinitialize the graph.

        This method will:
        1. Remove the graph storage file if it exists
        2. Reset the graph to an empty ``nx.MultiDiGraph()``
        3. Update flags to notify other processes
        4. Changes are persisted to disk immediately

        Caller contract:
            ``drop`` is destructive and **not** serialized by this storage
            class. The caller must hold the pipeline ``busy`` reservation
            (the ``/documents/clear`` endpoint does this) before invoking
            it — running ``drop`` concurrently with an active document
            pipeline will tear down storage out from under the writer and
            silently lose data. See class docstring,
            *Non-pipeline write paths*.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            async with self._storage_lock:
                # delete _client_file_name
                if os.path.exists(self._graphml_xml_file):
                    os.remove(self._graphml_xml_file)
                self._graph = nx.MultiDiGraph()
                # Notify other processes that data has been updated
                await set_all_update_flags(self.namespace, workspace=self.workspace)
                # Reset own update flag to avoid self-reloading
                self.storage_updated.value = False
                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop graph file:{self._graphml_xml_file}"
                )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping graph file:{self._graphml_xml_file}: {e}"
            )
            return {"status": "error", "message": str(e)}
