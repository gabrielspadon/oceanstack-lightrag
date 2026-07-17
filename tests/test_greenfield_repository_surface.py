"""Greenfield repository surface excludes legacy migration and local ontology assets."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _defined_symbols(relative_path: str) -> set[str]:
    source = (REPO_ROOT / relative_path).read_text()
    return {
        node.name
        for node in ast.walk(ast.parse(source, filename=relative_path))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_legacy_storage_migration_surface_is_absent():
    assert not (REPO_ROOT / "lightrag" / "storage_migrations.py").exists()

    core_source = (REPO_ROOT / "lightrag" / "lightrag.py").read_text()
    server_source = (REPO_ROOT / "lightrag" / "api" / "lightrag_server.py").read_text()
    repository_guidance = (REPO_ROOT / "AGENTS.md").read_text()
    assert "_StorageMigrationMixin" not in core_source
    assert "check_and_migrate_data" not in server_source
    assert "storage_migrations" not in repository_guidance
    assert "_StorageMigrationMixin" not in repository_guidance

    from lightrag import LightRAG

    assert not hasattr(LightRAG, "check_and_migrate_data")
    assert "_StorageMigrationMixin" not in {base.__name__ for base in LightRAG.__mro__}


def test_storage_backends_expose_only_greenfield_bootstrap_paths():
    forbidden_symbols = {
        "lightrag/kg/postgres_impl.py": {
            "_migrate_llm_cache_schema",
            "_check_llm_cache_needs_migration",
            "_migrate_llm_cache_to_flattened_keys",
            "_migrate_doc_status_add_chunks_list",
            "_migrate_text_chunks_add_llm_cache_list",
            "_migrate_doc_status_add_track_id",
            "_migrate_doc_status_add_metadata_error_msg",
            "_migrate_doc_full_add_pipeline_fields",
            "_migrate_doc_status_add_content_hash",
            "_migrate_text_chunks_add_heading_sidecar",
            "_migrate_create_full_entities_relations_tables",
        },
        "lightrag/kg/json_kv_impl.py": {"_migrate_legacy_cache_structure"},
        "lightrag/kg/redis_impl.py": {"_migrate_legacy_cache_structure"},
        "lightrag/kg/qdrant_impl.py": {
            "_find_legacy_collection",
            "_legacy_collection_has_workspace_field",
        },
        "lightrag/kg/milvus_impl.py": {
            "_get_migrated_metadata_field_limits",
            "_normalize_migration_row",
            "_check_metadata_schema_migration_needed",
            "_recover_interrupted_inplace_migration",
            "_migrate_collection_schema",
            "_migrate_collection_schema_attempt",
            "_repair_missing_vector_index",
            "_get_index_params",
            "_create_scalar_index_fallback",
        },
        "lightrag/kg/mongo_impl.py": {
            "create_and_migrate_indexes_if_not_exists",
            "create_edge_indexes_and_migrate_if_not_exists",
            "_dedupe_legacy_edges",
        },
    }

    for relative_path, symbols in forbidden_symbols.items():
        assert not (_defined_symbols(relative_path) & symbols), relative_path

    forbidden_fragments = {
        "lightrag/kg/qdrant_impl.py": {"DataMigrationError", "legacy_collection"},
        "lightrag/kg/neo4j_impl.py": {"legacy_index_name"},
    }
    for relative_path, fragments in forbidden_fragments.items():
        source = (REPO_ROOT / relative_path).read_text()
        assert not ({fragment for fragment in fragments if fragment in source}), (
            relative_path
        )


def test_storage_migration_tools_are_absent():
    forbidden_paths = (
        "lightrag/tools/migrate_llm_cache.py",
        "lightrag/tools/prepare_qdrant_legacy_data.py",
        "lightrag/tools/README_MIGRATE_LLM_CACHE.md",
        "lightrag/tools/rebuild_vdb.py",
        "lightrag/tools/README_REBUILD_VDB.md",
        "tests/tools/test_rebuild_vdb.py",
        "tests/kg/qdrant_impl/test_qdrant_migration.py",
        "tests/llm/test_dimension_mismatch.py",
        "tests/llm/test_no_model_suffix_safety.py",
        "tests/kg/milvus_impl/test_milvus_migration_retry.py",
        "tests/kg/milvus_impl/test_milvus_migration_memory.py",
        "tests/kg/opensearch_impl/test_llm_cache_tools_opensearch.py",
    )
    assert not [path for path in forbidden_paths if (REPO_ROOT / path).exists()]


def test_legacy_workspace_launchers_and_service_templates_are_absent():
    forbidden_paths = (
        "lightrag.service.example",
        "ops/lib/project-env.sh",
        "ops/lightrag_mcp.py",
        "ops/projects/code.env",
        "ops/projects/ships.env",
        "ops/start.sh",
        "ops/systemd/lightrag@.service",
    )

    assert not [path for path in forbidden_paths if (REPO_ROOT / path).exists()]
    assert (REPO_ROOT / "ops/.env.enc").exists()
    assert (REPO_ROOT / "ops/.age-recipient").exists()


def test_deployment_surfaces_use_exact_generation_storage_contract():
    required_storage = {
        "LIGHTRAG_KV_STORAGE=PGKVStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage",
        "LIGHTRAG_GRAPH_STORAGE=PGGraphStorage",
        "LIGHTRAG_VECTOR_STORAGE=PGVectorStorage",
    }
    for relative_path in ("env.example", "env.docker-compose-full"):
        source = (REPO_ROOT / relative_path).read_text()
        assert required_storage <= set(source.splitlines())
        assert "POSTGRES_WORKSPACE" not in source

    full_compose = (REPO_ROOT / "docker-compose-full.yml").read_text()
    assert "  neo4j:" not in full_compose
    assert "  milvus:" not in full_compose
    assert "dockerfile: Dockerfile.postgres" in full_compose

    for relative_path in (
        "docker-compose.yml",
        "docker-compose.podman.yml",
        "docker-compose-full.yml",
    ):
        assert "data/inputs" not in (REPO_ROOT / relative_path).read_text()


def test_only_api_graph_visualization_example_remains():
    examples = REPO_ROOT / "examples"
    assert [path.name for path in examples.glob("graph_visual_with_*.py")] == [
        "graph_visual_with_opensearch.py"
    ]

    api_example = (examples / "graph_visual_with_opensearch.py").read_text()
    compile(api_example, "graph_visual_with_opensearch.py", "exec")
    assert "read_graphml" not in api_example
    assert 'f"{server_url}/graphs"' in api_example


def test_generic_fork_has_no_oceanstack_ontology_profile():
    samples = REPO_ROOT / "prompts" / "samples"
    assert not list(samples.glob("oceanstack*"))


def test_api_routers_expose_only_plane_surface():
    """Only the plane router (mounted) and the retained internal document
    ingestion machinery (never mounted) may exist under api/routers."""
    routers = REPO_ROOT / "lightrag" / "api" / "routers"
    assert sorted(path.name for path in routers.glob("*.py")) == [
        "__init__.py",
        "document_routes.py",
        "plane_routes.py",
    ]

    server_source = (REPO_ROOT / "lightrag" / "api" / "lightrag_server.py").read_text()
    for removed in (
        "create_document_routes",
        "create_query_routes",
        "create_graph_routes",
        "create_map_routes",
        "OllamaAPI",
    ):
        assert removed not in server_source
