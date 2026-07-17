"""Greenfield repository surface excludes legacy migration and local ontology assets."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
