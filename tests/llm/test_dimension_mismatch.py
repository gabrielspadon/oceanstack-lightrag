"""
Tests for dimension mismatch handling during migration.

This test module verifies that both PostgreSQL and Qdrant storage backends
properly detect and handle vector dimension mismatches when migrating from
legacy collections/tables to new ones with different embedding models.
"""

import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage
from lightrag.exceptions import DataMigrationError


# Note: Tests should use proper table names that have DDL templates
# Valid base tables: LIGHTRAG_VDB_CHUNKS, LIGHTRAG_VDB_ENTITIES, LIGHTRAG_VDB_RELATIONSHIPS,
#                    LIGHTRAG_DOC_CHUNKS, LIGHTRAG_DOC_FULL_DOCS, LIGHTRAG_DOC_TEXT_CHUNKS


class TestQdrantDimensionMismatch:
    """Test suite for Qdrant dimension mismatch handling."""

    def test_qdrant_dimension_mismatch_raises_error(self):
        """
        Test that Qdrant raises DataMigrationError when dimensions don't match.

        Scenario: Legacy collection has 1536d vectors, new model expects 3072d.
        Expected: DataMigrationError is raised to prevent data corruption.
        """
        from qdrant_client import models

        # Setup mock client
        client = MagicMock()

        # Mock legacy collection with 1536d vectors
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        # Setup collection existence checks
        def collection_exists_side_effect(name):
            if (
                name == "lightrag_vdb_chunks"
            ):  # legacy (matches _find_legacy_collection pattern)
                return True
            elif name == "lightrag_chunks_model_3072d":  # new
                return False
            return False

        client.collection_exists.side_effect = collection_exists_side_effect
        client.get_collection.return_value = legacy_collection_info
        client.count.return_value.count = 100  # Legacy has data

        # Patch _find_legacy_collection to return the legacy collection name
        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_vdb_chunks",
        ):
            # Call setup_collection with 3072d (different from legacy 1536d)
            # Should raise DataMigrationError due to dimension mismatch
            with pytest.raises(DataMigrationError) as exc_info:
                QdrantVectorDBStorage.setup_collection(
                    client,
                    "lightrag_chunks_model_3072d",
                    namespace="chunks",
                    workspace="test",
                    vectors_config=models.VectorParams(
                        size=3072, distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0,
                    ),
                    model_suffix="model_3072d",
                )

        # Verify error message contains dimension information
        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)

        # Verify new collection was NOT created (error raised before creation)
        client.create_collection.assert_not_called()

        # Verify migration was NOT attempted
        client.scroll.assert_not_called()
        client.upsert.assert_not_called()

    def test_qdrant_dimension_match_proceed_migration(self):
        """
        Test that Qdrant proceeds with migration when dimensions match.

        Scenario: Legacy collection has 1536d vectors, new model also expects 1536d.
        Expected: Migration proceeds normally.
        """
        from qdrant_client import models

        client = MagicMock()

        # Mock legacy collection with 1536d vectors (matching new)
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        def collection_exists_side_effect(name):
            if name == "lightrag_chunks":  # legacy
                return True
            elif name == "lightrag_chunks_model_1536d":  # new
                return False
            return False

        client.collection_exists.side_effect = collection_exists_side_effect
        client.get_collection.return_value = legacy_collection_info

        # Track whether upsert has been called (migration occurred)
        migration_done = {"value": False}

        def upsert_side_effect(*args, **kwargs):
            migration_done["value"] = True
            return MagicMock()

        client.upsert.side_effect = upsert_side_effect

        # Mock count to return different values based on collection name and migration state
        # Before migration: new collection has 0 records
        # After migration: new collection has 1 record (matching migrated data)
        def count_side_effect(collection_name, **kwargs):
            result = MagicMock()
            if collection_name == "lightrag_chunks":  # legacy
                result.count = 1  # Legacy has 1 record
            elif collection_name == "lightrag_chunks_model_1536d":  # new
                # Return 0 before migration, 1 after migration
                result.count = 1 if migration_done["value"] else 0
            else:
                result.count = 0
            return result

        client.count.side_effect = count_side_effect

        # Mock scroll to return sample data (1 record for easier verification)
        sample_point = MagicMock()
        sample_point.id = "test_id"
        sample_point.vector = [0.1] * 1536
        sample_point.payload = {"id": "test"}
        client.scroll.return_value = ([sample_point], None)

        # Mock _find_legacy_collection to return the legacy collection name
        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_chunks",
        ):
            # Call setup_collection with matching 1536d
            QdrantVectorDBStorage.setup_collection(
                client,
                "lightrag_chunks_model_1536d",
                namespace="chunks",
                workspace="test",
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
                model_suffix="model_1536d",
            )

        # Verify migration WAS attempted
        client.create_collection.assert_called_once()
        client.scroll.assert_called()
        client.upsert.assert_called()
