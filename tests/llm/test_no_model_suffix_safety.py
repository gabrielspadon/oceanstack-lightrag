"""
Tests for safety when model suffix is absent (no model_name provided).

This test module verifies that the system correctly handles the case when
no model_name is provided, preventing accidental deletion of the only table/collection
on restart.

Critical Bug: When model_suffix is empty, table_name == legacy_table_name.
On second startup, Case 1 logic would delete the only table/collection thinking
it's "legacy", causing all subsequent operations to fail.
"""

from unittest.mock import MagicMock, patch

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage


class TestNoModelSuffixSafety:
    """Test suite for preventing data loss when model_suffix is absent."""

    def test_qdrant_no_suffix_second_startup(self):
        """
        Test Qdrant doesn't delete collection on second startup when no model_name.

        Scenario:
        1. First startup: Creates collection without suffix
        2. Collection is empty
        3. Second startup: Should NOT delete the collection

        Bug: Without fix, Case 1 would delete the only collection.
        """
        from qdrant_client import models

        client = MagicMock()

        # Simulate second startup: collection already exists and is empty
        # IMPORTANT: Without suffix, collection_name == legacy collection name
        collection_name = "lightrag_vdb_chunks"  # No suffix, same as legacy

        # Both exist (they're the same collection)
        client.collection_exists.return_value = True

        # Collection is empty
        client.count.return_value.count = 0

        # Patch _find_legacy_collection to return the SAME collection name
        # This simulates the scenario where new collection == legacy collection
        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_vdb_chunks",  # Same as collection_name
        ):
            # Call setup_collection
            # This should detect that new == legacy and skip deletion
            QdrantVectorDBStorage.setup_collection(
                client,
                collection_name,
                namespace="chunks",
                workspace="_",
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
                model_suffix="",  # Empty suffix to simulate no model_name provided
            )

        # CRITICAL: Collection should NOT be deleted
        client.delete_collection.assert_not_called()

        # Verify we returned early (skipped Case 1 cleanup)
        # The collection_exists was checked, but we didn't proceed to count
        # because we detected same name
        assert client.collection_exists.call_count >= 1

    def test_qdrant_with_suffix_case1_still_works(self):
        """
        Test that Case 1 cleanup still works when there IS a suffix.

        This ensures our fix doesn't break the normal Case 1 scenario.
        """
        from qdrant_client import models

        client = MagicMock()

        # Different names (normal case)
        collection_name = "lightrag_vdb_chunks_ada_002_1536d"  # With suffix
        legacy_collection = "lightrag_vdb_chunks"  # Without suffix

        # Setup: both exist
        def collection_exists_side_effect(name):
            return name in [collection_name, legacy_collection]

        client.collection_exists.side_effect = collection_exists_side_effect

        # Legacy is empty
        client.count.return_value.count = 0

        # Call setup_collection
        QdrantVectorDBStorage.setup_collection(
            client,
            collection_name,
            namespace="chunks",
            workspace="_",
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                payload_m=16,
                m=0,
            ),
            model_suffix="ada_002_1536d",
        )

        # SHOULD delete legacy (normal Case 1 behavior)
        client.delete_collection.assert_called_once_with(
            collection_name=legacy_collection
        )
