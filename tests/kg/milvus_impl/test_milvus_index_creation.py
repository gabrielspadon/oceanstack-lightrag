"""
Tests for Milvus index creation behavior

This test suite validates:
1. Current IndexParams construction
2. Vector and scalar index creation failures are surfaced to callers
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
from lightrag.kg.milvus_impl import (
    MILVUS_MAX_VARCHAR_BYTES,
    MilvusIndexConfig,
    MilvusVectorDBStorage,
)


def _make_storage(namespace="entities"):
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=set(),
    )


def _field_max_length(field):
    return int(field.params["max_length"])


def _collection_info(field_names):
    fields = [
        {
            "name": "id",
            "type": "VarChar",
            "is_primary": True,
            "params": {"max_length": 64},
        },
        {"name": "vector", "type": "FloatVector", "params": {"dim": 128}},
        {"name": "created_at", "type": "Int64"},
    ]
    fields.extend(
        {
            "name": field_name,
            "type": "VarChar",
            "params": {
                "max_length": {
                    "entity_name": 512,
                    "src_id": 512,
                    "tgt_id": 512,
                    "full_doc_id": 64,
                    "file_path": 32768,
                }.get(field_name, MILVUS_MAX_VARCHAR_BYTES)
            },
        }
        for field_name in field_names
    )
    return {"fields": fields}


class _EmbeddingFunc:
    def __init__(self, dim=128, model_name="text-embedding-3-small"):
        self.embedding_dim = dim
        self.model_name = model_name


def _make_model_storage(namespace="entities", workspace="test_workspace", dim=128):
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=_EmbeddingFunc(dim=dim),
        meta_fields=set(),
    )


def _wire_collection_state(storage, collections, describe_by_name=None):
    storage._client = MagicMock()
    describe_by_name = describe_by_name or {}

    def has_collection(collection_name):
        return collection_name in collections

    def create_collection(collection_name, schema):
        collections.add(collection_name)

    def drop_collection(collection_name):
        collections.discard(collection_name)

    def rename_collection(source, target):
        collections.discard(source)
        collections.add(target)

    def describe_collection(collection_name):
        return describe_by_name.get(collection_name, _collection_info([]))

    storage._client.has_collection.side_effect = has_collection
    storage._client.create_collection.side_effect = create_collection
    storage._client.drop_collection.side_effect = drop_collection
    storage._client.rename_collection.side_effect = rename_collection
    storage._client.describe_collection.side_effect = describe_collection
    return storage._client


@pytest.mark.offline
class TestMilvusIndexCreation:
    """Test index creation behavior and error handling"""

    @pytest.mark.parametrize(
        ("namespace", "expected_fields"),
        [
            ("entities", {"content", "source_id"}),
            ("relationships", {"content", "source_id"}),
            ("chunks", {"content"}),
        ],
    )
    def test_schema_promotes_core_metadata_fields(self, namespace, expected_fields):
        storage = _make_storage(namespace=namespace)

        fields_by_name = {
            field.name: field for field in storage._create_schema_for_namespace().fields
        }

        assert expected_fields.issubset(fields_by_name)
        for field_name in expected_fields:
            assert (
                _field_max_length(fields_by_name[field_name])
                == MILVUS_MAX_VARCHAR_BYTES
            )

    def test_model_suffix_collection_naming_with_workspace(self):
        storage = MilvusVectorDBStorage(
            namespace="chunks",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(
                dim=3072, model_name="text-embedding-3-large"
            ),
            meta_fields=set(),
        )

        assert storage.final_namespace == "space1_chunks_text_embedding_3_large_3072d"

    def test_model_suffix_collection_naming_without_workspace(self):
        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(dim=2560, model_name=" qwen3-embedding:4b "),
            meta_fields=set(),
        )

        assert storage.final_namespace == "entities_qwen3_embedding_4b_2560d"

    @pytest.mark.parametrize("model_name", ["", "   ", 123])
    def test_missing_or_invalid_model_name_uses_base_collection_name(self, model_name):
        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=_EmbeddingFunc(model_name=model_name),
            meta_fields=set(),
        )

        assert storage.model_suffix is None
        assert storage.final_namespace == "space1_entities"

    def test_creates_suffixed_collection_when_no_collection_exists(self):
        storage = _make_model_storage()
        client = _wire_collection_state(storage, set())

        with patch.object(storage, "_create_indexes_after_collection"):
            storage._create_collection_if_not_exist()

        client.create_collection.assert_called_once()
        assert client.create_collection.call_args.kwargs["collection_name"] == (
            storage.final_namespace
        )
        client.query_iterator.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_existing_suffixed_collection_is_validated_and_used(self):
        storage = _make_model_storage()
        client = _wire_collection_state(
            storage,
            {storage.final_namespace},
            {
                storage.final_namespace: _collection_info(
                    ["entity_name", "content", "source_id", "file_path"]
                )
            },
        )

        storage._create_collection_if_not_exist()
        client.create_collection.assert_not_called()
        client.load_collection.assert_called_with(storage.final_namespace)

    def test_vector_index_creation_failure_is_raised(self):
        """Vector index creation failures are raised to the caller."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        mock_client.create_index.side_effect = Exception("Index creation failed")

        with pytest.raises(Exception, match="Index creation failed"):
            storage._create_indexes_after_collection()

    def test_scalar_index_creation_failure_is_raised(self):
        """Scalar index creation failures are raised to the caller."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "AUTOINDEX",  # No custom vector index
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        mock_client.create_index.side_effect = [
            None,
            Exception("Scalar index creation failed"),
        ]

        with pytest.raises(Exception, match="Scalar index creation failed"):
            storage._create_indexes_after_collection()

    def test_build_index_params_uses_passed_index_params(self):
        """Test that build_index_params uses the passed index_params parameter (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            hnsw_m=32,
            hnsw_ef_construction=256,
        )

        mock_index_params = MagicMock()

        # Call build_index_params with the mock_index_params
        result = config.build_index_params(mock_index_params)

        # Verify that it used the passed index_params
        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once()

    def test_build_index_params_raises_when_index_params_is_none_for_custom_type(self):
        """Test that build_index_params raises RuntimeError when index_params is None for custom types (P1 fix)"""
        config = MilvusIndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
        )

        with pytest.raises(RuntimeError, match="IndexParams not available"):
            config.build_index_params(None)

    def test_build_index_params_raises_for_autoindex_when_index_params_is_none(
        self,
    ):
        """AUTOINDEX also requires the current IndexParams API."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        with pytest.raises(RuntimeError, match="IndexParams not available"):
            config.build_index_params(None)

    def test_build_index_params_autoindex_uses_index_params_object(self):
        """Test AUTOINDEX still creates an explicit vector index when IndexParams is available."""
        config = MilvusIndexConfig(
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        mock_index_params = MagicMock()

        result = config.build_index_params(mock_index_params)

        assert result == mock_index_params
        mock_index_params.add_index.assert_called_once_with(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
            params={},
        )

    def test_create_indexes_uses_current_index_params_api(self):
        """Collection setup uses the current IndexParams API for every index."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_index_params = MagicMock()
        mock_client.prepare_index_params.return_value = mock_index_params

        storage._client = mock_client
        storage.final_namespace = "test_entities"

        storage._create_indexes_after_collection()

        assert mock_client.prepare_index_params.call_count == 2
        assert mock_client.create_index.call_count == 2

    def test_version_probing_only_for_hnsw_sq(self):
        """Test that get_server_version is only called when index type requires it (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        # Test with HNSW (no version requirement) - should NOT call get_server_version
        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version should NOT be called for HNSW
        mock_client.get_server_version.assert_not_called()

    def test_version_probing_called_for_hnsw_sq(self):
        """Test that get_server_version IS called when HNSW_SQ is configured (P2 fix)"""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="test_workspace",
            global_config={
                "embedding_batch_num": 100,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                    "index_type": "HNSW_SQ",
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        mock_client = MagicMock()
        mock_client.get_server_version.return_value = "2.6.9"
        storage._client = mock_client

        # Mock the init lock as an async context manager
        mock_lock = AsyncMock()

        with patch(
            "lightrag.kg.milvus_impl.get_data_init_lock", return_value=mock_lock
        ):
            with patch.object(storage, "_create_collection_if_not_exist"):
                asyncio.run(storage.initialize())

        # get_server_version SHOULD be called for HNSW_SQ
        mock_client.get_server_version.assert_called_once()

    def test_initialize_creates_missing_database_before_collection_setup(self):
        """Test that initialize bootstraps a missing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ) as mock_client_cls:
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        mock_client_cls.assert_called_once_with(
            uri="http://milvus:19530",
            user=None,
            password=None,
            token=None,
        )
        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_called_once_with("lightrag")
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_initialize_uses_existing_database_without_recreating_it(self):
        """Test that initialize switches to an existing configured Milvus database."""
        from unittest.mock import AsyncMock

        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="test_entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )

        bootstrap_client = MagicMock()
        bootstrap_client.list_databases.return_value = ["default", "lightrag"]
        mock_lock = AsyncMock()

        with patch.dict(
            "os.environ",
            {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_DB_NAME": "lightrag",
            },
            clear=False,
        ):
            with patch(
                "lightrag.kg.milvus_impl.MilvusClient", return_value=bootstrap_client
            ):
                with patch(
                    "lightrag.kg.milvus_impl.get_data_init_lock",
                    return_value=mock_lock,
                ):
                    with patch.object(storage, "_create_collection_if_not_exist"):
                        asyncio.run(storage.initialize())

        bootstrap_client.list_databases.assert_called_once_with()
        bootstrap_client.create_database.assert_not_called()
        bootstrap_client.use_database.assert_called_once_with("lightrag")

    def test_existing_collection_schema_validation_failure_raises(self):
        """Invalid existing schemas stop initialization."""
        mock_embedding_func = MagicMock()
        mock_embedding_func.embedding_dim = 128

        storage = MilvusVectorDBStorage(
            namespace="entities",
            workspace="space1",
            global_config={
                "embedding_batch_num": 100,
                "working_dir": "/tmp/lightrag",
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.3,
                },
            },
            embedding_func=mock_embedding_func,
            meta_fields=set(),
        )
        storage.final_namespace = "space1_entities"
        storage._client = MagicMock()
        storage._client.has_collection.return_value = True

        storage._client.describe_collection.return_value = {}

        with pytest.raises(ValueError, match="Vector field not found"):
            storage._create_collection_if_not_exist()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
