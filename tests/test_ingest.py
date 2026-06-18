from unittest.mock import MagicMock
from ingest import AtlasIngestor


def make_mock_ingestor():
    return AtlasIngestor(db_path="test_db")

def test_add_documents_generate_default_ids():
    ingestor = make_mock_ingestor()
    ingestor.collection.add = MagicMock()

    ingestor.add_documents(["doc1", "doc2"])
    ingestor.collection.add.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=None,
        ids=["id_0", "id_1"],
    )

def test_add_documents_with_custom_ids():
    ingestor = make_mock_ingestor()
    ingestor.collection.add = MagicMock()

    ingestor.add_documents(["doc1", "doc2"], ids=["custom_id_1", "custom_id_2"])
    ingestor.collection.add.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=None,
        ids=["custom_id_1", "custom_id_2"],
    )