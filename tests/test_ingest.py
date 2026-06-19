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

def test_search_returns_documents():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(return_value={
        'documents': [["doc1", "doc2"]],
        'ids': [["id_0", "id_1"]]
    })

    results = ingestor.search("query")
    assert results == ["doc1", "doc2"]

def test_search_with_ids_returns_ids_and_documents():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(return_value={
        'documents': [["doc1", "doc2"]],
        'ids': [["id_0", "id_1"]]
    })

    results = ingestor.search_with_ids("query")
    assert results == [("id_0", "doc1"), ("id_1", "doc2")]

def test_search_with_ids_empty_results():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(return_value={
        'documents': [[]],
        'ids': [[]]
    })

    results = ingestor.search_with_ids("query")
    assert results == []