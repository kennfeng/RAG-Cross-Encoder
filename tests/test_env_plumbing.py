import os
from unittest.mock import patch

import pytest

from rag_pipeline import LangChainRAG

ATLAS_ENV_VARS = (
    "ATLAS_DB_PATH",
    "ATLAS_RERANKER_MODEL",
    "ATLAS_LLM_MODEL",
    "ATLAS_PROVIDER",
    "ATLAS_N_RESULTS",
    "ATLAS_TOP_N",
    "ATLAS_OLLAMA_BASE_URL",
)


@pytest.fixture
def clear_atlas_env():
    saved = {name: os.environ.get(name) for name in ATLAS_ENV_VARS}
    for name in ATLAS_ENV_VARS:
        os.environ.pop(name, None)
    yield
    for name, old in saved.items():
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def test_env_db_path_sets_ingestor_path(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor") as mock_ingestor,
        patch("rag_pipeline.AtlasReRanker"),
        patch("rag_pipeline.create_llm"),
        patch.dict(os.environ, {"ATLAS_DB_PATH": "/data/atlas_db"}),
    ):
        LangChainRAG.from_defaults()
    mock_ingestor.assert_called_once_with(db_path="/data/atlas_db")


def test_env_reranker_model_sets_model_name(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor"),
        patch("rag_pipeline.AtlasReRanker") as mock_ranker,
        patch("rag_pipeline.create_llm"),
        patch.dict(os.environ, {"ATLAS_RERANKER_MODEL": "BAAI/bge-reranker-large"}),
    ):
        LangChainRAG.from_defaults()
    mock_ranker.assert_called_once_with(model_name="BAAI/bge-reranker-large")


def test_env_llm_model_and_provider_override_defaults(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor"),
        patch("rag_pipeline.AtlasReRanker"),
        patch("rag_pipeline.create_llm") as mock_llm,
        patch.dict(
            os.environ,
            {"ATLAS_LLM_MODEL": "gpt-4o-mini", "ATLAS_PROVIDER": "openai"},
        ),
    ):
        LangChainRAG.from_defaults()
    mock_llm.assert_called_once_with(
        provider="openai", model_name="gpt-4o-mini", temperature=0.0
    )


def test_env_n_results_and_top_n_wire_adapters(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor"),
        patch("rag_pipeline.AtlasReRanker"),
        patch("rag_pipeline.create_llm"),
        patch.dict(os.environ, {"ATLAS_N_RESULTS": "7", "ATLAS_TOP_N": "2"}),
    ):
        pipeline = LangChainRAG.from_defaults()
    assert pipeline.retriever.n_results == 7
    assert pipeline.reranker.top_n == 2


def test_explicit_args_override_env(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor") as mock_ingestor,
        patch("rag_pipeline.AtlasReRanker") as mock_ranker,
        patch("rag_pipeline.create_llm") as mock_llm,
        patch.dict(
            os.environ,
            {
                "ATLAS_DB_PATH": "/env/db",
                "ATLAS_RERANKER_MODEL": "env-reranker",
                "ATLAS_LLM_MODEL": "env-llm",
                "ATLAS_PROVIDER": "openai",
                "ATLAS_N_RESULTS": "7",
                "ATLAS_TOP_N": "2",
                "ATLAS_OLLAMA_BASE_URL": "http://env",
            },
        ),
    ):
        pipeline = LangChainRAG.from_defaults(
            db_path="/arg/db",
            model_name="arg-reranker",
            llm_model_name="arg-llm",
            provider="ollama",
            n_results=0,
            top_n=5,
            base_url="http://arg",
        )
    mock_ingestor.assert_called_once_with(db_path="/arg/db")
    mock_ranker.assert_called_once_with(model_name="arg-reranker")
    mock_llm.assert_called_once_with(
        provider="ollama",
        model_name="arg-llm",
        temperature=0.0,
        base_url="http://arg",
    )
    assert pipeline.retriever.n_results == 0
    assert pipeline.reranker.top_n == 5


def test_env_base_url_forwarded_to_create_llm(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor"),
        patch("rag_pipeline.AtlasReRanker"),
        patch("rag_pipeline.create_llm") as mock_llm,
        patch.dict(os.environ, {"ATLAS_OLLAMA_BASE_URL": "http://localhost:11434"}),
    ):
        LangChainRAG.from_defaults()
    mock_llm.assert_called_once()
    assert mock_llm.call_args.kwargs["base_url"] == "http://localhost:11434"


def test_env_base_url_unset_omits_base_url_kwarg(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor"),
        patch("rag_pipeline.AtlasReRanker"),
        patch("rag_pipeline.create_llm") as mock_llm,
    ):
        LangChainRAG.from_defaults()
    mock_llm.assert_called_once_with(
        provider="ollama", model_name="llama3.2:1b", temperature=0.0
    )


def test_empty_env_uses_today_defaults(clear_atlas_env):
    with (
        patch("rag_pipeline.AtlasIngestor") as mock_ingestor,
        patch("rag_pipeline.AtlasReRanker") as mock_ranker,
        patch("rag_pipeline.create_llm") as mock_llm,
    ):
        LangChainRAG.from_defaults()
    mock_ingestor.assert_called_once_with(db_path="./atlas_db")
    mock_ranker.assert_called_once_with(model_name="BAAI/bge-reranker-base")
    mock_llm.assert_called_once_with(
        provider="ollama", model_name="llama3.2:1b", temperature=0.0
    )
