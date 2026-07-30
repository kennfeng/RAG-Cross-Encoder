from collections.abc import Generator
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ingest import AtlasIngestor
from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    create_llm,
)
from reranker import AtlasReRanker

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question based only on the "
    "provided context. If the context does not contain enough information to answer "
    'the question, say "I don\'t have enough information to answer this question." '
    "Do not make up or infer information beyond what is stated in the context. "
    "Cite specific parts of the context when possible."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


class LangChainRAG:
    def __init__(
        self,
        retriever: ChromaRetrieverAdapter,
        reranker: CrossEncoderRerankerAdapter,
        llm: Runnable,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_TEMPLATE),
            ]
        )
        self._chain: Runnable | None = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self.prompt | self.llm | StrOutputParser()
        return self._chain

    @classmethod
    def from_defaults(
        cls,
        db_path: str = "./atlas_db",
        model_name: str = "BAAI/bge-reranker-base",
        llm_model_name: str = "llama3.2:1b",
        provider: str = "ollama",
        n_results: int = 10,
        top_n: int = 3,
        sample_docs: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> "LangChainRAG":
        ingestor = AtlasIngestor(db_path=db_path)
        if ingestor.collection.count() == 0:
            sample_kb = sample_docs or [
                "PyTorch is an open source machine learning framework based on the Torch library.",
                "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
                "RAG stands for Retrieval-Augmented Generation, a technique to provide external data to LLMs.",
                "A Cross-Encoder is a type of deep learning model that processes pairs of inputs simultaneously to determine relevance.",
                "Vector databases like ChromaDB store high-dimensional embeddings for fast similarity search.",
                "Ollama allows you to run large language models locally on your machine.",
                "Two-stage RAG uses a fast retriever (Stage 1) and a precise re-ranker (Stage 2) for better accuracy.",
            ]
            ingestor.add_documents(sample_kb)

        ranker = AtlasReRanker(model_name=model_name)
        retriever = ChromaRetrieverAdapter(ingestor, n_results=n_results)
        reranker = CrossEncoderRerankerAdapter(ranker, top_n=top_n)

        llm_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            llm_kwargs["num_predict" if provider == "ollama" else "max_tokens"] = (
                max_tokens
            )

        primary_llm = create_llm(
            provider=provider, model_name=llm_model_name, **llm_kwargs
        )
        fallback_llm = create_llm(
            provider=provider, model_name=llm_model_name, **llm_kwargs
        )
        llm = primary_llm.with_fallbacks([fallback_llm])

        return cls(retriever, reranker, llm)

    def _prepare(self, query: str) -> dict[str, Any]:
        candidates = self.retriever.get_relevant_documents_with_ids(query)
        if not candidates:
            return {"context": "", "source_documents": []}

        ranked_results = self.reranker.rerank(query, candidates)
        context_docs = [res["document"] for res in ranked_results]
        context = "\n\n".join(context_docs)
        return {"context": context, "source_documents": ranked_results}

    def ask(self, query: str) -> dict[str, Any]:
        prepared = self._prepare(query)
        if not prepared["source_documents"]:
            return {
                "answer": "I couldn't find any relevant documents in the database.",
                "source_documents": [],
            }

        answer = self.chain.invoke(
            {
                "context": prepared["context"],
                "query": query,
            }
        )
        return {
            "answer": answer,
            "source_documents": prepared["source_documents"],
        }

    def ask_stream(self, query: str) -> Generator[str, None, None]:
        prepared = self._prepare(query)
        if not prepared["source_documents"]:
            yield "I couldn't find any relevant documents in the database."
            return

        yield from self.chain.stream(
            {
                "context": prepared["context"],
                "query": query,
            }
        )
