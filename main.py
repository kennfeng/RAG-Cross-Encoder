from typing import Any

import httpx

from rag_pipeline import LangChainRAG


class AtlasRAG:
    def __init__(
        self,
        pipeline: LangChainRAG | None = None,
        model: str = "llama3.2:1b",
        provider: str = "ollama",
        sample_docs: list[str] | None = None,
        n_results: int = 10,
        top_n: int = 3,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        print("--- Initializing RAG System ---")
        self.pipeline = pipeline or LangChainRAG.from_defaults(
            llm_model_name=model,
            provider=provider,
            sample_docs=sample_docs,
            n_results=n_results,
            top_n=top_n,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def ask(self, query: str) -> dict[str, Any]:
        print(f"\n[QUERY]: {query}")
        try:
            return self.pipeline.ask(query)
        except (ConnectionError, httpx.TransportError) as e:
            return {
                "answer": f"ERROR: Could not connect to Ollama ({type(e).__name__}: {e})",
                "source_documents": [],
            }


if __name__ == "__main__":
    rag = AtlasRAG()
    while True:
        try:
            user_input = input("\nAsk Atlas (or type 'exit'): ")
            if user_input.lower() == "exit":
                break
            if not user_input.strip():
                continue

            result = rag.ask(user_input)
            print(f"\n--- RESPONSE ---\n{result['answer']}")

            if result["source_documents"]:
                print("\n--- SOURCES (Re-ranked) ---")
                for i, src in enumerate(result["source_documents"]):
                    print(f"{i + 1}. [{src['score']:.4f}] {src['document']}")
        except (EOFError, KeyboardInterrupt):
            break
