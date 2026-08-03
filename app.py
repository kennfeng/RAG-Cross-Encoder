import os
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from main import AtlasRAG


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


def check_db(db_path: str) -> bool:
    return Path(db_path).is_dir()


def check_ollama(base_url: str | None) -> bool:
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def get_rag(request: Request) -> AtlasRAG:
    state = request.app.state
    if getattr(state, "rag", None) is None:
        with state.lock:
            if getattr(state, "rag", None) is None:
                if state.rag_factory is not None:
                    state.rag = state.rag_factory()
                else:
                    state.rag = AtlasRAG()
    return state.rag


def create_app(rag_factory: Any = None) -> FastAPI:
    app = FastAPI()
    app.state.rag = None
    app.state.rag_factory = rag_factory
    app.state.lock = Lock()

    @app.get("/health")
    def health() -> JSONResponse:
        db_path = os.environ.get("ATLAS_DB_PATH", "./atlas_db")
        base_url = os.environ.get("ATLAS_OLLAMA_BASE_URL")
        db_ready = check_db(db_path)
        ollama_reachable = check_ollama(base_url)
        pipeline_initialized = getattr(app.state, "rag", None) is not None
        status = "ok" if db_ready and ollama_reachable else "degraded"
        return JSONResponse(
            content={
                "status": status,
                "pipeline_initialized": pipeline_initialized,
                "db_ready": db_ready,
                "ollama_reachable": ollama_reachable,
            },
            status_code=200 if status == "ok" else 503,
        )

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict:
        rag = get_rag(request)
        return rag.ask(body.query)

    return app


app = create_app()
