"""Memory server — neural memory for AI agents."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api import ingest, recall, status, store, train
from .config import settings
from .embeddings import get_embedder
from .llm.engine import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("memory-server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info(f"Memory server starting on {settings.host}:{settings.port}")
    logger.info(f"Data dir: {settings.data_dir}")
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    # Load embedding model (fast, ~33MB download on first run)
    logger.info("Loading embedding model...")
    get_embedder().warm_up()
    logger.info("Embedding model ready")

    # Launch mlx-lm server
    engine = get_engine()
    logger.info(f"Launching mlx-lm server (model: {settings.model_repo}, port: {settings.model_server_port})...")
    await engine.connect()

    yield

    await engine.disconnect()
    logger.info("Memory server shutting down")


app = FastAPI(
    title="Memory Server",
    description="Neural memory for AI agents — persistent knowledge via LLM-powered recall",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(recall.router)
app.include_router(store.router)
app.include_router(ingest.router)
app.include_router(status.router)
app.include_router(train.router)


@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "status": "ok",
        "engine_loaded": engine.loaded,
    }


@app.post("/v1/internal/generate")
async def internal_generate(request: dict):
    """Internal endpoint for training data generation.

    Used by the training subprocess to generate Q&A pairs using the server's LLM.
    Not intended for external use.
    """
    engine = get_engine()
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 1000)
    temperature = request.get("temperature", 0.7)

    content = await engine.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {"content": content}


def run():
    """Entry point for the memory-server command."""
    uvicorn.run(
        "memory_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
