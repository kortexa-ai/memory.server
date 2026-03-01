"""Store endpoint — save a piece of knowledge to an agent's memory."""

from fastapi import APIRouter

from ..embeddings import get_embedder
from ..models import StoreRequest, StoreResponse
from ..storage.sqlite import store_memory

router = APIRouter()


@router.post("/v1/memory/store", response_model=StoreResponse)
async def store(req: StoreRequest) -> StoreResponse:
    """Store a piece of knowledge in an agent's memory."""
    embedder = get_embedder()
    vec = embedder.embed_texts([req.content])[0]
    embedding = embedder.to_bytes(vec)

    memory_id = store_memory(
        agent_id=req.agent_id,
        content=req.content,
        source=req.source,
        tags=req.tags,
        embedding=embedding,
    )
    return StoreResponse(agent_id=req.agent_id, memory_id=memory_id)
