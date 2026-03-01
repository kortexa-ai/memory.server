"""Status and admin endpoints — check agent memory state, reindex embeddings."""

from fastapi import APIRouter, BackgroundTasks

from ..config import settings
from ..embeddings import get_embedder
from ..llm.engine import get_engine
from ..models import AgentStatus, ReindexResponse
from ..storage.sqlite import (
    batch_update_embeddings,
    get_all_tags,
    get_last_stored,
    get_memory_count,
    get_unembedded_count,
    get_unembedded_memories,
)

router = APIRouter()


@router.get("/v1/memory/status/{agent_id}", response_model=AgentStatus)
async def agent_status(agent_id: str) -> AgentStatus:
    """Get status of an agent's memory store."""
    engine = get_engine()
    adapter_path = engine.get_adapter_path(agent_id)

    return AgentStatus(
        agent_id=agent_id,
        memory_count=get_memory_count(agent_id),
        unembedded_count=get_unembedded_count(agent_id),
        has_adapter=adapter_path is not None,
        last_stored=get_last_stored(agent_id),
        tags=get_all_tags(agent_id),
    )


async def _reindex_agent(agent_id: str) -> int:
    """Compute embeddings for all unembedded memories."""
    embedder = get_embedder()
    total = 0

    while True:
        batch = get_unembedded_memories(agent_id, limit=settings.embedding_batch_size)
        if not batch:
            break

        ids, texts = zip(*batch)
        vecs = embedder.embed_texts(list(texts))
        updates = [(mid, embedder.to_bytes(vec)) for mid, vec in zip(ids, vecs)]
        batch_update_embeddings(agent_id, updates)
        total += len(updates)

    return total


@router.post("/v1/memory/reindex/{agent_id}", response_model=ReindexResponse)
async def reindex(agent_id: str, background_tasks: BackgroundTasks) -> ReindexResponse:
    """Compute embeddings for all unembedded memories.

    Runs in the background — returns immediately with the count of memories
    that need reindexing.
    """
    remaining = get_unembedded_count(agent_id)

    if remaining == 0:
        return ReindexResponse(agent_id=agent_id, reindexed=0, remaining=0)

    background_tasks.add_task(_reindex_agent, agent_id)

    return ReindexResponse(
        agent_id=agent_id,
        reindexed=-1,  # -1 = pending (background)
        remaining=remaining,
        status="reindexing",
    )
