"""Recall endpoint — query an agent's memory."""

from fastapi import APIRouter

from ..llm.client import synthesize_recall
from ..models import RecallRequest, RecallResponse
from ..storage.sqlite import search_memories

router = APIRouter()


@router.post("/v1/memory/recall", response_model=RecallResponse)
async def recall(req: RecallRequest) -> RecallResponse:
    """Query an agent's memory for relevant knowledge.

    1. Search stored memories for relevance to the query
    2. Pass retrieved memories through the LLM for synthesis (with agent's LoRA if available)
    3. Return both raw memories and synthesized answer
    """
    memories = search_memories(req.agent_id, req.query, limit=req.limit)

    synthesis = ""
    if memories:
        memory_texts = [m.content for m in memories]
        synthesis = await synthesize_recall(
            req.query, memory_texts, max_tokens=req.max_tokens, agent_id=req.agent_id
        )

    return RecallResponse(
        agent_id=req.agent_id,
        query=req.query,
        memories=memories,
        synthesis=synthesis,
    )
