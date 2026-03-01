"""Ingest endpoint — process a full session transcript into discrete memories."""

from fastapi import APIRouter, BackgroundTasks

from ..config import settings
from ..embeddings import get_embedder
from ..llm.client import extract_knowledge
from ..models import IngestRequest, IngestResponse
from ..storage.sqlite import store_memory

router = APIRouter()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def _ingest_transcript(agent_id: str, transcript: str, session_id: str | None) -> int:
    """Extract knowledge from transcript and store as memories with embeddings."""
    embedder = get_embedder()
    chunks = _chunk_text(transcript, settings.chunk_size, settings.chunk_overlap)
    total_stored = 0

    for i, chunk in enumerate(chunks):
        facts = await extract_knowledge(chunk, chunk_index=i)
        if not facts:
            continue

        # Batch-embed all facts from this chunk
        vecs = embedder.embed_texts(facts)
        for fact, vec in zip(facts, vecs):
            store_memory(
                agent_id=agent_id,
                content=fact,
                source=session_id or "session-ingest",
                tags=["session-ingest"],
                embedding=embedder.to_bytes(vec),
            )
            total_stored += 1

    return total_stored


@router.post("/v1/memory/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> IngestResponse:
    """Ingest a full session transcript into an agent's memory.

    If extract_knowledge is True (default), uses the LLM to extract discrete
    facts from the transcript before storing. Otherwise stores the raw transcript
    as a single memory.

    Knowledge extraction runs in the background — the response returns immediately.
    """
    if not req.extract_knowledge:
        # Store raw transcript as a single memory (with embedding)
        embedder = get_embedder()
        vec = embedder.embed_texts([req.transcript])[0]
        store_memory(
            agent_id=req.agent_id,
            content=req.transcript,
            source=req.session_id,
            tags=["raw-transcript"],
            embedding=embedder.to_bytes(vec),
        )
        return IngestResponse(
            agent_id=req.agent_id,
            session_id=req.session_id,
            memories_created=1,
        )

    # Run extraction in background
    background_tasks.add_task(_ingest_transcript, req.agent_id, req.transcript, req.session_id)

    return IngestResponse(
        agent_id=req.agent_id,
        session_id=req.session_id,
        memories_created=-1,  # -1 = pending (background extraction)
        status="ingesting",
    )
