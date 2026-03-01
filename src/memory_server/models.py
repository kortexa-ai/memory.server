"""Pydantic models for the memory server API."""

from datetime import datetime

from pydantic import BaseModel, Field


# --- Request models ---


class RecallRequest(BaseModel):
    """Query an agent's memory for relevant knowledge."""

    agent_id: str = Field(..., description="Agent identifier (e.g. 'avery', 'openclaw')")
    query: str = Field(..., description="Natural language query")
    limit: int = Field(10, ge=1, le=50, description="Max memories to return")
    max_tokens: int = Field(4000, ge=100, le=16000, description="Max tokens in response")


class StoreRequest(BaseModel):
    """Store a piece of knowledge in an agent's memory."""

    agent_id: str
    content: str = Field(..., description="The knowledge to store")
    source: str | None = Field(None, description="Where this came from (session ID, URL, etc.)")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")


class IngestRequest(BaseModel):
    """Ingest a full session transcript into an agent's memory."""

    agent_id: str
    transcript: str = Field(..., description="Full session transcript")
    session_id: str | None = Field(None, description="Session identifier")
    extract_knowledge: bool = Field(
        True, description="Use LLM to extract key facts before storing"
    )


# --- Response models ---


class Memory(BaseModel):
    """A single memory entry."""

    id: int
    content: str
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    relevance: float = Field(0.0, description="Relevance score (0-1) from recall query")


class RecallResponse(BaseModel):
    """Response from a recall query."""

    agent_id: str
    query: str
    memories: list[Memory]
    synthesis: str = Field("", description="LLM-synthesized answer from memories")


class StoreResponse(BaseModel):
    """Response from storing a memory."""

    agent_id: str
    memory_id: int
    stored: bool = True


class IngestResponse(BaseModel):
    """Response from ingesting a transcript."""

    agent_id: str
    session_id: str | None
    memories_created: int
    status: str = "ok"


class AgentStatus(BaseModel):
    """Status of an agent's memory store."""

    agent_id: str
    memory_count: int
    unembedded_count: int = 0
    has_adapter: bool = False
    last_stored: datetime | None = None
    tags: list[str] = Field(default_factory=list, description="All tags used by this agent")


class ReindexResponse(BaseModel):
    """Response from reindexing memories with embeddings."""

    agent_id: str
    reindexed: int
    remaining: int
    status: str = "ok"


# --- Training models (Phase 3) ---


class TrainRequest(BaseModel):
    """Request to train a LoRA adapter for an agent."""

    min_memories: int = Field(100, ge=10, description="Minimum memories required to start training")
    max_iters: int = Field(200, ge=10, le=2000, description="Maximum training iterations")
    lora_rank: int = Field(8, ge=2, le=64, description="LoRA adapter rank")


class TrainResponse(BaseModel):
    """Response from starting a training job."""

    agent_id: str
    status: str  # "training_started" | "insufficient_memories" | "already_training"
    memory_count: int
    training_pairs: int | None = None


class TrainStatus(BaseModel):
    """Status of a training job."""

    agent_id: str
    status: str  # "idle" | "preparing" | "training" | "converting" | "complete" | "failed"
    progress: float | None = None
    adapter_exists: bool
    adapter_updated: datetime | None = None
