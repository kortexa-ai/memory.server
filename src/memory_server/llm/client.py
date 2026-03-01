"""LLM client for the memory server.

Uses llama-server (Qwen 3.5 35B A3B) via the engine HTTP client for:
1. Recall synthesis — given retrieved memories + a query, produce a coherent answer
2. Knowledge extraction — given a transcript, extract key facts as discrete memories
"""

import logging

from .engine import get_engine

logger = logging.getLogger("memory-server.llm")


async def synthesize_recall(
    query: str, memories: list[str], max_tokens: int = 2000, agent_id: str | None = None
) -> str:
    """Given a query and retrieved memories, synthesize a coherent response.

    If the agent has a LoRA adapter, the engine swaps it in before inference
    for personalized synthesis. Otherwise uses the base model.
    """
    if not memories:
        return ""

    engine = get_engine()
    memory_block = "\n\n---\n\n".join(f"[Memory {i+1}]\n{m}" for i, m in enumerate(memories))

    return await engine.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a memory recall system. Given a query and a set of stored memories, "
                    "synthesize the relevant information into a clear, concise response. "
                    "Only include information that is directly relevant to the query. "
                    "If the memories don't contain relevant information, say so briefly. "
                    "Do not fabricate information not present in the memories."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nStored memories:\n\n{memory_block}",
            },
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        agent_id=agent_id,
    )


async def extract_knowledge(transcript: str, chunk_index: int = 0) -> list[str]:
    """Extract discrete knowledge facts from a session transcript chunk.

    Returns a list of self-contained knowledge statements that can be stored
    as individual memories. Each should be understandable without context.
    """
    engine = get_engine()

    raw = await engine.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledge extraction system. Given a conversation transcript, "
                    "extract the key facts, decisions, preferences, and insights as a list of "
                    "self-contained statements. Each statement should:\n"
                    "- Be understandable without the original conversation context\n"
                    "- Contain one discrete piece of knowledge\n"
                    "- Include relevant specifics (names, numbers, dates)\n"
                    "- Skip trivial small talk and filler\n\n"
                    "Return one statement per line, no numbering or bullets."
                ),
            },
            {
                "role": "user",
                "content": f"Extract key knowledge from this transcript:\n\n{transcript}",
            },
        ],
        max_tokens=2000,
        temperature=0.2,
    )

    # Split into individual statements, filter empties
    return [line.strip() for line in raw.split("\n") if line.strip()]
