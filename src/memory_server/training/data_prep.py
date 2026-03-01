"""Data preparation for LoRA training.

Reads an agent's memories from SQLite and generates Q&A training pairs
by calling the memory server's API (or a local LLM instance).

Output: JSONL file at data/agents/{agent_id}/training_data.jsonl
Format: {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
"""

import json
import logging
from pathlib import Path

import httpx

from ..config import settings
from ..storage.sqlite import get_all_memories

logger = logging.getLogger("memory-server.training.data_prep")

# Prompt for generating Q&A pairs from a memory
QA_GENERATION_PROMPT = """\
You are generating training data for a personalized AI memory system.
Given a piece of stored knowledge, generate {n} diverse question-answer pairs
where the question asks about the information and the answer provides it.

Rules:
- Questions should be natural and varied (who, what, when, how, why, etc.)
- Answers should be complete but concise
- Each pair should be self-contained
- Include the key facts from the knowledge in the answers

Return exactly {n} pairs, one per line, in this format:
Q: <question>
A: <answer>

Knowledge to generate pairs from:
{content}"""


async def prepare_training_data(
    agent_id: str,
    server_url: str = "http://localhost:2090",
    qa_pairs_per_memory: int | None = None,
) -> Path:
    """Generate Q&A training pairs from an agent's memories.

    Uses the memory server's own LLM (via HTTP) to generate the Q&A pairs.
    This keeps the training data generation separate from the training process.

    Returns path to the JSONL training data file.
    """
    if qa_pairs_per_memory is None:
        qa_pairs_per_memory = settings.training_qa_pairs_per_memory

    output_dir = settings.data_dir / agent_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.jsonl"

    # Get all memories for this agent
    memories = get_all_memories(agent_id, limit=settings.max_memories_per_agent)
    if not memories:
        raise ValueError(f"No memories found for agent '{agent_id}'")

    logger.info(f"Generating Q&A pairs from {len(memories)} memories for agent '{agent_id}'")

    all_pairs: list[dict] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, memory in enumerate(memories):
            prompt = QA_GENERATION_PROMPT.format(
                n=qa_pairs_per_memory, content=memory.content
            )

            # Use the memory server's own /v1/memory/recall-like inference
            # We call the engine directly via a simple chat endpoint
            try:
                resp = await client.post(
                    f"{server_url}/v1/internal/generate",
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a training data generator."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()["content"]
            except Exception as e:
                logger.warning(f"Failed to generate Q&A for memory {memory.id}: {e}")
                continue

            # Parse Q&A pairs from the response
            pairs = _parse_qa_pairs(raw)
            for q, a in pairs:
                all_pairs.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                })

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(memories)} memories, {len(all_pairs)} pairs so far")

    # Write JSONL
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Wrote {len(all_pairs)} training pairs to {output_path}")
    return output_path


def _parse_qa_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse Q: / A: formatted pairs from LLM output."""
    pairs: list[tuple[str, str]] = []
    current_q: str | None = None
    current_a: str | None = None

    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.startswith("Q:"):
            # If we have a complete pair, save it
            if current_q and current_a:
                pairs.append((current_q, current_a))
            current_q = line[2:].strip()
            current_a = None
        elif line.startswith("A:") and current_q:
            current_a = line[2:].strip()

    # Don't forget the last pair
    if current_q and current_a:
        pairs.append((current_q, current_a))

    return pairs
