"""Data preparation for LoRA training.

Reads an agent's memories from SQLite and generates Q&A training pairs
by calling the memory server's API (or a local LLM instance).

Memories are batched (20 per LLM call) and LLM calls run in parallel
(default 4 concurrent) for speed.

Output: JSONL file at data/agents/{agent_id}/training_data.jsonl
Format: {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
"""

import asyncio
import json
import logging
from pathlib import Path

import httpx

from ..config import settings
from ..storage.sqlite import get_all_memories

logger = logging.getLogger("memory-server.training.data_prep")

# How many memories to bundle into a single LLM call
BATCH_SIZE = 20

# How many LLM calls to run concurrently
CONCURRENCY = 4

QA_GENERATION_PROMPT = """\
You are generating training data for a personalized AI memory system.
Given a set of stored knowledge items, generate question-answer pairs
that test recall of the information.

Rules:
- Generate 2-3 Q&A pairs per knowledge item
- Questions should be natural and varied (who, what, when, how, why, etc.)
- Answers should be complete but concise
- Each pair should be self-contained (understandable without the other items)
- Include the key facts from the knowledge in the answers
- Skip items that are too vague or trivial to generate good questions about

Return pairs in this exact format, one per line:
Q: <question>
A: <answer>

Knowledge items:

{content}"""


async def _process_batch(
    client: httpx.AsyncClient,
    server_url: str,
    batch_idx: int,
    batch: list,
    semaphore: asyncio.Semaphore,
) -> list[tuple[str, str]]:
    """Process one batch of memories, returning Q&A pairs."""
    content = "\n".join(
        f"[{i+1}] {m.content}" for i, m in enumerate(batch)
    )
    prompt = QA_GENERATION_PROMPT.format(content=content)

    async with semaphore:
        try:
            resp = await client.post(
                f"{server_url}/v1/internal/generate",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a training data generator."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["content"]
        except Exception as e:
            logger.warning(f"Failed to generate Q&A for batch {batch_idx+1}: {e}")
            return []

    return _parse_qa_pairs(raw)


async def prepare_training_data(
    agent_id: str,
    server_url: str = "http://localhost:2090",
    qa_pairs_per_memory: int | None = None,
) -> Path:
    """Generate Q&A training pairs from an agent's memories.

    Uses the memory server's own LLM (via HTTP) to generate the Q&A pairs.
    Memories are batched and processed with concurrent LLM calls.

    Returns path to the JSONL training data file.
    """
    output_dir = settings.data_dir / agent_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.jsonl"

    # Get all memories for this agent
    memories = get_all_memories(agent_id, limit=settings.max_memories_per_agent)
    if not memories:
        raise ValueError(f"No memories found for agent '{agent_id}'")

    total = len(memories)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        f"Generating Q&A pairs from {total} memories "
        f"({num_batches} batches, {CONCURRENCY} concurrent) for agent '{agent_id}'"
    )

    # Create batches
    batches = []
    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batches.append(memories[start:end])

    # Process all batches with bounded concurrency
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_pairs: list[dict] = []
    completed = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            _process_batch(client, server_url, i, batch, semaphore)
            for i, batch in enumerate(batches)
        ]

        for coro in asyncio.as_completed(tasks):
            pairs = await coro
            for q, a in pairs:
                all_pairs.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                })
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                logger.info(f"  {completed}/{num_batches} batches done, {len(all_pairs)} pairs so far")

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
