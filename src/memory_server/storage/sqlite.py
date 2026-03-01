"""SQLite-based memory storage backend.

Each agent gets its own SQLite database file at data/agents/{agent_id}/memories.db.
Isolation by design — agents can't read each other's memories.

Memories are stored with optional embedding vectors (BLOB) for semantic search.
When embeddings are present, search uses cosine similarity. Falls back to keyword
matching for unembedded memories.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import settings
from ..embeddings import cosine_similarity_batch, get_embedder
from ..models import Memory


def _agent_db_path(agent_id: str) -> Path:
    """Get the database path for an agent, creating directories as needed."""
    agent_dir = settings.data_dir / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir / "memories.db"


def _get_conn(agent_id: str) -> sqlite3.Connection:
    """Get a connection to an agent's memory database, initializing schema if needed."""
    db_path = _agent_db_path(agent_id)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT,
            tags TEXT DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
    """)

    # Migration: add embedding column if missing (Phase 2)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "embedding" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        conn.commit()

    return conn


def store_memory(
    agent_id: str,
    content: str,
    source: str | None = None,
    tags: list[str] | None = None,
    embedding: bytes | None = None,
) -> int:
    """Store a memory and return its ID."""
    conn = _get_conn(agent_id)
    try:
        cursor = conn.execute(
            "INSERT INTO memories (content, source, tags, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (
                content,
                source,
                json.dumps(tags or []),
                datetime.now(timezone.utc).isoformat(),
                embedding,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def search_memories(agent_id: str, query: str, limit: int = 10) -> list[Memory]:
    """Search memories using embedding cosine similarity.

    Falls back to keyword matching for memories that don't have embeddings yet.
    """
    conn = _get_conn(agent_id)
    try:
        embedder = get_embedder()

        # Fetch all memories (embedded and unembedded separately)
        embedded_rows = conn.execute(
            "SELECT id, content, source, tags, created_at, embedding "
            "FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        unembedded_rows = conn.execute(
            "SELECT id, content, source, tags, created_at "
            "FROM memories WHERE embedding IS NULL"
        ).fetchall()

        scored: list[tuple[sqlite3.Row, float]] = []

        # Score embedded memories by cosine similarity
        if embedded_rows:
            query_vec = embedder.embed_query(query)
            candidate_matrix = np.stack(
                [embedder.from_bytes(row["embedding"]) for row in embedded_rows]
            )
            similarities = cosine_similarity_batch(query_vec, candidate_matrix)
            for row, sim in zip(embedded_rows, similarities):
                # Map cosine similarity [-1, 1] to relevance [0, 1]
                scored.append((row, float((sim + 1) / 2)))

        # Score unembedded memories by keyword matching (graceful fallback)
        if unembedded_rows:
            keywords = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
            if keywords:
                for row in unembedded_rows:
                    content_lower = row["content"].lower()
                    hits = sum(1 for kw in keywords if kw in content_lower)
                    if hits > 0:
                        # Keyword relevance capped at 0.7 so embeddings always rank higher
                        relevance = (hits / len(keywords)) * 0.7
                        scored.append((row, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [_row_to_memory(row, rel) for row, rel in scored[:limit]]
    finally:
        conn.close()


def get_all_memories(agent_id: str, limit: int = 100, offset: int = 0) -> list[Memory]:
    """Get all memories for an agent, most recent first."""
    conn = _get_conn(agent_id)
    try:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_memory(row) for row in rows]
    finally:
        conn.close()


def get_memory_count(agent_id: str) -> int:
    """Get total number of memories for an agent."""
    conn = _get_conn(agent_id)
    try:
        row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_unembedded_count(agent_id: str) -> int:
    """Get number of memories without embeddings."""
    conn = _get_conn(agent_id)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE embedding IS NULL"
        ).fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()


def get_unembedded_memories(agent_id: str, limit: int = 100) -> list[tuple[int, str]]:
    """Get (id, content) pairs for memories that need embedding."""
    conn = _get_conn(agent_id)
    try:
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE embedding IS NULL LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["id"], row["content"]) for row in rows]
    finally:
        conn.close()


def batch_update_embeddings(agent_id: str, updates: list[tuple[int, bytes]]) -> int:
    """Batch update embeddings for existing memories. Returns count updated."""
    conn = _get_conn(agent_id)
    try:
        conn.executemany(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            [(emb, mid) for mid, emb in updates],
        )
        conn.commit()
        return len(updates)
    finally:
        conn.close()


def get_all_tags(agent_id: str) -> list[str]:
    """Get all unique tags used by an agent."""
    conn = _get_conn(agent_id)
    try:
        rows = conn.execute("SELECT DISTINCT tags FROM memories").fetchall()
        all_tags: set[str] = set()
        for row in rows:
            try:
                tags = json.loads(row["tags"])
                all_tags.update(tags)
            except (json.JSONDecodeError, TypeError):
                pass
        return sorted(all_tags)
    finally:
        conn.close()


def get_last_stored(agent_id: str) -> datetime | None:
    """Get timestamp of most recent memory."""
    conn = _get_conn(agent_id)
    try:
        row = conn.execute(
            "SELECT created_at FROM memories ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return datetime.fromisoformat(row["created_at"])
        return None
    finally:
        conn.close()


def _row_to_memory(row: sqlite3.Row, relevance: float = 0.0) -> Memory:
    """Convert a database row to a Memory model."""
    try:
        tags = json.loads(row["tags"]) if row["tags"] else []
    except (json.JSONDecodeError, TypeError):
        tags = []

    return Memory(
        id=row["id"],
        content=row["content"],
        source=row["source"],
        tags=tags,
        created_at=datetime.fromisoformat(row["created_at"]),
        relevance=relevance,
    )
