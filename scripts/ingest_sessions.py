"""Ingest Claude Code session transcripts into the memory server.

Reads JSONL session files, extracts conversation text, uses the LLM to
extract discrete facts, and stores each fact via /v1/memory/store.

Runs synchronously — one chunk at a time — so we get progress output
and don't overwhelm llama-server with concurrent requests.

Usage:
    uv run python scripts/ingest_sessions.py [--agent-id NAME] [--server URL]
"""

import argparse
import json
import os
import sys
import time

import httpx

SESSIONS = [
    ("penumbra-server", "-Users-francip-src-penumbra-server/2f359fc6-1836-49a7-9a4e-03a0bd61a111.jsonl"),
    ("shapes-chat", "-Users-francip-src-shapes-shapes-chat/fbddd956-b8cb-4904-b081-dead0214fe34.jsonl"),
    ("shapes-chat-sub", "-Users-francip-src-shapes-shapes-chat/fbddd956-b8cb-4904-b081-dead0214fe34/subagents/agent-a917989.jsonl"),
    ("shapes-app", "-Users-francip-src-shapes-app/a4e328eb-1511-4cb4-b32d-0b750045c9d3.jsonl"),
    ("src-general-1", "-Users-francip-src/b48aae1f-7098-488a-8277-3a587b652ec3.jsonl"),
    ("app-zen-villani", "-Users-francip--claude-worktrees-app-zen-villani/62430802-60b4-44cd-b674-1c569b527689.jsonl"),
    ("shardling-ai", "-Users-francip-src-shardling-ai/d5d3e259-2adf-4dec-be34-ee476bd8f679.jsonl"),
    ("api-server", "-Users-francip-src-api-server/e8106c3d-e4d0-4489-ba1d-45b9b678def6.jsonl"),
    ("control-desktop", "-Users-francip-src-control-desktop/1978a651-8d82-4c1a-867e-a7ad76793fc5.jsonl"),
    ("src-general-2", "-Users-francip-src/fd410dda-286b-42a2-9b0a-bcb36a0d4a41.jsonl"),
]

BASE_DIR = os.path.expanduser("~/.claude/projects")

# Chunk size for transcript splitting (chars)
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500

EXTRACT_PROMPT = {
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
}


def extract_transcript(session_path: str) -> str:
    """Extract user/assistant conversation text from a JSONL session file."""
    lines = []
    with open(session_path) as f:
        for raw_line in f:
            obj = json.loads(raw_line)
            msg = obj.get("message", {})
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "\n".join(parts).strip()
            else:
                continue
            if text:
                prefix = "User" if role == "user" else "Assistant"
                lines.append(f"{prefix}: {text}")
    return "\n\n".join(lines)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks at message boundaries."""
    messages = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0

    for msg in messages:
        if current_len + len(msg) + 2 > CHUNK_SIZE and current:
            chunks.append("\n\n".join(current))
            # Keep last message for overlap
            overlap_msgs = []
            overlap_len = 0
            for m in reversed(current):
                if overlap_len + len(m) > CHUNK_OVERLAP:
                    break
                overlap_msgs.insert(0, m)
                overlap_len += len(m) + 2
            current = overlap_msgs
            current_len = overlap_len
        current.append(msg)
        current_len += len(msg) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def extract_facts(client: httpx.Client, chunk: str) -> list[str]:
    """Call LLM to extract facts from a transcript chunk."""
    resp = client.post(
        "/v1/internal/generate",
        json={
            "messages": [
                EXTRACT_PROMPT,
                {"role": "user", "content": f"Extract key knowledge from this transcript:\n\n{chunk}"},
            ],
            "max_tokens": 2000,
            "temperature": 0.2,
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    raw = resp.json().get("content", "")
    return [line.strip() for line in raw.split("\n") if line.strip()]


def store_fact(client: httpx.Client, agent_id: str, fact: str, source: str, tags: list[str]):
    """Store a single fact via the memory server (embedding computed server-side)."""
    resp = client.post(
        "/v1/memory/store",
        json={
            "agent_id": agent_id,
            "content": fact,
            "source": source,
            "tags": tags,
        },
        timeout=30.0,
    )
    resp.raise_for_status()


def process_session(client: httpx.Client, agent_id: str, name: str, transcript: str) -> int:
    """Process one session: chunk → extract → store. Returns total memories stored."""
    chunks = chunk_text(transcript)
    print(f"\n  {name}: {len(transcript):,} chars → {len(chunks)} chunks")

    total = 0
    for i, chunk in enumerate(chunks):
        t0 = time.time()
        try:
            facts = extract_facts(client, chunk)
        except Exception as e:
            print(f"    chunk {i+1}/{len(chunks)}: EXTRACT ERROR — {e}", file=sys.stderr)
            continue

        elapsed = time.time() - t0
        for fact in facts:
            try:
                store_fact(client, agent_id, fact, f"claude-session-{name}", ["session-ingest", name])
            except Exception as e:
                print(f"    store error: {e}", file=sys.stderr)
                continue
            total += 1

        print(f"    chunk {i+1}/{len(chunks)}: {len(facts)} facts ({elapsed:.1f}s)")

    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest Claude sessions into memory server")
    parser.add_argument("--agent-id", default="francip", help="Agent ID (default: francip)")
    parser.add_argument("--server", default="http://localhost:2090", help="Memory server URL")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.server)

    # Verify server is up
    try:
        resp = client.get("/health")
        resp.raise_for_status()
        health = resp.json()
        if not health.get("engine_loaded"):
            print("WARNING: llama-server not connected — extraction will fail", file=sys.stderr)
        print(f"Memory server OK at {args.server}")
    except Exception as e:
        print(f"Memory server not reachable at {args.server}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Agent ID: {args.agent_id}")
    print(f"Sessions: {len(SESSIONS)}")

    t0 = time.time()
    grand_total = 0

    for name, path in SESSIONS:
        full_path = os.path.join(BASE_DIR, path)
        if not os.path.exists(full_path):
            print(f"\n  {name}: SKIP (file not found)", file=sys.stderr)
            continue

        transcript = extract_transcript(full_path)
        if not transcript:
            print(f"\n  {name}: SKIP (no conversation text)")
            continue

        total = process_session(client, args.agent_id, name, transcript)
        grand_total += total
        print(f"  → {total} memories stored (running total: {grand_total})")

    elapsed = time.time() - t0
    hours = elapsed / 3600
    minutes = elapsed / 60
    if hours >= 1:
        print(f"\nDone in {hours:.1f} hours")
    else:
        print(f"\nDone in {minutes:.1f} minutes")

    # Show final status
    resp = client.get(f"/v1/memory/status/{args.agent_id}")
    if resp.status_code == 200:
        status = resp.json()
        print(f"Agent '{args.agent_id}': {status.get('memory_count', '?')} total memories")


if __name__ == "__main__":
    main()
