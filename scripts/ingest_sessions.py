"""Bulk-ingest session transcripts into the memory server.

Auto-discovers Claude Code and Codex sessions from data/sessions/,
extracts conversation text, uses the LLM to extract discrete facts,
and stores each fact via /v1/memory/store.

Supports resume — tracks completed sessions in a progress file.
Ctrl+C safe — a session is only marked complete after all its facts are stored.

Usage:
    uv run python scripts/ingest_sessions.py [--agent-id NAME] [--server URL]
    uv run python scripts/ingest_sessions.py --force   # re-ingest everything
    uv run python scripts/ingest_sessions.py --min-size 20000  # skip files < 20KB
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"

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


# --- Session discovery ---

def discover_sessions(min_size: int) -> list[dict]:
    """Walk data/sessions/ and return all session files with metadata.

    Returns list of dicts: {path, source, name, size}
    """
    sessions = []

    # Claude Code sessions: data/sessions/claude/{project_dir}/{uuid}.jsonl
    claude_dir = SESSIONS_DIR / "claude"
    if claude_dir.exists():
        for jsonl_path in claude_dir.rglob("*.jsonl"):
            size = jsonl_path.stat().st_size
            if size < min_size:
                continue
            name = _claude_session_name(jsonl_path)
            sessions.append({
                "path": str(jsonl_path),
                "source": "claude",
                "name": name,
                "size": size,
            })

    # Codex sessions: JSONL files in data/sessions/codex/{year}/{month}/{day}/
    codex_dir = SESSIONS_DIR / "codex"
    if codex_dir.exists():
        for jsonl_path in codex_dir.rglob("*.jsonl"):
            size = jsonl_path.stat().st_size
            if size < min_size:
                continue
            name = _codex_session_name(jsonl_path)
            sessions.append({
                "path": str(jsonl_path),
                "source": "codex",
                "name": name,
                "size": size,
            })
        # Older Codex JSON format (top-level .json files)
        for json_path in codex_dir.glob("*.json"):
            size = json_path.stat().st_size
            if size < min_size:
                continue
            name = f"codex/{json_path.stem}"
            sessions.append({
                "path": str(json_path),
                "source": "codex-json",
                "name": name,
                "size": size,
            })

    # Sort by path for deterministic order
    sessions.sort(key=lambda s: s["path"])
    return sessions


def _claude_session_name(path: Path) -> str:
    """Derive a human-readable name from a Claude session path.

    data/sessions/claude/-Users-francip-src-api-server/abc.jsonl → claude/api-server/abc[:8]
    .../subagents/agent-abc.jsonl → claude/api-server/sub-abc[:8]
    """
    rel = path.relative_to(SESSIONS_DIR / "claude")
    parts = rel.parts

    # Project dir is the first component (e.g. -Users-francip-src-api-server)
    project_dir = parts[0]
    # Strip common prefixes to get clean project name
    project = project_dir
    for prefix in ("-Users-francip-src-", "-Users-francip--claude-worktrees-", "-Users-francip-"):
        if project.startswith(prefix):
            project = project[len(prefix):]
            break

    # Include short session ID for uniqueness
    short_id = path.stem[:8]

    if "subagents" in parts:
        agent_file = path.stem  # agent-a7f787a40365c32d0
        short_id = agent_file.replace("agent-", "")[:8]
        return f"claude/{project}/sub-{short_id}"
    else:
        return f"claude/{project}/{short_id}"


def _codex_session_name(path: Path) -> str:
    """Derive name from a Codex session path.

    data/sessions/codex/2025/11/03/rollout-2025-11-03T19-51-37-uuid.jsonl → codex/2025-11-03/uuid[:8]
    """
    stem = path.stem  # rollout-2025-11-03T19-51-37-019a4cfd-e013-70b2-...
    # Try to extract the date and a short ID
    parts = stem.split("-")
    # Find the date portion (YYYY-MM-DD or from T-separated timestamp)
    if stem.startswith("rollout-") and len(parts) >= 4:
        # rollout-YYYY-MM-DDT... or rollout-YYYY-MM-DD-uuid
        date_part = f"{parts[1]}-{parts[2]}-{parts[3][:2]}"
        # Use last 8 chars of the full stem as short ID
        short_id = stem[-8:]
        return f"codex/{date_part}/{short_id}"
    return f"codex/{stem[:30]}"


# --- Transcript extraction ---

def extract_transcript_claude(session_path: str) -> str:
    """Extract user/assistant conversation text from a Claude Code JSONL session."""
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


def extract_transcript_codex_jsonl(session_path: str) -> str:
    """Extract user/assistant text from a Codex JSONL session.

    Handles two formats:
    1. Wrapped: {"type": "response_item", "payload": {"type": "message", ...}}
    2. Top-level: {"type": "message", "role": "user"|"assistant", "content": [...]}
    """
    lines = []
    with open(session_path) as f:
        for raw_line in f:
            obj = json.loads(raw_line)

            # Format 1: wrapped in response_item
            if obj.get("type") == "response_item":
                payload = obj.get("payload", {})
                if payload.get("type") != "message":
                    continue
                role = payload.get("role", "")
                content = payload.get("content", [])
            # Format 2: top-level message
            elif obj.get("type") == "message":
                role = obj.get("role", "")
                content = obj.get("content", [])
            else:
                continue

            if role not in ("user", "assistant"):
                continue
            text = _extract_codex_content(content)
            if text:
                prefix = "User" if role == "user" else "Assistant"
                lines.append(f"{prefix}: {text}")
    return "\n\n".join(lines)


def extract_transcript_codex_json(session_path: str) -> str:
    """Extract user/assistant text from an older Codex JSON session.

    Format: {"session": {...}, "items": [{"type": "message", "role": "user"|"assistant",
             "content": [{"type": "input_text"|"output_text", "text": "..."}]}]}
    """
    with open(session_path) as f:
        data = json.load(f)

    lines = []
    for item in data.get("items", []):
        if item.get("type") != "message":
            continue
        role = item.get("role", "")
        if role not in ("user", "assistant"):
            continue
        text = _extract_codex_content(item.get("content", []))
        if text:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
    return "\n\n".join(lines)


def _extract_codex_content(content_blocks: list) -> str:
    """Extract text from Codex content blocks (input_text, output_text, text)."""
    parts = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype in ("input_text", "output_text", "text"):
            t = block.get("text", "").strip()
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def extract_transcript(session: dict) -> str:
    """Extract transcript from a session, dispatching to the right extractor."""
    source = session["source"]
    path = session["path"]
    if source == "claude":
        return extract_transcript_claude(path)
    elif source == "codex":
        return extract_transcript_codex_jsonl(path)
    elif source == "codex-json":
        return extract_transcript_codex_json(path)
    return ""


# --- Chunking ---

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


# --- mlx-lm server management ---

def restart_mlx_server() -> bool:
    """Kill and restart the mlx-lm server. Returns True if it comes back healthy."""
    print("  [restarting mlx-lm server...]", file=sys.stderr)

    # Kill any existing mlx-lm processes
    result = subprocess.run(["pkill", "-f", "mlx_lm.server"], capture_output=True)

    time.sleep(3)

    # Start new mlx-lm server
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm.server",
         "--model", "NexVeridian/Qwen3.5-35B-A3B-4bit",
         "--host", "0.0.0.0", "--port", "2091"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to become healthy (model is cached, should be ~30s)
    for attempt in range(60):
        try:
            resp = httpx.get("http://localhost:2091/v1/models", timeout=5.0)
            if resp.status_code == 200:
                print("  [mlx-lm server restarted successfully]", file=sys.stderr)
                return True
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(5)

    print("  [mlx-lm server failed to restart within 5 minutes]", file=sys.stderr)
    return False


# --- LLM extraction + storage ---

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


# --- Progress tracking ---

def progress_path(agent_id: str) -> Path:
    return Path("data") / "agents" / agent_id / "ingest_progress.json"


def load_progress(agent_id: str) -> set[str]:
    """Load set of completed session paths."""
    p = progress_path(agent_id)
    if not p.exists():
        return set()
    with open(p) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def save_progress(agent_id: str, completed: set[str]):
    """Save completed session paths to progress file."""
    p = progress_path(agent_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


# --- ETA ---

class ETA:
    """Running average ETA calculator."""

    def __init__(self):
        self.chunk_times: list[float] = []

    def record(self, seconds: float):
        self.chunk_times.append(seconds)

    def estimate(self, remaining_chunks: int) -> str:
        if not self.chunk_times:
            return "?"
        # Use last 50 chunks for rolling average
        recent = self.chunk_times[-50:]
        avg = sum(recent) / len(recent)
        remaining_secs = avg * remaining_chunks
        if remaining_secs > 3600:
            return f"{remaining_secs / 3600:.1f}h"
        elif remaining_secs > 60:
            return f"{remaining_secs / 60:.0f}m"
        return f"{remaining_secs:.0f}s"


# --- Main processing ---

def process_session(client: httpx.Client, agent_id: str, session: dict, eta: ETA,
                    total_chunks_remaining: int) -> int:
    """Process one session: chunk → extract → store. Returns total memories stored."""
    transcript = extract_transcript(session)
    if not transcript:
        print(f"  SKIP (no conversation text)")
        session["_no_text"] = True
        return 0

    chunks = chunk_text(transcript)
    print(f"  {len(transcript):,} chars → {len(chunks)} chunks  [ETA {eta.estimate(total_chunks_remaining)}]")

    total = 0
    consecutive_chunk_errors = 0
    source_tag = session["name"]
    source_label = f"session-{session['name']}"

    for i, chunk in enumerate(chunks):
        t0 = time.time()
        try:
            facts = extract_facts(client, chunk)
            consecutive_chunk_errors = 0
        except Exception as e:
            print(f"    chunk {i+1}/{len(chunks)}: EXTRACT ERROR — {e}", file=sys.stderr)
            elapsed = time.time() - t0
            eta.record(elapsed)
            consecutive_chunk_errors += 1
            if consecutive_chunk_errors >= 5:
                print(f"    5 consecutive chunk errors — aborting session", file=sys.stderr)
                break
            continue

        elapsed = time.time() - t0
        eta.record(elapsed)

        for fact in facts:
            try:
                store_fact(client, agent_id, fact, source_label, ["session-ingest", source_tag])
            except Exception as e:
                print(f"    store error: {e}", file=sys.stderr)
                continue
            total += 1

        print(f"    chunk {i+1}/{len(chunks)}: {len(facts)} facts ({elapsed:.1f}s)  [ETA {eta.estimate(total_chunks_remaining - i - 1)}]")

    return total


def estimate_chunks(session: dict) -> int:
    """Rough estimate of chunk count from file size.

    Actual text ratio varies: ~10% for small files, ~2-4% for large ones
    (most JSONL content is tool calls, not conversation text).
    """
    size = session["size"]
    if size < 50_000:
        ratio = 0.10
    elif size < 200_000:
        ratio = 0.06
    else:
        ratio = 0.03
    return max(1, int(size * ratio / CHUNK_SIZE))


def main():
    parser = argparse.ArgumentParser(description="Bulk-ingest sessions into memory server")
    parser.add_argument("--agent-id", default="francip", help="Agent ID (default: francip)")
    parser.add_argument("--server", default="http://localhost:2090", help="Memory server URL")
    parser.add_argument("--min-size", type=int, default=10240, help="Skip files smaller than N bytes (default: 10KB)")
    parser.add_argument("--force", action="store_true", help="Re-ingest already-completed sessions")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.server)

    # Verify server is up
    try:
        resp = client.get("/health")
        resp.raise_for_status()
        health = resp.json()
        if not health.get("engine_loaded"):
            print("WARNING: LLM server not connected — extraction will fail", file=sys.stderr)
        print(f"Memory server OK at {args.server}")
    except Exception as e:
        print(f"Memory server not reachable at {args.server}: {e}", file=sys.stderr)
        sys.exit(1)

    # Discover sessions
    sessions = discover_sessions(args.min_size)
    print(f"Discovered {len(sessions)} sessions (>= {args.min_size // 1024}KB)")

    # Count by source
    by_source = {}
    for s in sessions:
        by_source.setdefault(s["source"], 0)
        by_source[s["source"]] += 1
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")

    # Filter already-completed
    completed = load_progress(args.agent_id) if not args.force else set()
    remaining = [s for s in sessions if s["path"] not in completed]
    if len(remaining) < len(sessions):
        print(f"Already ingested: {len(sessions) - len(remaining)} sessions")
    print(f"To process: {len(remaining)} sessions")

    if not remaining:
        print("Nothing to do.")
        return

    # Estimate total chunks for ETA
    total_estimated_chunks = sum(estimate_chunks(s) for s in remaining)
    print(f"Estimated total chunks: ~{total_estimated_chunks}")

    eta = ETA()
    t0 = time.time()
    grand_total = 0
    chunks_done = 0

    consecutive_failures = 0
    restarts = 0
    max_restarts = 5
    for idx, session in enumerate(remaining):
        est_chunks = estimate_chunks(session)
        remaining_chunks = total_estimated_chunks - chunks_done

        print(f"\n[{idx+1}/{len(remaining)}] {session['name']} ({session['size'] // 1024}KB)")

        total = process_session(client, args.agent_id, session, eta, remaining_chunks)

        # Only mark complete if we actually extracted something (or had no text)
        # Don't mark complete on all-errors — we want to retry next run
        if total > 0 or session.get("_no_text"):
            completed.add(session["path"])
            save_progress(args.agent_id, completed)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                if restarts >= max_restarts:
                    print(f"\nmlx-lm crashed {max_restarts} times — giving up.", file=sys.stderr)
                    break
                print(f"\n3 consecutive failures — restarting mlx-lm (attempt {restarts + 1}/{max_restarts})...")
                if restart_mlx_server():
                    consecutive_failures = 0
                    restarts += 1
                    # Retry the sessions that failed (rewind idx isn't easy, so just continue —
                    # they'll be retried on the next script run since they weren't marked complete)
                else:
                    print(f"\nFailed to restart mlx-lm. Stopping.", file=sys.stderr)
                    break

        chunks_done += est_chunks
        grand_total += total
        print(f"  → {total} memories (running total: {grand_total})")

    elapsed = time.time() - t0
    hours = elapsed / 3600
    if hours >= 1:
        print(f"\nDone in {hours:.1f} hours")
    else:
        print(f"\nDone in {elapsed / 60:.1f} minutes")

    print(f"Total new memories: {grand_total}")

    # Show final status
    try:
        resp = client.get(f"/v1/memory/status/{args.agent_id}")
        if resp.status_code == 200:
            status = resp.json()
            print(f"Agent '{args.agent_id}': {status.get('memory_count', '?')} total memories")
    except Exception:
        pass


if __name__ == "__main__":
    main()
