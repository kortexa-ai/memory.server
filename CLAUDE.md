# Memory Server

Neural long-term memory for AI agents. Uses mlx-lm server (Qwen 3.5 35B A3B) for synthesis and knowledge extraction, stores knowledge in per-agent SQLite databases with embedding-based semantic search, supports per-agent LoRA adapters for weight-space memory.

**macOS (Apple Silicon) only** — uses MLX for both inference and training.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Memory Server (this project)         │
│              FastAPI on port 2090                 │
│                                                   │
│  Embedding model (bge-small-en-v1.5) in-process  │
│  Launches mlx-lm server on port 2091 (child)     │
│  Per-agent LoRA adapters loaded per-request       │
│                                                   │
│  /v1/memory/recall   — semantic search + synthesize│
│  /v1/memory/store    — save knowledge + embed      │
│  /v1/memory/ingest   — process session transcript  │
│  /v1/memory/reindex  — backfill embeddings         │
│  /v1/memory/train    — trigger LoRA training       │
│  /v1/memory/status   — check memory state          │
│  /health             — liveness + engine status     │
└───────────────────┬──────────────────────────────┘
                    │ HTTP (localhost)
┌───────────────────▼──────────────────────────────┐
│          mlx-lm server (port 2091, child proc)   │
│  NexVeridian/Qwen3.5-35B-A3B-4bit (~19.5GB)     │
│  Per-request LoRA via "adapters" field            │
│  No restart needed after training                 │
└──────────────────────────────────────────────────┘
          ↑ HTTP from agents on local network
     ┌────┼────┐
     │    │    │
  Agent A  B  C  (any agent that speaks HTTP)
```

The memory server owns the mlx-lm server process (launched as a subprocess on startup). mlx-lm handles model loading, Metal GPU offload, and per-request LoRA adapter loading. This gives us 3x faster inference (~74 tok/s vs ~24 tok/s) compared to llama-server, and native safetensors adapter format (no GGUF conversion needed).

## Quick Start

### Install

```bash
cd ~/src/memory.server
uv sync
# First run downloads embedding model (~33MB) and MLX model (~19.5GB)
```

### Start memory server

```bash
uv run memory-server
# → http://localhost:2090
# → Swagger docs at http://localhost:2090/docs
# → Launches mlx-lm server on port 2091 automatically
```

The memory server launches the mlx-lm server as a subprocess on startup. First run downloads the model from HuggingFace (~19.5GB, cached in `~/.cache/huggingface/`). Override the port with `MEMORY_MODEL_SERVER_PORT=2091`.

### Running as a service

Example launchd unit is in `launchd/`. Contains hardcoded paths — edit to match your setup before installing.

## Project Structure

```
src/memory_server/
├── main.py           # FastAPI app, model loading on startup, uvicorn entry
├── config.py         # Settings (pydantic-settings, env vars)
├── models.py         # Pydantic request/response models
├── embeddings.py     # Singleton embedding model (fastembed/bge-small-en-v1.5)
├── api/
│   ├── recall.py     # POST /v1/memory/recall — query + synthesize
│   ├── store.py      # POST /v1/memory/store — save knowledge (with embedding)
│   ├── ingest.py     # POST /v1/memory/ingest — extract from transcript (with embedding)
│   ├── status.py     # GET  /v1/memory/status, POST /v1/memory/reindex
│   └── train.py      # POST /v1/memory/train, GET .../status
├── llm/
│   ├── engine.py     # Launches mlx-lm server, httpx client, per-request adapter loading
│   └── client.py     # High-level: synthesis prompts, knowledge extraction prompts
├── storage/
│   └── sqlite.py     # Per-agent SQLite databases (with embedding BLOBs)
└── training/
    ├── data_prep.py  # Memories → Q&A training pairs
    ├── train.py      # CLI entry point, training pipeline
    └── train_mlx.py  # MLX LoRA training (macOS)

data/agents/{agent_id}/
├── memories.db             # SQLite database (content + embedding BLOBs)
├── adapter/            # LoRA adapter (adapters.safetensors + adapter_config.json)
├── training_data.jsonl     # Q&A pairs for training
└── training_status.json    # Training job status
```

## API

### Recall (query memory)

```bash
curl -X POST http://localhost:2090/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "avery",
    "query": "What do I know about Meta financials?",
    "limit": 10,
    "max_tokens": 4000
  }'
```

1. Searches agent's SQLite DB for relevant memories
2. Sends memories + query to Qwen for synthesis (with agent's LoRA adapter if trained)
3. Returns raw memories + synthesized answer

### Store (save knowledge)

```bash
curl -X POST http://localhost:2090/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "avery",
    "content": "Meta Q4 2025 revenue was $43B, up 22% YoY",
    "source": "research-session-2026-02-27",
    "tags": ["meta", "financials"]
  }'
```

### Ingest (process transcript)

```bash
curl -X POST http://localhost:2090/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "avery",
    "transcript": "...full session transcript...",
    "session_id": "session-abc123",
    "extract_knowledge": true
  }'
```

Feeds transcript to Qwen → extracts discrete facts → stores each as a memory. Runs in the background.

### Status

```bash
curl http://localhost:2090/v1/memory/status/avery
```

## Configuration

All settings via environment variables (prefix `MEMORY_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HOST` | `0.0.0.0` | Bind address |
| `MEMORY_PORT` | `2090` | Server port |
| `MEMORY_MODEL_REPO` | `NexVeridian/Qwen3.5-35B-A3B-4bit` | MLX model for inference + training |
| `MEMORY_MODEL_SERVER_PORT` | `2091` | mlx-lm server port |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (fastembed) |
| `MEMORY_DATA_DIR` | `data/agents` | Where agent DBs and adapters live |
| `MEMORY_TRAINING_MIN_MEMORIES` | `100` | Min memories before LoRA training |

## Agent-Neutral Design

The `agent_id` is an opaque string namespace. Any agent that can make HTTP calls can use it. The memory server has no knowledge of any specific agent system.

## LoRA Adapter Flow

Per-agent LoRA adapters live at `data/agents/{agent_id}/adapter/adapters.safetensors`.

mlx-lm loads adapters per-request via the `"adapters"` field in chat completion requests. No server restart needed — after training, the new adapter is picked up on the next request.

### Training LoRA adapters

Trigger via `POST /v1/memory/train/{agent_id}`. The training pipeline runs as a subprocess:

1. Collect accumulated memories for the agent from SQLite
2. Generate Q&A training pairs using the LLM (batched, concurrent)
3. Fine-tune a LoRA adapter using mlx-lm
4. Output: `data/agents/{agent_id}/adapter/adapters.safetensors` — used directly, no conversion

## Development Phases

### Phase 1: Smart RAG (done)

- [x] SQLite storage with keyword-based search
- [x] Qwen for synthesis and knowledge extraction
- [x] REST API (store, recall, ingest, status)
- [x] Per-agent isolation
- [ ] Add MCP server wrapper for Claude SDK agents
- [ ] Add post-session ingest hook for agent orchestrators
- [ ] Service setup (launchd)
- [ ] Tests

### Phase 2: Embedding-Based Retrieval (done — tested end-to-end)

- [x] fastembed (ONNX, BAAI/bge-small-en-v1.5, 384-dim) — no PyTorch dependency
- [x] Embeddings stored as BLOBs in SQLite, cosine similarity via numpy
- [x] Auto-migration: adds `embedding` column to existing DBs
- [x] Embed on store and ingest, keyword fallback for unembedded memories
- [x] Reindex endpoint: `POST /v1/memory/reindex/{agent_id}`
- [x] Verified with real data (store + semantic recall + ingest)

### Phase 3: LoRA Memory (done)

- [x] Per-request LoRA adapter loading via mlx-lm's `"adapters"` field
- [x] Adapter path check per agent (`data/agents/{id}/adapter/adapters.safetensors`)
- [x] Training pipeline: data_prep → train → safetensors (no conversion)
- [x] mlx-lm for both inference and training (macOS only)
- [x] Training runs as subprocess — doesn't load training libs into server
- [x] API: `POST /v1/memory/train/{agent_id}`, `GET .../status`
- [ ] Test with real LoRA adapter
- [ ] Automatic "nap" cycle trigger (after N ingests)

### Phase 4: Live Self-Modification (two paths)

See `docs/live-self-modification.md` for full design.

**Our LoRA is 98% MoE experts (252M params, 480MB) and only 1.9% attention+shared (4.8M params, 9MB).** Strategy: split adapter, only modify attention live, keep MoE static.

**Path 1 — Live Conversation Adapter (fast path, days to experiment):**
- Model updates its own attention LoRA mid-conversation via `reshape_thinking` tool
- Micro-training: 20 iterations on attention-only LoRA (~15-30s)
- Async option: train in background, adapter catches up after 1-2 turns
- Steps: split adapter → reshape endpoint → async conversation reshape → measure → MCP tool

**Path 2 — Memory Corpus Hypernetwork (long path, months):**
- Train a Perceiver+Transformer hypernetwork (~200M params) on 400K+ memories
- Generates attention LoRA in single forward pass (<1s)
- Architecture: text → frozen base model activations → Perceiver → M2P Transformer → LoRA weights
- Training: KL distillation (teacher=model+context vs student=model+generated-LoRA)
- Based on SHINE, Doc-to-LoRA (MIT), Text-to-LoRA research
- Steps: data pipeline → architecture in MLX → train → swap into same endpoint

## Platform Support

| Platform | GPU Backend | Status |
|----------|------------|--------|
| macOS (M-series) | Metal via MLX | Supported |

## Key Files

| File | Purpose |
|------|---------|
| `src/memory_server/llm/engine.py` | Core: launches mlx-lm server, per-request adapter loading |
| `src/memory_server/llm/client.py` | Prompts for synthesis and knowledge extraction |
| `src/memory_server/embeddings.py` | Embedding model singleton (fastembed) |
| `src/memory_server/storage/sqlite.py` | Per-agent SQLite storage with embedding BLOBs |
| `src/memory_server/config.py` | All settings (pydantic-settings, env vars) |
| `src/memory_server/api/recall.py` | The main endpoint — search + synthesize |
| `src/memory_server/api/ingest.py` | Background transcript processing |
| `src/memory_server/api/train.py` | Training trigger + status endpoints |
| `src/memory_server/training/train.py` | Training pipeline entry point |
