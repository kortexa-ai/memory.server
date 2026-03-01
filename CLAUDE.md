# Memory Server

Neural long-term memory for AI agents. Connects to llama-server (Qwen 3.5 35B A3B) for synthesis and knowledge extraction, stores knowledge in per-agent SQLite databases with embedding-based semantic search, supports per-agent LoRA adapters for weight-space memory.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Memory Server (this project)         │
│              FastAPI on port 2090                 │
│                                                   │
│  Embedding model (bge-small-en-v1.5) in-process  │
│  Connects to llama-server on port 2027           │
│  Per-agent LoRA adapters via llama-server API     │
│                                                   │
│  /v1/memory/recall   — semantic search + synthesize│
│  /v1/memory/store    — save knowledge + embed      │
│  /v1/memory/ingest   — process session transcript  │
│  /v1/memory/reindex  — backfill embeddings         │
│  /v1/memory/train    — trigger LoRA training       │
│  /v1/memory/status   — check memory state          │
│  /health             — liveness + engine status     │
└───────────────────┬──────────────────────────────┘
                    │ HTTP
┌───────────────────▼──────────────────────────────┐
│              llama-server (port 2027)             │
│  Qwen 3.5 35B A3B (Metal on Mac, CUDA on GPU)   │
│  Q4_K_XL (Mac) / Q8_K_XL (Linux)                │
│  LoRA adapters via /lora-adapters endpoint        │
└──────────────────────────────────────────────────┘
          ↑ HTTP from agents on local network
     ┌────┼────┐
     │    │    │
  Agent A  B  C  (any agent that speaks HTTP)
```

The memory server connects to a llama-server instance for inference. llama-server handles model loading, GPU offload, KV cache, and LoRA adapter management. This gives us latest llama.cpp features (qwen35moe architecture) without waiting for Python bindings.

## Quick Start

### Install

```bash
cd ~/src/memory.server
uv sync
# First run downloads embedding model (~33MB, cached after that)
```

### Start llama-server

The memory server needs a running llama-server instance for LLM inference (synthesis, knowledge extraction, training data generation). It talks to it via the OpenAI-compatible HTTP API.

**macOS (Metal):**

```bash
llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 --port 2027 \
  --jinja -ngl 99 --threads -1 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0
```

**Linux (NVIDIA GPU):**

```bash
llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q8_K_XL \
  --host 0.0.0.0 --port 2027 \
  --jinja -ngl 99 --threads -1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

First run downloads the model from HuggingFace (~20GB, cached in `~/.cache/llama.cpp/`). The `qwen35moe` architecture requires llama.cpp build 8171+ (Jan 2025).

To verify: `curl http://localhost:2027/health` → `{"status":"ok"}`

### Start memory server

```bash
uv run memory-server
# → http://localhost:2090
# → Swagger docs at http://localhost:2090/docs
```

The memory server will connect to llama-server on startup. If llama-server isn't running yet, it logs a warning and keeps going — it'll work as soon as llama-server comes up. Override the URL with `MEMORY_ENGINE_SERVER_URL=http://host:port`.

### Running as a service

Example launchd (macOS) and systemd (Linux) units are in `launchd/` and `systemd/`. These contain hardcoded paths — edit them to match your setup before installing.

## Project Structure

```
src/memory_server/
├── main.py           # FastAPI app, model loading on startup, uvicorn entry
├── config.py         # Settings (pydantic-settings, env vars, platform detection)
├── models.py         # Pydantic request/response models
├── embeddings.py     # Singleton embedding model (fastembed/bge-small-en-v1.5)
├── api/
│   ├── recall.py     # POST /v1/memory/recall — query + synthesize
│   ├── store.py      # POST /v1/memory/store — save knowledge (with embedding)
│   ├── ingest.py     # POST /v1/memory/ingest — extract from transcript (with embedding)
│   ├── status.py     # GET  /v1/memory/status, POST /v1/memory/reindex
│   └── train.py      # POST /v1/memory/train, GET .../status
├── llm/
│   ├── engine.py     # httpx client to llama-server, LoRA swap, inference
│   └── client.py     # High-level: synthesis prompts, knowledge extraction prompts
├── storage/
│   └── sqlite.py     # Per-agent SQLite databases (with embedding BLOBs)
└── training/
    ├── data_prep.py  # Memories → Q&A training pairs
    ├── train.py      # CLI entry point, platform dispatch
    ├── train_mlx.py  # MLX LoRA training (macOS)
    ├── train_unsloth.py  # unsloth QLoRA (Linux/NVIDIA)
    └── convert.py    # MLX/HF adapter → GGUF conversion

data/agents/{agent_id}/
├── memories.db             # SQLite database (content + embedding BLOBs)
├── adapter.gguf            # LoRA adapter (when trained)
├── adapter_raw/            # Raw training output (safetensors)
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
2. Loads agent's LoRA adapter if it exists (`data/agents/avery/adapter.gguf`)
3. Sends memories + query to Qwen for synthesis
4. Returns raw memories + synthesized answer

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
| `MEMORY_ENGINE_SERVER_URL` | `http://localhost:2027` | llama-server URL |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (fastembed) |
| `MEMORY_DATA_DIR` | `data/agents` | Where agent DBs and adapters live |
| `MEMORY_TRAINING_MIN_MEMORIES` | `100` | Min memories before LoRA training |
| `MEMORY_LLAMA_CPP_DIR` | `~/src/llama.cpp` | Path to llama.cpp source (needs `convert_lora_to_gguf.py`) |

## Agent-Neutral Design

The `agent_id` is an opaque string namespace. Any agent that can make HTTP calls can use it. The memory server has no knowledge of any specific agent system.

## LoRA Adapter Flow

Per-agent LoRA adapters live at `data/agents/{agent_id}/adapter.gguf`. When an agent calls `/v1/memory/recall`, the engine tells llama-server to load the adapter via its `/lora-adapters` API before running inference.

### Training LoRA adapters (Phase 3)

Trigger via `POST /v1/memory/train/{agent_id}`. The training pipeline runs as a subprocess:

1. Collect accumulated memories for the agent from SQLite
2. Generate Q&A training pairs using the LLM (via `/v1/internal/generate`)
3. Fine-tune a LoRA adapter: MLX on macOS, unsloth on Linux
4. Convert to GGUF format (`~/src/llama.cpp/convert_lora_to_gguf.py`)
5. Place at `data/agents/{agent_id}/adapter.gguf`
6. Next recall request automatically loads it via llama-server

## Development Phases

### Phase 1: Smart RAG (done)

- [x] SQLite storage with keyword-based search
- [x] Qwen for synthesis and knowledge extraction (via llama-server)
- [x] REST API (store, recall, ingest, status)
- [x] Per-agent isolation
- [ ] Add MCP server wrapper for Claude SDK agents
- [ ] Add post-session ingest hook for agent orchestrators
- [ ] Service setup (launchd/systemd)
- [ ] Tests

### Phase 2: Embedding-Based Retrieval (done — tested end-to-end)

- [x] fastembed (ONNX, BAAI/bge-small-en-v1.5, 384-dim) — no PyTorch dependency
- [x] Embeddings stored as BLOBs in SQLite, cosine similarity via numpy
- [x] Auto-migration: adds `embedding` column to existing DBs
- [x] Embed on store and ingest, keyword fallback for unembedded memories
- [x] Reindex endpoint: `POST /v1/memory/reindex/{agent_id}`
- [x] Verified with real data (store + semantic recall + ingest)

### Phase 3: LoRA Memory (done — code written, untested)

- [x] Dynamic LoRA swapping via llama-server's `/lora-adapters` API
- [x] Adapter path check per agent (`data/agents/{id}/adapter.gguf`)
- [x] Training pipeline: data_prep → train → convert → adapter.gguf
- [x] Platform-adaptive: MLX (macOS) / unsloth (Linux)
- [x] Training runs as subprocess — doesn't load training libs into server
- [x] API: `POST /v1/memory/train/{agent_id}`, `GET .../status`
- [ ] Test with real LoRA adapter
- [ ] Automatic "nap" cycle trigger (after N ingests)

### Phase 4: Doc2LoRA Hypernetwork

- Train a hypernetwork that generates LoRA adapters in a single forward pass
- Sub-second adapter updates instead of minutes of fine-tuning
- Real-time memory encoding — store and immediately recall through weights

## Platform Support

| Platform | GPU Backend | Quantization | Status |
|----------|------------|--------------|--------|
| macOS (M-series) | Metal | Q4_K_XL | Primary dev target |
| Linux (NVIDIA) | CUDA | Q8_K_XL | Production target (Blackwell) |

The memory server itself is pure Python (no native deps beyond ONNX for embeddings). GPU backend is determined by how llama-server is compiled (see Quick Start above).

## Key Files

| File | Purpose |
|------|---------|
| `src/memory_server/llm/engine.py` | Core: httpx client to llama-server, LoRA swap, inference |
| `src/memory_server/llm/client.py` | Prompts for synthesis and knowledge extraction |
| `src/memory_server/embeddings.py` | Embedding model singleton (fastembed) |
| `src/memory_server/storage/sqlite.py` | Per-agent SQLite storage with embedding BLOBs |
| `src/memory_server/config.py` | All settings (pydantic-settings, env vars) |
| `src/memory_server/api/recall.py` | The main endpoint — search + synthesize |
| `src/memory_server/api/ingest.py` | Background transcript processing |
| `src/memory_server/api/train.py` | Training trigger + status endpoints |
| `src/memory_server/training/train.py` | Training pipeline entry point |
| `src/memory_server/training/convert.py` | Adapter format conversion → GGUF |
