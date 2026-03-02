# Memory Server

Neural long-term memory for AI agents. Store knowledge, recall it semantically, and — over time — encode it into per-agent LoRA adapters so the model *knows* what an agent has learned, not just retrieves text about it.

Inspired by Sakana AI's [Doc-to-LoRA](https://pub.sakana.ai/doc-to-lora/) research on hypernetworks that convert documents into LoRA adapters in a single forward pass.

## How it works

```
         store("Meta Q4 revenue was $43B")
                    │
                    ▼
┌──────────────────────────────────────────┐
│           Memory Server (:2090)          │
│                                          │
│   1. Embed text (bge-small-en-v1.5)     │
│   2. Store in per-agent SQLite DB       │
│   3. On recall: cosine similarity →     │
│      top-k memories → LLM synthesis     │
│   4. Optional: LoRA adapter per agent   │
│      for weight-space memory            │
└──────────────┬───────────────────────────┘
               │ HTTP
┌──────────────▼───────────────────────────┐
│         llama-server (:2027)             │
│   Qwen 3.5 35B A3B · LoRA hot-swap     │
└──────────────────────────────────────────┘
```

Any agent that speaks HTTP can use it. The `agent_id` is an opaque string — each agent gets its own SQLite database, its own embeddings, and (optionally) its own LoRA adapter.

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) build 8171+ (for `qwen35moe` architecture support)

### 1. Start llama-server

The memory server needs a running llama-server for LLM inference. To enable LoRA adapter support, start with `--lora` and `--lora-init-without-apply` flags (see [LoRA adapters](#lora-adapters) below).

**macOS (Metal):**

```bash
llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 --port 2027 \
  --jinja -ngl 99 --threads -1 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --lora data/agents/my-agent/adapter.gguf --lora-init-without-apply
```

**Linux (NVIDIA):**

```bash
llama-server \
  -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q8_K_XL \
  --host 0.0.0.0 --port 2027 \
  --jinja -ngl 99 --threads -1 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --lora data/agents/my-agent/adapter.gguf --lora-init-without-apply
```

First run downloads the model (~20GB, cached in `~/.cache/llama.cpp/`).

### 2. Start memory server

```bash
git clone https://github.com/kortexa-ai/memory.server.git
cd memory.server
uv sync
uv run memory-server
```

The server starts on `http://localhost:2090`. First run downloads the embedding model (~33MB).

Swagger docs at `http://localhost:2090/docs`.

## API

### Store a memory

```bash
curl -X POST http://localhost:2090/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "content": "The user prefers dark mode and Vim keybindings",
    "source": "session-001",
    "tags": ["preferences", "ui"]
  }'
```

### Recall from memory

```bash
curl -X POST http://localhost:2090/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "query": "What are the user preferences?"
  }'
```

Returns the top matching memories (by embedding similarity) plus a synthesized answer from the LLM.

### Ingest a transcript

```bash
curl -X POST http://localhost:2090/v1/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "transcript": "User: I just switched to Neovim. Agent: Nice! ...",
    "session_id": "session-002",
    "extract_knowledge": true
  }'
```

The LLM extracts discrete facts from the transcript and stores each as a separate memory with embeddings.

### Check agent status

```bash
curl http://localhost:2090/v1/memory/status/my-agent
```

### Train a LoRA adapter

Once an agent has accumulated enough memories (default: 100+), you can train a personalized LoRA adapter:

```bash
curl -X POST http://localhost:2090/v1/memory/train/my-agent
```

Training runs as a background subprocess (MLX on macOS, unsloth on Linux). After training completes, restart llama-server to load the new adapter (see [LoRA adapters](#lora-adapters) below).

## Configuration

All settings via environment variables (prefix `MEMORY_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HOST` | `0.0.0.0` | Bind address |
| `MEMORY_PORT` | `2090` | Server port |
| `MEMORY_ENGINE_SERVER_URL` | `http://localhost:2027` | llama-server URL |
| `MEMORY_DATA_DIR` | `data/agents` | Where agent databases and adapters live |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (via fastembed) |
| `MEMORY_TRAINING_MIN_MEMORIES` | `100` | Minimum memories before training is allowed |
| `MEMORY_LLAMA_CPP_DIR` | `~/src/llama.cpp` | llama.cpp source dir (for LoRA → GGUF conversion) |

## Architecture

The memory server is a FastAPI app that handles storage, retrieval, and training. It does **not** load the LLM itself — it connects to a separate llama-server instance via HTTP.

This separation gives us:
- Latest llama.cpp features without waiting for Python bindings
- Proper continuous batching and KV cache management
- Dynamic LoRA adapter swapping via `/lora-adapters` API
- The LLM can serve multiple clients (memory server, other tools, direct use)

### Retrieval pipeline

1. **Store**: text → embedding (bge-small-en-v1.5, 384-dim) → SQLite with BLOB
2. **Recall**: query → embedding → cosine similarity over all agent memories → top-k → LLM synthesis
3. **Ingest**: transcript → LLM extracts facts → each fact stored + embedded

### LoRA adapters

llama-server **cannot** load LoRA adapters dynamically from disk at runtime. Adapters must be specified at startup with `--lora`. The API only toggles adapters on/off — it can't add new ones.

This means:

- The `--lora path.gguf` file must exist before llama-server starts
- `--lora-init-without-apply` loads the adapter but doesn't activate it (scale=0), so the memory server can enable it per-request
- After training a new adapter, llama-server must be restarted to pick it up

**Null adapters:** If no real adapter exists yet, use `memory-ensure-adapter` to create a 192-byte no-op GGUF file. This lets llama-server start with `--lora` before any training has happened — the null adapter loads zero tensors and has zero effect on inference.

```bash
# Creates a null adapter only if the file doesn't exist; no-op if it does
cd ~/src/memory.server
uv run memory-ensure-adapter data/agents/my-agent/adapter.gguf
```

**Enabling/disabling via API:**

```bash
# Enable adapter (id is 0-indexed order of --lora flags)
curl -X POST http://localhost:2027/lora-adapters \
  -d '[{"id": 0, "scale": 1.0}]'

# Disable all adapters
curl -X POST http://localhost:2027/lora-adapters -d '[]'
```

### LoRA training pipeline

1. Read accumulated memories from SQLite
2. Generate Q&A training pairs via the LLM
3. Fine-tune QLoRA adapter (MLX on macOS, unsloth on Linux)
4. Convert to GGUF format (direct gguf-py conversion, bypasses llama.cpp's converter which can't handle Qwen3.5 MoE)
5. Place at `data/agents/{agent_id}/adapter.gguf`
6. Restart llama-server to load the new adapter

## Roadmap

- **Phase 1** (done): SQLite storage, LLM synthesis, REST API
- **Phase 2** (done): Embedding-based semantic retrieval (fastembed + cosine similarity)
- **Phase 3** (done): Per-agent LoRA training pipeline + dynamic adapter swapping
- **Phase 4** (planned): [Doc2LoRA](https://pub.sakana.ai/doc-to-lora/) hypernetwork for sub-second adapter generation

## Platform support

| Platform | GPU Backend | Quantization | Status |
|----------|------------|--------------|--------|
| macOS (M-series) | Metal | Q4_K_XL | Primary dev target |
| Linux (NVIDIA) | CUDA | Q8_K_XL | Production target |

The memory server itself is pure Python. GPU backend depends on how llama-server is compiled.

## License

MIT
