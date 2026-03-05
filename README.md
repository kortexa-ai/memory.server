# Memory Server

Neural long-term memory for AI agents. Store knowledge, recall it semantically, and — over time — encode it into per-agent LoRA adapters so the model *knows* what an agent has learned, not just retrieves text about it.

Inspired by Sakana AI's [Doc-to-LoRA](https://pub.sakana.ai/doc-to-lora/) research on hypernetworks that convert documents into LoRA adapters in a single forward pass.

**macOS (Apple Silicon) only** — uses MLX for both inference and training.

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
               │ HTTP (localhost)
┌──────────────▼───────────────────────────┐
│    mlx-lm server (:2091, child proc)    │
│   Qwen 3.5 35B A3B · per-request LoRA  │
└──────────────────────────────────────────┘
```

Any agent that speaks HTTP can use it. The `agent_id` is an opaque string — each agent gets its own SQLite database, its own embeddings, and (optionally) its own LoRA adapter.

## Quick start

### Prerequisites

- macOS with Apple Silicon (M1+)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Start memory server

```bash
git clone https://github.com/kortexa-ai/memory.server.git
cd memory.server
uv sync
uv run memory-server
```

The server starts on `http://localhost:2090` and launches an mlx-lm server on port 2091 as a subprocess. First run downloads the embedding model (~33MB) and MLX model (~19.5GB, cached in `~/.cache/huggingface/`).

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

Training runs as a background subprocess using mlx-lm. After training completes, the adapter is picked up on the next request — no restart needed.

## Configuration

All settings via environment variables (prefix `MEMORY_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HOST` | `0.0.0.0` | Bind address |
| `MEMORY_PORT` | `2090` | Server port |
| `MEMORY_MODEL_REPO` | `NexVeridian/Qwen3.5-35B-A3B-4bit` | MLX model for inference + training |
| `MEMORY_MODEL_SERVER_PORT` | `2091` | mlx-lm server port |
| `MEMORY_DATA_DIR` | `data/agents` | Where agent databases and adapters live |
| `MEMORY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (via fastembed) |
| `MEMORY_TRAINING_MIN_MEMORIES` | `100` | Minimum memories before training is allowed |

## Architecture

The memory server is a FastAPI app that handles storage, retrieval, and training. It launches an mlx-lm server as a subprocess for LLM inference.

This gives us:
- ~74 tok/s inference on M4 Pro (3x faster than llama-server)
- Per-request LoRA adapter loading — no restart after training
- Same model for inference and training (no format conversion)
- Memory server controls the inference server's lifecycle

### Retrieval pipeline

1. **Store**: text → embedding (bge-small-en-v1.5, 384-dim) → SQLite with BLOB
2. **Recall**: query → embedding → cosine similarity over all agent memories → top-k → LLM synthesis
3. **Ingest**: transcript → LLM extracts facts → each fact stored + embedded

### LoRA adapters

mlx-lm loads LoRA adapters per-request via the `"adapters"` field in chat completion requests. Adapters are stored as safetensors (the native MLX training output format) — no conversion step needed.

After training, the adapter at `data/agents/{agent_id}/adapter/adapters.safetensors` is loaded on the next recall request automatically.

### LoRA training pipeline

1. Read accumulated memories from SQLite
2. Generate Q&A training pairs via the LLM
3. Fine-tune QLoRA adapter using mlx-lm
4. Output: `data/agents/{agent_id}/adapter/adapters.safetensors` — loaded directly, no conversion

## Roadmap

- **Phase 1** (done): SQLite storage, LLM synthesis, REST API
- **Phase 2** (done): Embedding-based semantic retrieval (fastembed + cosine similarity)
- **Phase 3** (done): Per-agent LoRA training + per-request adapter loading via mlx-lm
- **Phase 4** (planned): [Doc2LoRA](https://pub.sakana.ai/doc-to-lora/) hypernetwork for sub-second adapter generation

## License

MIT
