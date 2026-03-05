# Migration Plan: LLM Work → DGX Spark

## The Machine

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Memory | 128GB unified (CPU+GPU shared) |
| CPU | 20-core ARM64 (Grace) |
| Storage | 3.7TB NVMe |
| OS | Ubuntu 24.04 (aarch64) |
| CUDA | 13.0, Driver 580.126.09 |
| Tools | Python 3.14, uv, Docker |
| SSH | `francip-spark` or `7` (passwordless) |

## Architecture: Split Deployment

Keep the memory server on Mac, move only the LLM to Spark.

```
Mac (current)                          Spark (new)
┌──────────────────────┐               ┌──────────────────────────┐
│  Memory Server :2090 │               │  vLLM server :2091       │
│  FastAPI + embeddings│──── HTTP ────▶│  Qwen 3.5 35B A3B       │
│  SQLite storage      │               │  Per-request LoRA        │
│  fastembed (ONNX)    │               │  128GB unified memory    │
└──────────────────────┘               │                          │
                                       │  Training (on demand)    │
                                       │  unsloth / PEFT          │
                                       └──────────────────────────┘
```

**Why split, not move everything:**
- SQLite stays local (no NFS headaches)
- Embeddings are fast on CPU anyway (~33MB model)
- Only the LLM benefits from the big GPU
- Minimal code changes — just point engine.py at `francip-spark:2091`
- Can fall back to local mlx-lm if Spark is down

## Step 1: Set Up vLLM on Spark

vLLM is the best fit: OpenAI-compatible API (same as mlx-lm), native per-request LoRA loading, great Qwen MoE support.

```bash
ssh francip-spark

# Create project directory
mkdir -p ~/src/memory-llm && cd ~/src/memory-llm

# Install vLLM (needs CUDA 13 compatible build)
uv init && uv add vllm torch

# Download Qwen 3.5 35B A3B (HuggingFace native weights, not MLX format)
# With 128GB unified memory, we can run FP16 (~70GB) — no quantization needed!
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-35B-A3B')"

# Launch vLLM server
uv run vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 2091 \
  --enable-lora \
  --max-lora-rank 64 \
  --max-model-len 4096 \
  --dtype float16
```

**FP16 on Spark vs 4-bit on Mac:**
- Mac: 4-bit quantized (~19.5GB) because only 64-128GB shared with OS
- Spark: FP16 (~70GB) — 128GB unified, plenty of room. Better quality, still fast.

### vLLM LoRA per-request

vLLM supports per-request LoRA via the `model` field in OpenAI-compatible API:

```bash
# Register adapter at startup or via API
# Then reference by name in chat completion:
curl http://francip-spark:2091/v1/chat/completions \
  -d '{"model": "francip-adapter", "messages": [...]}'
```

Alternatively, vLLM can load adapters from a directory dynamically.

## Step 2: Modify engine.py for Remote LLM

Minimal change — the engine already communicates via HTTP. Just need to:

1. Add a `model_server_host` config setting (default `localhost`)
2. Skip subprocess launch when host is remote
3. Adjust LoRA adapter loading for vLLM's API

```python
# config.py — add one setting:
model_server_host: str = "localhost"  # or "francip-spark"

# engine.py — key changes:
@property
def _base_url(self) -> str:
    return f"http://{settings.model_server_host}:{settings.model_server_port}"

async def connect(self) -> None:
    if settings.model_server_host == "localhost":
        # Launch local mlx-lm server (existing code)
        ...
    else:
        # Remote server — just connect, don't launch
        self._client = httpx.AsyncClient(...)
        # Wait for remote server to be reachable
        ...
```

## Step 3: Move Training to Spark

Training benefits the most from CUDA. Two options:

### Option A: unsloth (fast, easy)

```bash
ssh francip-spark
cd ~/src/memory-llm

uv add unsloth

# Train LoRA — unsloth handles Qwen MoE
uv run python train_cuda.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --data training_data.jsonl \
  --output adapter/ \
  --iters 1000 \
  --lora-rank 8
```

### Option B: HF PEFT + transformers (more control)

```bash
uv add peft transformers accelerate bitsandbytes
```

### Training workflow

1. Mac generates training data (data_prep.py — uses LLM on Spark for Q&A generation)
2. `scp` training data to Spark (or Spark pulls via SSH)
3. Train on Spark → adapter saved to Spark's disk
4. vLLM picks up new adapter (no restart needed with dynamic loading)

**Note:** Adapter stays on Spark (where vLLM loads it). No need to copy back to Mac.

## Step 4: Adapter Sync

With split deployment, adapters live on Spark (where the LLM server runs). The adapter path in engine.py becomes a vLLM adapter name, not a local file path.

```python
# Before (local mlx-lm):
body["adapters"] = "/path/to/local/adapter"

# After (remote vLLM):
body["model"] = "francip-adapter"  # vLLM registered adapter name
```

## Step 5: Resume Ingestion on Spark

Once vLLM is running on Spark, ingestion becomes dramatically faster:
- FP16 model (better quality than 4-bit)
- Blackwell GB10 >> M4 Pro for inference throughput
- Remaining 558 sessions could finish in hours instead of days

```bash
# On Mac — just point at Spark and run
MEMORY_MODEL_SERVER_HOST=francip-spark uv run python -u scripts/ingest_sessions.py --agent-id francip
```

## Step 6: systemd Service on Spark

```ini
# /etc/systemd/system/memory-llm.service
[Unit]
Description=Memory LLM Server (vLLM)
After=network.target

[Service]
Type=simple
User=francip
WorkingDirectory=/home/francip/src/memory-llm
ExecStart=/home/francip/.local/bin/uv run vllm serve Qwen/Qwen3.5-35B-A3B --host 0.0.0.0 --port 2091 --enable-lora --max-lora-rank 64 --dtype float16
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Implementation Order

| # | Task | Time | Depends On |
|---|------|------|------------|
| 1 | Install vLLM + download model on Spark | ~1hr (model download) | Spark access |
| 2 | Test vLLM serves Qwen correctly | 10 min | 1 |
| 3 | Add `model_server_host` config + remote connect | 20 min | — |
| 4 | Test recall/ingest against Spark | 10 min | 2, 3 |
| 5 | Resume + finish ingestion via Spark | hours | 4 |
| 6 | Set up training pipeline on Spark | 1-2 hr | 2 |
| 7 | Retrain LoRA on full dataset | ~30 min | 5, 6 |
| 8 | systemd service + auto-start | 15 min | 2 |

## Fallback

If vLLM has issues on CUDA 13 / aarch64 (it's bleeding edge), alternatives:
1. **SGLang** — similar to vLLM, good Qwen support, OpenAI-compatible
2. **TensorRT-LLM** — NVIDIA's own, guaranteed to work on DGX but more complex setup
3. **ollama** — dead simple but weaker LoRA support
4. **Docker** — run any of the above in a container if native install is painful

## What Doesn't Change

- Memory server (FastAPI on Mac)
- SQLite storage
- Embeddings (fastembed, ONNX, runs on Mac CPU)
- API endpoints
- Ingest script
- All data lives on Mac (except adapter files → Spark)
