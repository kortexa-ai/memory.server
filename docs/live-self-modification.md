# Live Self-Modification: Model-Driven LoRA Rewriting

## The Idea

Give a model a tool that lets it rewrite its own LoRA weights mid-conversation. The model decides "I need to think differently about X," calls the tool, and its next inference is literally running through different weights. A brain rewiring itself between thoughts.

Two independent paths to get there:

- **Path 1 — Live Conversation Adapter** (fast path): The model updates its own weights from the current conversation. Minimal infrastructure, get to "let's experiment" fast.
- **Path 2 — Memory Corpus Hypernetwork** (long path): Train a hypernetwork on 400K+ accumulated memories to generate LoRA adapters in sub-second time. Better results, much longer to build.

They share the same plumbing (mlx-lm per-request adapter loading, split adapter strategy) and converge eventually, but Path 1 can start producing experiments in days, not months.

## Shared Foundation

### Why This Is Possible Here

- **mlx-lm loads LoRA adapters per-request** via the `"adapters"` field in the chat completion body. No restart, no reload — just point to a different safetensors file.
- **Per-agent adapter isolation** — each agent has its own `data/agents/{id}/adapter/adapters.safetensors`.
- **Training pipeline exists** — data_prep → mlx-lm LoRA → safetensors, picked up on next request.

### Our LoRA: The MoE Complication

Qwen 3.5 35B A3B is a Mixture-of-Experts model. Our actual adapter breakdown:

```
Attention LoRA:       3,250,176 params  (  6.2 MB)  —  1.3%
Shared expert LoRA:   1,540,224 params  (  2.9 MB)  —  0.6%
MoE expert LoRA:    251,658,240 params  (480.0 MB)  — 98.1%  ← 256 experts
─────────────────────────────────────────────────────────────
Total:              256,448,640 params  (489.1 MB)

408 tensors across 16 layers (layers 24-39)
26 tensors per layer, including 3D switch_mlp tensors (256, rank, dim)
```

**Strategy for both paths: split the adapter.**

1. **Attention + shared expert LoRA** (~4.8M params, 9MB) — the part we modify live. Controls *how* the model reasons: what it attends to, reasoning style, focus areas.
2. **MoE expert LoRA** (~252M params, 480MB) — static from overnight training. Encodes *deep specialized knowledge* across the expert routing.

At inference time, we compose the two parts into a single adapter file that mlx-lm loads.

### Shared Tool Interface

Both paths expose the same tool. The backend changes, the interface doesn't.

```json
{
  "name": "reshape_thinking",
  "description": "Modify your own reasoning patterns by updating your LoRA adapter weights. Changes take effect on your next inference call.",
  "parameters": {
    "agent_id": "string — your agent identity",
    "intent": "string — what you want to change about your thinking",
    "examples": [{"input": "...", "output": "..."}],
    "strength": "float 0.0-1.0 — how much to shift weights (default 0.3)"
  }
}
```

### Safety (Both Paths)

- **Strength cap** — default 0.3, max 1.0. Prevents violent weight swings.
- **Adapter versioning** — keep last N adapters for rollback.
- **MoE never touched** — overnight-trained expert LoRA is read-only during live reshape.
- **Rate limiting** — max 1 reshape per conversation turn.
- **Runaway detection** — alert after 5 reshape calls in a session.

---

# Path 1: Live Conversation Adapter (Fast Path)

**Goal:** Model updates its own attention LoRA from the current conversation. Get to a working experiment as fast as possible, then iterate.

**Principle:** Don't let perfect be the enemy of interesting. Even if the behavior change is subtle or the latency is 30 seconds, we learn something by trying it.

## Step 1: Adapter Split Tool (day 1)

Write a script that splits the existing adapter into two files:
- `data/agents/{id}/adapter/attn_base.safetensors` — attention + shared expert tensors
- `data/agents/{id}/adapter/moe_static.safetensors` — MoE expert tensors

And a compose function that merges them back into a single `adapters.safetensors` for mlx-lm.

This is just MLX array ops + safetensors I/O — no LLM, no training, pure plumbing.

**Deliverable:** `scripts/split_adapter.py` + compose utility in `memory_server/training/`.

## Step 2: Micro-Training Reshape Endpoint (days 2-3)

Add `POST /v1/memory/reshape` to the memory server:

1. Accept `intent` (string) + `examples` (list of input/output pairs) + `strength` (float)
2. If no examples provided, use intent to auto-generate a few Q&A pairs via LLM (quick, 1 call)
3. Write examples as JSONL in chat-completion format
4. Shell out to `mlx-lm lora` targeting only attention layers:
   - `--num-layers 16` (our 16 LoRA layers)
   - `--iters 20` (fast, ~15s)
   - `--batch-size 1`
   - Use the attention-only adapter as base
5. Interpolate: `new_attn = (1 - strength) * attn_base + strength * fresh_attn`
6. Compose with static MoE adapter → write `adapters.safetensors`
7. Return: `{"status": "reshaped", "latency_ms": ..., "params_modified": ...}`

**Expected latency:** ~15-30s for 20 iterations on attention-only LoRA (much smaller than full adapter).

**Deliverable:** Working endpoint, callable via curl.

## Step 3: Conversation-Driven Auto-Reshape (days 4-5)

The model shouldn't have to manually craft examples. Instead:

1. After each user turn, extract the last N exchanges from the conversation
2. The model calls `reshape_thinking(intent="[summary of what I'm learning from this conversation]")`
3. The endpoint auto-generates training examples from the conversation context:
   - Take recent exchanges → LLM extracts key patterns/corrections
   - Format as Q&A pairs representing "how I should think about this topic"
4. Run micro-training, update adapter

This can run **async** — train in the background while the conversation continues. The adapter updates silently after 1-2 turns. The model doesn't block waiting.

```
Turn 1: User asks about React performance
Turn 2: Model answers (using old adapter)
         → background: extract patterns, micro-train, update adapter
Turn 3: User follows up
Turn 4: Model answers (now using updated adapter) ← behavior shift here
```

**Deliverable:** Async reshape triggered by conversation flow, not just explicit tool calls.

## Step 4: Experiment & Measure (week 2)

This is where we find out if it actually works:

1. **A/B test**: Same questions, with and without live reshape. Is there a measurable difference?
2. **Conversation continuity**: Does reshaping mid-conversation help with follow-up questions?
3. **Attention-only impact**: Is modifying just attention LoRA enough to see behavior change? Or do we need shared expert too?
4. **Optimal parameters**: How many training iters? What learning rate? What strength?

We might discover that 20 iterations on 4.8M attention params does nothing perceptible. Or we might discover it's transformative. Either way, we learn something concrete.

**Deliverable:** Eval script + results log.

## Step 5: MCP Tool Wrapper (week 2-3)

Wrap the reshape endpoint as an MCP tool so Claude SDK agents can call it:

```python
@mcp.tool()
async def reshape_thinking(agent_id: str, intent: str, examples: list = None, strength: float = 0.3):
    """Modify your reasoning patterns by updating your LoRA adapter weights."""
    resp = await httpx.post(f"{MEMORY_SERVER}/v1/memory/reshape", json={...})
    return resp.json()
```

At this point an agent can literally modify its own weights mid-conversation. Whether it's *useful* is what Step 4 tells us.

**Deliverable:** MCP server config, agent can call reshape_thinking.

## Path 1 Timeline

```
Day 1:     Adapter split script
Days 2-3:  Reshape endpoint (micro-training)
Days 4-5:  Async conversation-driven reshape
Week 2:    Experiment, measure, tune
Week 2-3:  MCP tool wrapper

→ "Let's experiment" stage: end of day 5
→ "Is this actually useful?" answer: end of week 2
```

---

# Path 2: Memory Corpus Hypernetwork (Long Path)

**Goal:** Train a hypernetwork that generates LoRA adapters in a single forward pass (<1s). Trained on our 400K+ memory corpus. Better quality than micro-training, sub-second latency.

**Principle:** Do this right. The data pipeline and training are slow but the result is a fundamentally better system.

## Prior Art

- **[SHINE](https://arxiv.org/abs/2602.06358)** (Feb 2026) — In-context hypernetwork. Processes context through base LLM with learnable memory embeddings, 4-layer M2P Transformer generates LoRA weights. Tested on Qwen3-8B. Adapter generation: 0.3s.
- **[Text-to-LoRA](https://arxiv.org/abs/2506.06105)** (Jun 2025) — Natural language → LoRA in one forward pass. Trained on 9 adapters, generalizes to unseen tasks.
- **[Doc-to-LoRA](https://github.com/SakanaAI/doc-to-lora)** (Feb 2026, Sakana AI) — Perceiver cross-attention hypernetwork (~309M params). KL distillation training. Sub-second. Targets Gemma-2-2b. MIT licensed, code available.

Key insight from all three: **feed text through the frozen base LLM first**, extract per-layer activations, then use a learned module to map activations → LoRA weights. The base model itself is part of the hypernetwork.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  Hypernetwork Pipeline                                         │
│                                                                │
│  1. Tokenize intent/document text                              │
│                                                                │
│  2. Forward pass through frozen Qwen 3.5 35B A3B               │
│     → extract per-layer hidden states from layers 24-39        │
│     → h_i ∈ (seq_len, 2048) for each of 16 layers             │
│                                                                │
│  3. Perceiver cross-attention encoder (learnable)              │
│     → M learnable query tokens cross-attend to h_i             │
│     → compress variable-length activations to fixed size       │
│     → output: (16, M, 2048) — per-layer memory states         │
│                                                                │
│  4. M2P Transformer (4 layers, sparse attention)               │
│     → alternating column (across layers) and row attention     │
│     → output: (16, M, 2048) — refined memory states           │
│                                                                │
│  5. Linear projections → LoRA weight matrices                  │
│     → per layer: reshape memory → lora_A, lora_B              │
│     → attention + shared expert only (~300K params/layer)      │
│                                                                │
│  Output: 4.8M params as attention/shared-expert LoRA deltas    │
│  Latency: <1s (base model forward + hypernetwork forward)      │
└────────────────────────────────────────────────────────────────┘
```

### Hypernetwork Size

```
Perceiver encoder:    ~17M params  (32 query tokens, 4 heads, dim 2048)
M2P Transformer:      ~34M params  (4 layers, sparse attn, dim 2048)
Output projections:  ~154M params  (16 layers × linear → LoRA matrices)
─────────────────────────────────
Total:               ~200M params  (~400MB float16)
```

Fits comfortably on M4 Pro alongside the base model.

### Training Approach: KL Distillation

Following Doc-to-LoRA (which found KL distillation significantly outperforms next-token prediction):

```
Teacher: Qwen 3.5 35B + document in context → output distribution P_teacher
Student: Qwen 3.5 35B + hypernetwork-generated LoRA (no document) → P_student
Loss:    KL(P_teacher || P_student)
```

No need to pre-train hundreds of individual LoRA adapters. The teacher signal is free.

### Long Document Chunking

For documents longer than training context:
- Chunk into 1024-token segments
- Generate a rank-r LoRA per chunk
- Concatenate along rank dimension → effective rank r×K
- More context = higher effective rank, elegant scaling

## Phase 2A: Data Pipeline (weeks, LLM-bound)

Generate training data from our memory corpus:

1. **Cluster memories** by embedding similarity (k-means or HDBSCAN)
   - Target: 200-500 clusters of related knowledge
   - Each cluster = one training example's "document"
2. **Generate eval Q&A pairs** per cluster
   - Reuse `data_prep.py` pipeline (batched LLM calls)
   - 5-10 Q&A pairs per cluster
3. **Pre-compute teacher distributions**
   - For each (document, question): run base model with document in context
   - Cache the output logits/distributions
   - This is the expensive part — one forward pass per (doc, question) pair

**Bottleneck:** LLM inference for Q&A generation and teacher caching. Same ~25s/call bottleneck as the ingestion pipeline. For 500 clusters × 10 Q&A = 5,000 teacher forward passes = ~35 hours.

**Deliverable:** Training dataset in `data/hypernetwork/train/`.

## Phase 2B: Hypernetwork Architecture in MLX (weeks)

1. **Port Perceiver cross-attention** from Doc-to-LoRA (PyTorch → MLX)
   - Doc-to-LoRA is MIT licensed, use as reference
   - MLX has `nn.MultiHeadAttention` — build Perceiver on top
2. **Implement M2P Transformer** (from SHINE)
   - Sparse attention: alternating column/row pattern
   - 4 layers, manageable complexity
3. **Output heads** — linear projections per layer → LoRA weight shapes
4. **Composition** — combine hypernetwork output with static MoE adapter

**Deliverable:** `src/memory_server/hypernetwork/` module, forward pass works on dummy inputs.

## Phase 2C: Train (weeks-months)

1. **Start small** — fewer layers (4 instead of 16), lower rank, 50 clusters
2. **Training loop:**
   - Sample (document, Q&A pairs, teacher distributions) from dataset
   - Forward: document → base model activations → hypernetwork → LoRA
   - Forward: question → base model + generated LoRA → student distribution
   - Loss: KL(teacher || student)
   - Backward: through hypernetwork only (base model frozen)
3. **Grad checkpointing** — 200M params + base model activations, tight on M4 Pro
4. **Validate** — hold out 20% of clusters, compare generated adapters vs teacher on held-out Q&A
5. **Scale up** — more layers, full data, longer training, tune hyperparams

**Compute estimate:** 24-48 hours for initial training, multi-day for full convergence.

**Deliverable:** Trained hypernetwork checkpoint.

## Phase 2D: Integration (days, after 2C)

1. Swap hypernetwork into the reshape endpoint (same tool interface as Path 1)
2. `intent text → base model forward → hypernetwork forward → LoRA → compose → save`
3. <1s instead of 15-30s
4. Model can reshape multiple times in a conversation

**Deliverable:** Sub-second reshape, same MCP tool.

## Path 2 Timeline

```
Weeks 1-2:   Data pipeline (clustering, Q&A gen, teacher caching) — LLM-bound
Weeks 2-4:   Hypernetwork architecture in MLX
Weeks 4-8:   Training + iteration
Week 8+:     Integration, eval, tuning

→ First "does it work at all?" checkpoint: week 6
→ Production-grade reshape: week 10+
```

---

# How the Paths Converge

```
                    Path 1 (fast)              Path 2 (long)
                    ─────────────              ──────────────
Day 1               Split adapter              (not started)
Day 5               Micro-training works       (not started)
Week 2              Measuring results          Data pipeline running
Week 3              MCP tool live              Architecture in MLX
Week 6              Iterating on params        First hypernetwork training
Week 10             Still micro-training       Hypernetwork ready
                            │                        │
                            └──────────┬─────────────┘
                                       │
                              Same tool interface
                              Same adapter split
                              Same MoE static base
                                       │
                                       ▼
                            Swap backend: micro-training → hypernetwork
                            15-30s reshape → <1s reshape
                            Everything else unchanged
```

Path 1 gives us a working experiment in days. Path 2 replaces the backend with something 30x faster. The agent never knows the difference — same tool call, faster response.

## What the Model Experiences

From the model's perspective, it has a tool that says "reshape your thinking." It provides a natural language intent and optionally some examples. After calling the tool, its responses genuinely change — not because it's roleplaying, but because the actual weight matrices in its LoRA adapter are different.

This creates a feedback loop:
1. Model notices it's bad at something
2. Calls reshape_thinking with corrective examples
3. Next response is measurably different
4. Model can evaluate whether the change helped
5. Call reshape_thinking again to refine

This is **actual metacognition** — not prompt-based self-reflection, but physical weight modification driven by the model's own judgment about its performance.

## Relation to Existing Pipeline

Neither path replaces the overnight training pipeline — they complement it:

| | Overnight LoRA | Path 1 (micro-train) | Path 2 (hypernetwork) |
|---|---|---|---|
| **When** | Nap cycle | Mid-conversation | Mid-conversation |
| **What** | Full adapter (attn + MoE) | Attention LoRA only | Attention LoRA only |
| **Data** | All memories | Conversation context | Intent text |
| **Latency** | ~2 min + ~3h prep | ~15-30s | <1s |
| **Scope** | Broad knowledge | Narrow correction | Narrow correction |
| **Persistence** | Permanent | Session-scoped | Session-scoped |

The overnight pipeline builds the foundation (including deep MoE expert knowledge). Live reshape does tactical attention adjustments. Nightly consolidation can merge the day's reshapes back into the base.

## Open Questions

- **Attention-only effectiveness**: Is modifying only attention LoRA sufficient for meaningful behavior change? Path 1 will answer this empirically.
- **MLX Perceiver**: No off-the-shelf implementation. Port from Doc-to-LoRA (MIT). Blocking for Path 2, not Path 1.
- **Async vs sync reshape**: Path 1 Step 3 proposes async (train in background, catch up after 1-2 turns). Is the delayed effect confusing? Or is it natural, like "sleeping on it"?
- **What triggers reshape?**: Explicit tool call? Automatic after every N turns? On detected confusion/errors? Start explicit (Path 1), evolve to automatic.
- **Evaluation**: How do we measure whether a reshape actually worked? Need before/after comparison on the same questions. Build this into the experiment harness.

## References

- [SHINE: Scalable In-Context Hypernetwork](https://arxiv.org/abs/2602.06358) — Qwen3-8B, M2P Transformer, 0.3s adapter generation
- [Text-to-LoRA: Instant Transformer Adaption](https://arxiv.org/abs/2506.06105) — natural language → LoRA, zero-shot generalization
- [Doc-to-LoRA](https://github.com/SakanaAI/doc-to-lora) (Sakana AI) — Perceiver hypernetwork, KL distillation, sub-second, MIT licensed
