#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

AGENT_ID="${1:-$(whoami)}"
DATA_DIR="data/agents/${AGENT_ID}"
ADAPTER_DIR="${DATA_DIR}/adapter_raw"
ADAPTER_DATA="${ADAPTER_DIR}/data"
GGUF_PATH="${DATA_DIR}/adapter.gguf"

# Training hyperparameters (override via environment)
ITERS=${TRAIN_ITERS:-1000}
NUM_LAYERS=${TRAIN_NUM_LAYERS:-16}
LEARNING_RATE=${TRAIN_LR:-1e-5}
BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${TRAIN_MAX_SEQ_LENGTH:-2048}
SAVE_EVERY=${TRAIN_SAVE_EVERY:-200}

# Check training data exists
if [ ! -f "${ADAPTER_DATA}/train.jsonl" ]; then
    echo "No training data at ${ADAPTER_DATA}/train.jsonl"
    echo "Run the training pipeline first to generate Q&A pairs:"
    echo "  uv run python -m memory_server.training.train --agent-id ${AGENT_ID}"
    exit 1
fi

PAIRS=$(wc -l < "${ADAPTER_DATA}/train.jsonl")
echo "Training LoRA adapter for agent '${AGENT_ID}'"
echo "  Data: ${PAIRS} Q&A pairs"
echo "  Iterations: ${ITERS}, Layers: ${NUM_LAYERS}, LR: ${LEARNING_RATE}"
echo "  Batch: ${BATCH_SIZE}, Max seq: ${MAX_SEQ_LENGTH}"
echo ""

# Train
uv run --extra train-mac python -m mlx_lm lora \
    --model NexVeridian/Qwen3.5-35B-A3B-4bit \
    --train \
    --data "${ADAPTER_DATA}" \
    --adapter-path "${ADAPTER_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --iters "${ITERS}" \
    --learning-rate "${LEARNING_RATE}" \
    --save-every "${SAVE_EVERY}" \
    --steps-per-report 10 \
    --grad-checkpoint \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --mask-prompt

echo ""
echo "Training complete. Converting to GGUF..."

# Convert to GGUF
uv run python -c "
from pathlib import Path
from memory_server.training.convert import convert_to_gguf
convert_to_gguf(Path('${ADAPTER_DIR}'), Path('${GGUF_PATH}'))
"

echo ""
echo "Done! Adapter at: ${GGUF_PATH} ($(du -h "${GGUF_PATH}" | cut -f1))"
echo "Restart llama-server to pick it up."
