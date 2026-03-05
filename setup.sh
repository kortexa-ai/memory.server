#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo ""
    echo "To install uv, run one of the following:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  brew install uv"
    echo ""
    exit 1
fi

echo "Installing dependencies (includes mlx-lm for inference + training)..."
uv sync

echo ""
echo "Setup complete! Embedding model downloads automatically on first run (~33MB)."
echo "The mlx-lm model (~19.5GB) downloads on first server start."
echo "----------------------------------------------------------------"
echo "Run: ./run.sh                    # Start memory server on port 2090"
echo "     uv run memory-server        # Same thing"
echo "     curl localhost:2090/health   # Check health"
echo "     curl localhost:2090/docs     # Swagger UI"
echo "----------------------------------------------------------------"
