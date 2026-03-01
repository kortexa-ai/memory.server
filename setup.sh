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

echo "Installing dependencies..."
uv sync

echo ""
echo "Setup complete! Embedding model downloads automatically on first run (~33MB)."
echo "Make sure llama-server is running on port 2027 (see CLAUDE.md for the command)."
echo "----------------------------------------------------------------"
echo "Run: ./run.sh                    # Start memory server on port 2090"
echo "     uv run memory-server        # Same thing"
echo "     curl localhost:2090/health   # Check health"
echo "     curl localhost:2090/docs     # Swagger UI"
echo "----------------------------------------------------------------"
