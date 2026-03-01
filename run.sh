#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "[memory.server] UV_MISSING" >&2
    exit 2
fi

# Defaults (allow override via environment)
HOST=${MEMORY_HOST:-0.0.0.0}
PORT=${MEMORY_PORT:-2090}

echo "Starting Memory Server on ${HOST}:${PORT}..."

uv run memory-server "$@"
