"""Training endpoints — trigger and monitor LoRA adapter training."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..models import TrainRequest, TrainResponse, TrainStatus
from ..storage.sqlite import get_memory_count

router = APIRouter()

# Track running training subprocesses: agent_id → Popen
_training_procs: dict[str, subprocess.Popen] = {}


@router.post("/v1/memory/train/{agent_id}", response_model=TrainResponse)
async def start_training(agent_id: str, req: TrainRequest | None = None) -> TrainResponse:
    """Start LoRA adapter training for an agent.

    Launches the training pipeline as a subprocess so it doesn't block
    the server or load training dependencies into this process.
    """
    if req is None:
        req = TrainRequest()

    # Check if already training
    if agent_id in _training_procs:
        proc = _training_procs[agent_id]
        if proc.poll() is None:  # still running
            return TrainResponse(
                agent_id=agent_id,
                status="already_training",
                memory_count=get_memory_count(agent_id),
            )
        else:
            # Previous training finished — clean up
            del _training_procs[agent_id]

    # Check if enough memories
    memory_count = get_memory_count(agent_id)
    if memory_count < req.min_memories:
        return TrainResponse(
            agent_id=agent_id,
            status="insufficient_memories",
            memory_count=memory_count,
        )

    # Launch training as subprocess
    cmd = [
        sys.executable, "-m", "memory_server.training.train",
        "--agent-id", agent_id,
        "--max-iters", str(req.max_iters),
        "--lora-rank", str(req.lora_rank),
        "--batch-size", str(settings.training_batch_size),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _training_procs[agent_id] = proc

    return TrainResponse(
        agent_id=agent_id,
        status="training_started",
        memory_count=memory_count,
    )


@router.get("/v1/memory/train/{agent_id}/status", response_model=TrainStatus)
async def training_status(agent_id: str) -> TrainStatus:
    """Check the status of a training job for an agent."""
    agent_dir = settings.data_dir / agent_id
    status_path = agent_dir / "training_status.json"
    adapter_path = agent_dir / "adapter.gguf"

    adapter_exists = adapter_path.exists()
    adapter_updated = (
        datetime.fromtimestamp(adapter_path.stat().st_mtime, tz=timezone.utc)
        if adapter_exists
        else None
    )

    # Check status file written by the training subprocess
    status = "idle"
    progress = None
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            status = data.get("status", "idle")
        except (json.JSONDecodeError, OSError):
            pass

    # Cross-check with running process
    if agent_id in _training_procs:
        proc = _training_procs[agent_id]
        if proc.poll() is None:
            # Still running — trust the status file
            if status == "idle":
                status = "training"  # file might not be written yet
        else:
            # Process exited — check if it succeeded
            del _training_procs[agent_id]
            if status not in ("complete", "failed"):
                status = "failed"  # process exited but status file doesn't show complete

    return TrainStatus(
        agent_id=agent_id,
        status=status,
        progress=progress,
        adapter_exists=adapter_exists,
        adapter_updated=adapter_updated,
    )
