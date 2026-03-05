"""Training entry point — platform dispatch.

This module is invoked as a subprocess by the memory server:
    uv run python -m memory_server.training.train --agent-id avery

Runs the full pipeline: data prep → train. The training output (adapter_raw/adapters.safetensors)
is the final artifact — mlx-lm loads it directly per-request, no conversion needed.
"""

import argparse
import asyncio
import json
import logging
import platform
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("memory-server.training")


def _write_status(status_path: Path, status: str, **extra: object) -> None:
    """Write training status to a JSON file for the server to poll."""
    data = {"status": status, **extra}
    status_path.write_text(json.dumps(data))


async def run_training_pipeline(
    agent_id: str,
    server_url: str = "http://localhost:2090",
    max_iters: int = 200,
    lora_rank: int = 8,
    num_layers: int = 16,
    batch_size: int = 1,
) -> None:
    """Full training pipeline: data prep → train."""
    from ..config import settings

    agent_dir = settings.data_dir / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    status_path = agent_dir / "training_status.json"

    try:
        # Step 1: Data preparation
        _write_status(status_path, "preparing")
        logger.info(f"Step 1/2: Preparing training data for agent '{agent_id}'")

        from .data_prep import prepare_training_data
        data_path = await prepare_training_data(agent_id, server_url=server_url)

        # Count training pairs
        with open(data_path) as f:
            pair_count = sum(1 for _ in f)
        logger.info(f"  Generated {pair_count} training pairs")

        if pair_count < 10:
            _write_status(status_path, "failed", error="Too few training pairs generated")
            logger.error("Too few training pairs — need at least 10")
            return

        # Step 2: Train
        _write_status(status_path, "training")
        logger.info(f"Step 2/2: Training LoRA adapter (macOS/MLX)")

        from .train_mlx import run_training

        adapter_dir = agent_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        run_training(
            data_path=data_path,
            output_dir=adapter_dir,
            max_iters=max_iters,
            lora_rank=lora_rank,
            num_layers=num_layers,
            batch_size=batch_size,
        )

        # No conversion needed — mlx-lm loads adapters.safetensors directly
        _write_status(status_path, "complete")
        adapter_file = adapter_dir / "adapters.safetensors"
        logger.info(f"Training complete. Adapter at: {adapter_file}")

    except Exception as e:
        _write_status(status_path, "failed", error=str(e))
        logger.exception(f"Training failed for agent '{agent_id}'")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for an agent")
    parser.add_argument("--agent-id", required=True, help="Agent identifier")
    parser.add_argument("--server-url", default="http://localhost:2090", help="Memory server URL")
    parser.add_argument("--max-iters", type=int, default=200, help="Max training iterations")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to apply LoRA to (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    args = parser.parse_args()

    asyncio.run(run_training_pipeline(
        agent_id=args.agent_id,
        server_url=args.server_url,
        max_iters=args.max_iters,
        lora_rank=args.lora_rank,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()
