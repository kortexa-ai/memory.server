"""MLX LoRA training backend (macOS Apple Silicon).

Uses mlx-lm for QLoRA fine-tuning on Qwen 3.5 35B A3B.
Outputs adapter in MLX format (adapters.safetensors) — needs conversion to GGUF after.

Requires: pip install mlx-lm (or uv sync --extra train-mac)
"""

import logging
import subprocess
import sys
from pathlib import Path

from ..config import settings

logger = logging.getLogger("memory-server.training.mlx")


def run_training(
    data_path: Path,
    output_dir: Path,
    max_iters: int = 200,
    lora_rank: int = 8,
    batch_size: int = 1,
) -> None:
    """Run QLoRA training using mlx-lm.

    mlx-lm expects training data as a directory containing train.jsonl (and optionally
    valid.jsonl). Each line is: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    # mlx-lm wants a directory with train.jsonl
    train_dir = output_dir / "data"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Symlink or copy training data as train.jsonl
    train_jsonl = train_dir / "train.jsonl"
    if train_jsonl.exists():
        train_jsonl.unlink()
    train_jsonl.symlink_to(data_path.resolve())

    # Model to fine-tune — original weights (not GGUF) for MLX training
    model_name = settings.training_model_repo

    logger.info(f"Starting MLX LoRA training:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Iterations: {max_iters}, Rank: {lora_rank}, Batch: {batch_size}")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", str(train_dir),
        "--adapter-path", str(output_dir),
        "--batch-size", str(batch_size),
        "--num-layers", str(lora_rank),
        "--iters", str(max_iters),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"MLX training failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"MLX training failed with exit code {result.returncode}: {result.stderr}")

    logger.info(f"MLX training complete. Adapter saved to {output_dir}")

    # Verify output exists
    adapter_file = output_dir / "adapters.safetensors"
    if not adapter_file.exists():
        # mlx-lm might use a different output name
        safetensors = list(output_dir.glob("*.safetensors"))
        if not safetensors:
            raise RuntimeError(f"No adapter output found in {output_dir}")
        logger.info(f"Adapter file: {safetensors[0]}")
