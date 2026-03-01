"""Convert trained LoRA adapters to GGUF format.

Handles the adapter format differences between training backends:
- MLX outputs adapters.safetensors with MLX naming conventions
- Unsloth/HF PEFT outputs adapter_model.safetensors with HF naming conventions

Both need conversion via llama.cpp's convert_lora_to_gguf.py to produce adapter.gguf
that llama-server can load at runtime.
"""

import logging
import subprocess
import sys
from pathlib import Path

from ..config import settings

logger = logging.getLogger("memory-server.training.convert")


def convert_to_gguf(
    adapter_dir: Path,
    output_path: Path,
    source_platform: str = "mlx",  # "mlx" or "hf"
) -> None:
    """Convert a trained LoRA adapter to GGUF format.

    Args:
        adapter_dir: Directory containing the trained adapter files
        output_path: Where to write the adapter.gguf file
        source_platform: "mlx" for MLX adapters, "hf" for HuggingFace PEFT adapters
    """
    if source_platform == "mlx":
        _convert_mlx_to_hf(adapter_dir)

    _convert_hf_to_gguf(adapter_dir, output_path)


def _convert_mlx_to_hf(adapter_dir: Path) -> None:
    """Convert MLX adapter format to HuggingFace PEFT format.

    MLX LoRA adapters use different tensor naming conventions than HF PEFT:
    - MLX: layers.{n}.self_attn.q_proj.lora_a, etc.
    - HF:  base_model.model.model.layers.{n}.self_attn.q_proj.lora_A.weight, etc.

    MLX also transposes the B matrix relative to HF convention.
    """
    try:
        import safetensors.torch
        import torch
    except ImportError:
        raise RuntimeError(
            "torch and safetensors required for MLX→HF conversion. "
            "Install with: uv sync --extra convert"
        )

    # Find the MLX adapter file
    mlx_adapter = adapter_dir / "adapters.safetensors"
    if not mlx_adapter.exists():
        candidates = list(adapter_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No safetensors adapter found in {adapter_dir}")
        mlx_adapter = candidates[0]

    logger.info(f"Converting MLX adapter: {mlx_adapter}")

    tensors = safetensors.torch.load_file(str(mlx_adapter))
    converted: dict[str, object] = {}

    for key, tensor in tensors.items():
        # MLX key format: layers.{n}.{module}.lora_a / lora_b
        # HF key format: base_model.model.model.layers.{n}.{module}.lora_A.weight / lora_B.weight
        hf_key = key
        hf_key = hf_key.replace("lora_a", "lora_A.weight")
        hf_key = hf_key.replace("lora_b", "lora_B.weight")
        hf_key = f"base_model.model.model.{hf_key}"

        # MLX stores B transposed relative to HF convention
        if "lora_B" in hf_key:
            tensor = tensor.T

        converted[hf_key] = tensor

    # Write in HF format
    hf_adapter_path = adapter_dir / "adapter_model.safetensors"
    safetensors.torch.save_file(converted, str(hf_adapter_path))
    logger.info(f"Wrote HF-format adapter: {hf_adapter_path}")

    # Write minimal adapter_config.json for convert_lora_to_gguf.py
    import json
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": settings.engine_hf_repo.replace("-GGUF", ""),
        "r": settings.training_lora_rank,
        "lora_alpha": settings.training_lora_rank * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=2))


def _convert_hf_to_gguf(adapter_dir: Path, output_path: Path) -> None:
    """Convert HuggingFace PEFT adapter to GGUF using llama.cpp's conversion script."""
    convert_script = settings.llama_cpp_dir / "convert_lora_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found at {convert_script}. "
            f"Set MEMORY_LLAMA_CPP_DIR to your llama.cpp directory."
        )

    base_model = settings.engine_hf_repo.replace("-GGUF", "")

    cmd = [
        sys.executable,
        str(convert_script),
        "--base", base_model,
        "--outfile", str(output_path),
        str(adapter_dir),
    ]

    logger.info(f"Converting to GGUF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"GGUF conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

    if not output_path.exists():
        raise RuntimeError(f"GGUF conversion produced no output at {output_path}")

    logger.info(f"GGUF adapter written to {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
