"""Convert trained LoRA adapters to GGUF format.

Converts MLX LoRA adapters directly to GGUF format using the gguf-py library.
This bypasses llama.cpp's convert_lora_to_gguf.py which doesn't yet support
Qwen3.5 MoE LoRA adapters (LoraTorchTensor missing reshape for V head reorder).

Also provides create_null_adapter() for generating a zero-effect GGUF adapter
so llama-server can start with --lora before any real training has happened.

Requires: llama.cpp checkout (for gguf-py library), torch, safetensors
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from ..config import settings

logger = logging.getLogger("memory-server.training.convert")

# Architecture string for Qwen 3.5 MoE in GGUF
GGUF_ARCH = "qwen35moe"


def create_null_adapter(output_path: Path) -> None:
    """Create a zero-effect GGUF LoRA adapter (metadata only, no tensors).

    llama-server requires --lora to point at an existing file at startup.
    This creates a valid GGUF adapter that loads successfully but has zero
    effect on inference — allowing the server to start before any real
    training has happened. Replace it with a real adapter after training.
    """
    # Add gguf-py to path
    gguf_py_path = settings.llama_cpp_dir / "gguf-py"
    if not gguf_py_path.exists():
        raise FileNotFoundError(
            f"gguf-py not found at {gguf_py_path}. "
            f"Set MEMORY_LLAMA_CPP_DIR to your llama.cpp directory."
        )
    sys.path.insert(0, str(gguf_py_path))

    import gguf

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(output_path), GGUF_ARCH, use_temp_file=False)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 1.0)

    writer.write_header_to_file(path=str(output_path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    logger.info(f"Null adapter written to {output_path} ({output_path.stat().st_size} bytes)")


def convert_to_gguf(
    adapter_dir: Path,
    output_path: Path,
    source_platform: str = "mlx",  # "mlx" or "hf"
) -> None:
    """Convert a trained LoRA adapter to GGUF format.

    Reads the MLX adapter (adapters.safetensors), maps tensor names to GGUF
    conventions using llama.cpp's TensorNameMap, and writes a GGUF LoRA file.
    """
    # Add gguf-py to path
    gguf_py_path = settings.llama_cpp_dir / "gguf-py"
    if not gguf_py_path.exists():
        raise FileNotFoundError(
            f"gguf-py not found at {gguf_py_path}. "
            f"Set MEMORY_LLAMA_CPP_DIR to your llama.cpp directory."
        )
    sys.path.insert(0, str(gguf_py_path))

    if source_platform == "mlx":
        _convert_mlx_to_gguf(adapter_dir, output_path)
    else:
        _convert_hf_to_gguf(adapter_dir, output_path)


def _convert_mlx_to_gguf(adapter_dir: Path, output_path: Path) -> None:
    """Convert MLX LoRA adapter directly to GGUF format.

    MLX tensor naming: language_model.model.layers.{n}.{module}.lora_a/lora_b
    MLX convention: lora_a is (input_dim, rank), lora_b is (rank, output_dim)
    GGUF convention: lora_a is (rank, input_dim), lora_b is (output_dim, rank)
    GGUF names: blk.{n}.{gguf_module}.weight.lora_a/lora_b
    """
    import gguf
    from gguf.tensor_mapping import TensorNameMap
    from gguf.constants import MODEL_ARCH

    try:
        from safetensors.numpy import load_file
    except ImportError:
        raise RuntimeError(
            "safetensors required for conversion. Install with: uv sync --extra convert"
        )

    # Find the MLX adapter file
    mlx_adapter = adapter_dir / "adapters.safetensors"
    if not mlx_adapter.exists():
        candidates = list(adapter_dir.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No safetensors adapter found in {adapter_dir}")
        mlx_adapter = candidates[0]

    logger.info(f"Converting MLX adapter to GGUF: {mlx_adapter}")

    # Load adapter config for lora_alpha
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        adapter_config = json.loads(config_path.read_text())
        lora_alpha = adapter_config.get("lora_parameters", {}).get("scale", 20.0)
    else:
        lora_alpha = float(settings.training_lora_rank * 2)

    # Load tensors as numpy arrays
    tensors = load_file(str(mlx_adapter))
    logger.info(f"Loaded {len(tensors)} tensors from adapter")

    # Set up tensor name mapping (40 layers for Qwen 3.5 35B A3B)
    tmap = TensorNameMap(MODEL_ARCH.QWEN35MOE, 40)

    # Create GGUF writer
    writer = gguf.GGUFWriter(str(output_path), GGUF_ARCH, use_temp_file=False)

    # Write metadata
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha)

    # Group tensors by base weight name (pair lora_a and lora_b)
    pairs: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in tensors.items():
        # Split: "language_model.model.layers.32.self_attn.q_proj.lora_a" →
        #   base="language_model.model.layers.32.self_attn.q_proj", suffix="lora_a"
        parts = key.rsplit(".", 1)
        base_name, suffix = parts[0], parts[1]
        if base_name not in pairs:
            pairs[base_name] = {}
        pairs[base_name][suffix] = tensor

    written = 0
    skipped = 0
    for base_name, ab_pair in sorted(pairs.items()):
        if "lora_a" not in ab_pair or "lora_b" not in ab_pair:
            logger.warning(f"Incomplete LoRA pair for {base_name}, skipping")
            skipped += 1
            continue

        lora_a_mlx = ab_pair["lora_a"]  # (input_dim, rank)
        lora_b_mlx = ab_pair["lora_b"]  # (rank, output_dim)

        # Transpose: MLX → GGUF convention
        lora_a = np.ascontiguousarray(lora_a_mlx.T)  # (rank, input_dim)
        lora_b = np.ascontiguousarray(lora_b_mlx.T)  # (output_dim, rank)

        # Map tensor name: strip "language_model." prefix, add ".weight" suffix
        hf_name = base_name.replace("language_model.", "") + ".weight"
        gguf_name = tmap.get_name(key=hf_name, try_suffixes=(".weight", ".bias"))

        if gguf_name is None:
            logger.warning(f"No GGUF mapping for {hf_name}, skipping")
            skipped += 1
            continue

        # Write lora_a and lora_b tensors
        # GGUF LoRA names: replace ".weight" with ".weight.lora_a" / ".weight.lora_b"
        name_a = gguf_name.replace(".weight", ".weight.lora_a")
        name_b = gguf_name.replace(".weight", ".weight.lora_b")

        writer.add_tensor(name_a, lora_a)
        writer.add_tensor(name_b, lora_b)
        written += 1

    logger.info(f"Writing GGUF: {written} tensor pairs, {skipped} skipped")

    writer.write_header_to_file(path=str(output_path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"GGUF adapter written to {output_path} ({size_mb:.1f} MB)")


def _convert_hf_to_gguf(adapter_dir: Path, output_path: Path) -> None:
    """Convert HuggingFace PEFT adapter to GGUF using llama.cpp's conversion script.

    Fallback for non-MLX adapters (Linux/unsloth).
    """
    import subprocess

    convert_script = settings.llama_cpp_dir / "convert_lora_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp convert script not found at {convert_script}. "
            f"Set MEMORY_LLAMA_CPP_DIR to your llama.cpp directory."
        )

    cmd = [
        sys.executable,
        str(convert_script),
        "--base-model-id", "Qwen/Qwen3.5-35B-A3B",
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
