"""Unsloth QLoRA training backend (Linux/NVIDIA CUDA).

Uses unsloth for memory-efficient QLoRA fine-tuning on Qwen 3.5 35B A3B.
Outputs adapter in HuggingFace PEFT format (safetensors) — needs conversion to GGUF after.

Requires: pip install unsloth torch transformers peft trl (or uv sync --extra train-linux)
"""

import logging
from pathlib import Path

from ..config import settings

logger = logging.getLogger("memory-server.training.unsloth")


def run_training(
    data_path: Path,
    output_dir: Path,
    max_iters: int = 200,
    lora_rank: int = 8,
    batch_size: int = 1,
) -> None:
    """Run QLoRA training using unsloth.

    This is the Linux/NVIDIA path. Imports torch and unsloth at runtime
    (not loaded into the memory server's main process).
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError(
            "unsloth not installed. Install with: uv sync --extra train-linux"
        )

    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Model — use original weights (not GGUF)
    model_name = settings.engine_hf_repo.replace("-GGUF", "")

    logger.info(f"Starting unsloth QLoRA training:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Iterations: {max_iters}, Rank: {lora_rank}, Batch: {batch_size}")

    # Load model with QLoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    # Load training data
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            max_steps=max_iters,
            learning_rate=2e-4,
            warmup_steps=10,
            logging_steps=10,
            save_strategy="no",  # we save manually at the end
            fp16=True,
        ),
    )

    trainer.train()

    # Save the adapter in HuggingFace PEFT format
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info(f"Unsloth training complete. Adapter saved to {output_dir}")
