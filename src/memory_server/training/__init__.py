"""Training pipeline for per-agent LoRA adapters.

This package is imported by the memory server for launching training as a subprocess,
but the actual training dependencies (mlx-lm, unsloth, torch) are optional and only
imported inside the training scripts themselves.
"""
