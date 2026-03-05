"""Training pipeline for per-agent LoRA adapters.

This package is imported by the memory server for launching training as a subprocess.
Training uses mlx-lm (included in core deps). The actual training is run in a subprocess
to avoid loading training-specific state into the server process.
"""
