"""Configuration for the memory server."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Memory server settings. Override via environment variables or .env file."""

    # Server
    host: str = "0.0.0.0"
    port: int = 2090

    # LLM engine — connects to llama-server via HTTP
    engine_server_url: str = "http://localhost:2027"
    engine_server_port: int = 2027  # for display in help messages only

    # Storage
    data_dir: Path = Path("data/agents")

    # Memory limits
    max_memories_per_agent: int = 10000
    recall_default_limit: int = 10
    recall_max_tokens: int = 4000

    # Ingest settings
    chunk_size: int = 1000  # chars per chunk when splitting transcripts
    chunk_overlap: int = 200

    # Embedding model (Phase 2)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64

    # Training settings (Phase 3)
    training_model_repo: str = "NexVeridian/Qwen3.5-35B-A3B-4bit"  # 4-bit text-only MLX model (~19.5GB, converted with mlx-lm)
    training_min_memories: int = 100  # Minimum memories before training is allowed
    training_max_iters: int = 200
    training_lora_rank: int = 8
    training_batch_size: int = 1
    training_qa_pairs_per_memory: int = 5
    llama_cpp_dir: Path = Path.home() / "src/llama.cpp"

    model_config = {"env_prefix": "MEMORY_", "env_file": ".env"}


settings = Settings()
