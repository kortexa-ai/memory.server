"""LLM engine — talks to llama-server via OpenAI-compatible HTTP API.

The memory server does NOT load the model in-process. Instead it connects to
a llama-server instance (which we also own/control) that handles model loading,
GPU offload, KV cache, and LoRA adapter management.

This gives us:
- Latest llama.cpp features (qwen35moe, etc.) without waiting for Python bindings
- Proper continuous batching and KV cache management
- LoRA swapping via llama-server's API
- Separation of concerns: memory server = brain, llama-server = muscle
"""

import logging
from pathlib import Path

import httpx

from ..config import settings

logger = logging.getLogger("memory-server.engine")

# Singleton engine instance
_engine: "MemoryEngine | None" = None


class MemoryEngine:
    """LLM engine that connects to llama-server."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._active_agent: str | None = None
        self._healthy: bool = False

    async def connect(self) -> None:
        """Connect to llama-server and verify it's running."""
        self._client = httpx.AsyncClient(
            base_url=settings.engine_server_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            health = resp.json()
            self._healthy = True
            logger.info(f"Connected to llama-server at {settings.engine_server_url}")
            logger.info(f"  Server health: {health}")
        except httpx.ConnectError:
            logger.warning(
                f"llama-server not reachable at {settings.engine_server_url}. "
                f"Start it with: llama-server -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL --port {settings.engine_server_port}"
            )
            # Don't fail — server might come up later
            self._healthy = False

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_adapter_path(self, agent_id: str) -> Path | None:
        """Get LoRA adapter path for an agent, or None if it doesn't exist."""
        adapter_path = settings.data_dir / agent_id / "adapter.gguf"
        return adapter_path if adapter_path.exists() else None

    # --- LoRA adapter management (Phase 3) ---
    # llama-server supports LoRA via --lora flag at startup and
    # via the /lora-adapters endpoint for dynamic swapping.
    # For now, LoRA swapping is a TODO — the server needs to be started
    # with --lora pointing to the adapter file, or we use the API.

    async def swap_adapter(self, agent_id: str | None) -> None:
        """Swap to a different agent's LoRA adapter via llama-server API.

        llama-server exposes POST /lora-adapters for dynamic LoRA management.
        """
        if agent_id == self._active_agent:
            return

        if not self._client:
            return

        adapter_path = self.get_adapter_path(agent_id) if agent_id else None

        if adapter_path is None:
            # Clear any active adapter
            if self._active_agent is not None:
                try:
                    await self._client.post("/lora-adapters", json=[])
                    self._active_agent = None
                    logger.info("Cleared LoRA adapter (base model)")
                except httpx.HTTPError as e:
                    logger.warning(f"Failed to clear LoRA adapter: {e}")
            return

        # Set the adapter
        try:
            await self._client.post(
                "/lora-adapters",
                json=[{"path": str(adapter_path), "scale": 1.0}],
            )
            self._active_agent = agent_id
            logger.info(f"Activated LoRA adapter for agent '{agent_id}'")
        except httpx.HTTPError as e:
            logger.warning(f"Failed to set LoRA adapter for agent '{agent_id}': {e}")

    # --- Inference ---

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.3,
        agent_id: str | None = None,
    ) -> str:
        """Run chat completion via llama-server's OpenAI-compatible API."""
        if not self._client:
            raise RuntimeError("Not connected to llama-server — call connect() first")

        # Swap adapter if needed
        await self.swap_adapter(agent_id)

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    @property
    def loaded(self) -> bool:
        return self._healthy

    @property
    def active_agent(self) -> str | None:
        return self._active_agent


def get_engine() -> MemoryEngine:
    """Get the singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine
