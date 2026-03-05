"""LLM engine — launches and talks to mlx-lm server for inference.

The memory server owns the mlx-lm server process (launched as a subprocess).
mlx-lm provides:
- OpenAI-compatible chat completions API
- Per-request LoRA adapter loading via "adapters" field
- No restart needed after training — new adapter picked up immediately
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import httpx

from ..config import settings

logger = logging.getLogger("memory-server.engine")

# Singleton engine instance
_engine: "MemoryEngine | None" = None


class MemoryEngine:
    """LLM engine that launches and connects to mlx-lm server."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._process: subprocess.Popen | None = None
        self._healthy: bool = False

    @property
    def _base_url(self) -> str:
        return f"http://localhost:{settings.model_server_port}"

    async def connect(self) -> None:
        """Launch mlx-lm server as subprocess and wait for it to be ready."""
        # Start the mlx-lm server
        cmd = [
            sys.executable, "-m", "mlx_lm.server",
            "--model", settings.model_repo,
            "--host", settings.host,
            "--port", str(settings.model_server_port),
        ]

        logger.info(f"Launching mlx-lm server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

        # Wait for server to become healthy (model loading takes a while)
        for attempt in range(120):  # up to 2 minutes
            # Check if process died
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logger.error(f"mlx-lm server exited with code {self._process.returncode}: {stderr}")
                self._healthy = False
                return

            try:
                resp = await self._client.get("/v1/models")
                resp.raise_for_status()
                self._healthy = True
                logger.info(f"mlx-lm server ready on port {settings.model_server_port}")
                return
            except (httpx.ConnectError, httpx.HTTPStatusError):
                await asyncio.sleep(1)

        logger.warning("mlx-lm server did not become ready within 2 minutes")
        self._healthy = False

    async def disconnect(self) -> None:
        """Shut down the mlx-lm server and close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._process and self._process.poll() is None:
            logger.info("Shutting down mlx-lm server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            logger.info("mlx-lm server stopped")
        self._process = None

    def get_adapter_path(self, agent_id: str) -> str | None:
        """Get LoRA adapter directory for an agent, or None if not trained.

        Returns the path to the adapter_raw directory containing adapters.safetensors,
        which mlx-lm can load directly per-request.
        """
        adapter_dir = settings.data_dir / agent_id / "adapter"
        adapter_file = adapter_dir / "adapters.safetensors"
        return str(adapter_dir) if adapter_file.exists() else None

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 0.3,
        agent_id: str | None = None,
    ) -> str:
        """Run chat completion via mlx-lm server's OpenAI-compatible API."""
        if not self._client:
            raise RuntimeError("Not connected to mlx-lm server — call connect() first")

        # Disable Qwen 3.5 thinking mode by appending /no_think to the last user message
        # (mlx-lm server doesn't support chat_template_kwargs)
        messages = [m.copy() for m in messages]
        for m in reversed(messages):
            if m["role"] == "user":
                m["content"] += " /no_think"
                break

        body: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Load agent's LoRA adapter if available
        if agent_id:
            adapter_path = self.get_adapter_path(agent_id)
            if adapter_path:
                body["adapters"] = adapter_path

        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # mlx-lm may return thinking in "reasoning" field
        return msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""

    @property
    def loaded(self) -> bool:
        return self._healthy

    @property
    def active_agent(self) -> str | None:
        # No longer tracking active agent — adapters loaded per-request
        return None


def get_engine() -> MemoryEngine:
    """Get the singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine
