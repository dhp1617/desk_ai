"""llama-server process lifecycle manager and resilient reconnection."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from typing import Any, Callable, Coroutine

import httpx

from buddy.client.realtime import RealtimeSession
from buddy.config import ServerConfig

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages the llama-server subprocess."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._proc: subprocess.Popen[bytes] | None = None

    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    async def is_healthy(self) -> bool:
        """Probe /health endpoint."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.config.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def start(self) -> None:
        """Start llama-server with exact command arguments."""
        if self.is_running():
            logger.info("llama-server is already running (PID %d)", self._proc.pid if self._proc else 0)
            return

        cmd = [
            "llama-server",
            "-hf", self.config.model_hf,
            "--reasoning", "off",
            "-ngl", str(self.config.ngl),
            "-c", str(self.config.n_ctx),
            "--port", str(self.config.port),
            "--host", self.config.host,
        ]
        cmd.extend(self.config.extra_args)

        logger.info("Starting llama-server: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "llama-server executable not found on PATH. "
                "Ensure llama.cpp b11960+ is built and installed."
            )

    async def wait_healthy(self, timeout: float | None = None) -> bool:
        """Poll /health until healthy or timeout expires."""
        timeout = timeout or self.config.health_timeout
        deadline = time.monotonic() + timeout
        delay = 0.5

        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                rc = self._proc.returncode
                stdout_tail = ""
                if self._proc.stdout:
                    try:
                        stdout_tail = self._proc.stdout.read().decode("utf-8", errors="replace")[-500:]
                    except Exception:
                        pass
                raise RuntimeError(f"llama-server exited prematurely with code {rc}. Output: {stdout_tail}")

            if await self.is_healthy():
                logger.info("llama-server is healthy at %s", self.config.base_url)
                return True

            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)

        return False

    async def ensure_running(self) -> None:
        """Ensure server is up; start and wait if needed."""
        if await self.is_healthy():
            return
        if self.config.autostart:
            await self.start()
            if not await self.wait_healthy():
                raise TimeoutError(f"llama-server failed to become healthy within {self.config.health_timeout}s")
        else:
            raise ConnectionError(f"llama-server not reachable at {self.config.base_url} and autostart is disabled")

    async def stop(self) -> None:
        """Gracefully terminate server process."""
        if not self._proc:
            return
        proc = self._proc
        self._proc = None

        if proc.poll() is not None:
            return

        try:
            proc.terminate()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, proc.wait, 5.0)
        except (subprocess.TimeoutExpired, Exception):
            try:
                proc.kill()
            except Exception:
                pass


class ReconnectingSession:
    """Wraps RealtimeSession with automatic exponential backoff reconnects."""

    def __init__(
        self,
        url: str,
        setup_fn: Callable[[RealtimeSession], Coroutine[Any, Any, None]] | None = None,
        debug: bool = False,
        max_delay: float = 30.0,
    ):
        self.url = url
        self.setup_fn = setup_fn
        self.debug = debug
        self.max_delay = max_delay
        self.session: RealtimeSession | None = None
        self._closed = False
        self.reconnect_count = 0

    async def connect(self) -> RealtimeSession:
        self._closed = False
        delay = 1.0
        while not self._closed:
            try:
                sess = RealtimeSession(self.url, debug=self.debug)
                await sess.connect()
                if self.setup_fn:
                    await self.setup_fn(sess)
                self.session = sess
                self.reconnect_count += 1
                return sess
            except Exception as e:
                logger.warning("Session connect error (%s), retrying in %.1fs...", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, self.max_delay)

        raise RuntimeError("ReconnectingSession was closed during connect")

    async def close(self) -> None:
        self._closed = True
        if self.session:
            await self.session.close()
            self.session = None
