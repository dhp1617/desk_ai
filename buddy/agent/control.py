"""Unix domain control socket server for CLI and external management."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from buddy.agent.state import BuddyAgent

logger = logging.getLogger(__name__)


class ControlServer:
    """Listens on a Unix socket for JSON-delimited control commands."""

    def __init__(self, socket_path: Path, agent: BuddyAgent):
        self.socket_path = socket_path
        self.agent = agent
        self._server: asyncio.AbstractServer | None = None
        self._running = False

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        self._running = True
        logger.info("Control server listening on %s", self.socket_path)

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        logger.info("Control server stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line_bytes = await reader.readline()
            if not line_bytes:
                return
            line = line_bytes.decode("utf-8").strip()
            response = await self.handle_command(line)
            out = json.dumps(response) + "\n"
            writer.write(out.encode("utf-8"))
            await writer.drain()
        except Exception as e:
            logger.error("Error servicing control socket client: %s", e)
            try:
                err = json.dumps({"status": "error", "message": str(e)}) + "\n"
                writer.write(err.encode("utf-8"))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def handle_command(self, cmd_line: str) -> dict[str, Any]:
        parts = cmd_line.split(maxsplit=1)
        if not parts:
            return {"status": "error", "message": "Empty command"}

        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "status":
            return {
                "status": "ok",
                "state": self.agent.state.name,
                "mode": "always" if self.agent.is_always_mode else "wake",
                "muted": self.agent.is_muted,
                "audio_level": self.agent.audio_level,
            }
        elif cmd == "toggle":
            mode = await self.agent.toggle_always()
            return {"status": "ok", "mode": mode}
        elif cmd == "always":
            await self.agent.set_always_mode(True)
            return {"status": "ok", "mode": "always"}
        elif cmd == "wake":
            await self.agent.set_always_mode(False)
            return {"status": "ok", "mode": "wake"}
        elif cmd == "mute":
            await self.agent.set_muted(True)
            return {"status": "ok", "muted": True}
        elif cmd == "unmute":
            await self.agent.set_muted(False)
            return {"status": "ok", "muted": False}
        elif cmd == "say":
            if not arg:
                return {"status": "error", "message": "Missing text for 'say'"}
            ok = await self.agent.speak_notice(arg)
            return {"status": "ok" if ok else "busy", "spoken": ok}
        elif cmd == "quit":
            asyncio.create_task(self.agent.shutdown())
            return {"status": "ok", "message": "Shutting down"}
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}


async def send_control_command(socket_path: Path, cmd: str) -> dict[str, Any]:
    """Client utility for sending command over Unix socket."""
    if not socket_path.exists():
        raise FileNotFoundError(f"Control socket not found at {socket_path}. Is Buddy running?")

    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        writer.write((cmd.strip() + "\n").encode("utf-8"))
        await writer.drain()
        line = await reader.readline()
        if not line:
            return {"status": "error", "message": "No response"}
        return json.loads(line.decode("utf-8"))
    finally:
        writer.close()
        await writer.wait_closed()
