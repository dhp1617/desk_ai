"""WebSocket client for llama.cpp /v1/realtime (OpenAI-realtime-compatible)."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Accumulated tool/function call from a response."""
    name: str = ""
    call_id: str = ""
    arguments_json: str = ""

    @property
    def arguments(self) -> dict[str, Any]:
        try:
            return json.loads(self.arguments_json)
        except (json.JSONDecodeError, TypeError):
            return {}


@dataclass
class ResponseCollector:
    """Accumulates deltas for a single response."""
    response_id: str = ""
    text_parts: list[str] = field(default_factory=list)
    audio_parts: list[bytes] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_done: bool = False
    was_cancelled: bool = False
    usage: dict[str, int] | None = None
    _pending_calls: dict[str, ToolCall] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts)

    @property
    def full_audio(self) -> bytes:
        return b"".join(self.audio_parts)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class RealtimeSession:
    """WebSocket client for llama.cpp /v1/realtime endpoint."""

    def __init__(self, url: str, debug: bool = False):
        self._url = url
        self._debug = debug
        self._ws: Any = None
        self._input_rate: int = 16000
        self._output_rate: int = 24000
        self._connected = False
        self._session_id: str = ""
        self.active_response_id: str | None = None
        self._cancelled_ids: set[str] = set()

    @property
    def input_rate(self) -> int:
        return self._input_rate

    @property
    def output_rate(self) -> int:
        return self._output_rate

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Open WebSocket and wait for session.created."""
        import websockets

        self._ws = await websockets.connect(self._url)
        # Wait for session.created
        raw = await self._ws.recv()
        event = json.loads(raw)
        if self._debug:
            logger.debug("RECV: %s", json.dumps(event, indent=2)[:500])

        if event.get("type") != "session.created":
            err = event.get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Realtime session creation failed: {msg or event.get('type')}")

        session = event.get("session", {})
        self._session_id = session.get("id", "")
        audio = session.get("audio", {})
        inp = audio.get("input", {}).get("format", {})
        out = audio.get("output", {}).get("format", {})
        if "rate" in inp:
            self._input_rate = inp["rate"]
        if "rate" in out:
            self._output_rate = out["rate"]

        self._connected = True
        logger.info("Realtime session connected: %s (in=%dHz, out=%dHz)",
                     self._session_id, self._input_rate, self._output_rate)

    async def close(self) -> None:
        """Close the WebSocket."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send(self, event: dict) -> None:
        if not self._ws:
            raise RuntimeError("Not connected")
        raw = json.dumps(event)
        if self._debug:
            logger.debug("SEND: %s", raw[:500])
        await self._ws.send(raw)

    async def send_session_update(
        self,
        instructions: str,
        tools: list[dict],
        output_modalities: list[str] | None = None,
        max_output_tokens: int = 4096,
        input_audio_rate: int = 16000,
    ) -> None:
        """Send session.update. Must be called BEFORE first audio/message."""
        if output_modalities is None:
            output_modalities = ["text", "audio"]
        event: dict[str, Any] = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
                "tools": tools,
                "tool_choice": "auto",
                "output_modalities": output_modalities,
                "max_output_tokens": max_output_tokens,
                "audio": {
                    "input": {
                        "format": {"rate": input_audio_rate},
                    },
                },
            },
        }
        await self._send(event)

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send PCM16 mono audio chunk."""
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": b64,
        })

    async def send_commit(self) -> None:
        """Commit the audio buffer."""
        await self._send({"type": "input_audio_buffer.commit"})

    async def send_response_create(self) -> None:
        """Request a response from the model."""
        await self._send({"type": "response.create"})

    async def send_response_cancel(self) -> None:
        """Cancel the current response."""
        if self.active_response_id:
            self._cancelled_ids.add(self.active_response_id)
        await self._send({"type": "response.cancel"})

    async def send_text_message(self, text: str) -> None:
        """Send a text user message."""
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        })

    async def send_tool_result(self, call_id: str, output: str) -> None:
        """Send function call output."""
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })

    async def recv(self) -> dict:
        """Receive and parse next event."""
        if not self._ws:
            raise RuntimeError("Not connected")
        raw = await self._ws.recv()
        event = json.loads(raw)
        if self._debug:
            # Truncate audio data in debug logs
            log_event = dict(event)
            if "delta" in log_event and event.get("type", "").endswith("audio.delta"):
                log_event["delta"] = f"<{len(event['delta'])} chars base64>"
            logger.debug("RECV: %s", json.dumps(log_event, indent=2)[:500])
        return event

    async def recv_iter(self) -> AsyncIterator[dict]:
        """Async iterator over incoming events."""
        while self._connected:
            try:
                event = await self.recv()
                yield event
            except Exception:
                self._connected = False
                break

    def is_cancelled(self, response_id: str) -> bool:
        """Check if a response was cancelled."""
        return response_id in self._cancelled_ids

    def process_event(self, event: dict, collector: ResponseCollector) -> None:
        """Process an event into the response collector."""
        etype = event.get("type", "")
        resp_id = event.get("response_id", "")
        resp = event.get("response", {})
        if isinstance(resp, dict):
            resp_id = resp_id or resp.get("id", "")

        # Skip events from cancelled responses
        if resp_id and self.is_cancelled(resp_id):
            collector.was_cancelled = True
            return

        if etype == "response.created":
            collector.response_id = resp.get("id", "")
            self.active_response_id = collector.response_id

        elif etype == "response.output_text.delta":
            collector.text_parts.append(event.get("delta", ""))

        elif etype == "response.output_audio.delta":
            b64 = event.get("delta", "")
            if b64:
                try:
                    collector.audio_parts.append(base64.b64decode(b64))
                except Exception:
                    pass

        elif etype == "response.output_item.added":
            item = event.get("output_item", {})
            if item.get("type") == "function_call":
                tc = ToolCall(
                    name=item.get("name", ""),
                    call_id=item.get("call_id", ""),
                )
                collector._pending_calls[tc.call_id] = tc

        elif etype == "response.function_call_arguments.delta":
            call_id = event.get("call_id", "")
            if call_id in collector._pending_calls:
                collector._pending_calls[call_id].arguments_json += event.get("delta", "")

        elif etype == "response.function_call_arguments.done":
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            args = event.get("arguments", "")
            if call_id in collector._pending_calls:
                tc = collector._pending_calls.pop(call_id)
                tc.arguments_json = args
                tc.name = name or tc.name
                collector.tool_calls.append(tc)

        elif etype == "response.done":
            collector.is_done = True
            self.active_response_id = None
            usage = resp.get("usage")
            if usage:
                collector.usage = usage

        elif etype == "error":
            logger.error("Server error: %s", event.get("error", event))
