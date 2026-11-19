"""Mock realtime WebSocket server for testing.

Implements the llama.cpp /v1/realtime protocol with controllable responses.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any
from dataclasses import dataclass, field

try:
    import websockets
    import websockets.server
    from websockets.server import serve as ws_serve
except ImportError:
    websockets = None  # type: ignore
    ws_serve = None  # type: ignore


@dataclass
class MockToolCall:
    """A tool call the mock server should emit."""
    name: str
    call_id: str
    arguments: dict[str, Any]


@dataclass
class MockResponse:
    """A scripted response the mock server will emit."""
    text: str = ""
    audio_samples: int = 0  # number of PCM16 samples to generate (silence)
    tool_calls: list[MockToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    response_id: str = ""


class MockRealtimeServer:
    """Test double for llama.cpp /v1/realtime WebSocket endpoint."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self._server: Any = None
        self._responses: list[MockResponse] = []
        self._response_idx = 0
        self.received_events: list[dict] = []
        self.sessions_opened: int = 0
        self.session_updates: list[dict] = []
        self._ws: Any = None
        self._input_rate = 16000
        self._output_rate = 24000
        self._cancel_requested = False
        self._response_id_counter = 0

    def enqueue_response(self, resp: MockResponse) -> None:
        """Add a response to the queue."""
        if not resp.response_id:
            self._response_id_counter += 1
            resp.response_id = f"resp_{self._response_id_counter:04d}"
        self._responses.append(resp)

    def enqueue_responses(self, resps: list[MockResponse]) -> None:
        for r in resps:
            self.enqueue_response(r)

    async def start(self) -> int:
        """Start the server, returns the actual port."""
        assert ws_serve is not None, "websockets not installed"
        self._server = await ws_serve(
            self._handle, self.host, self.port,
        )
        # Get actual port
        for sock in self._server.sockets:
            addr = sock.getsockname()
            self.port = addr[1]
            break
        return self.port

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: Any) -> None:
        self._ws = ws
        self.sessions_opened += 1
        self._cancel_requested = False

        # Send session.created
        await ws.send(json.dumps({
            "type": "session.created",
            "session": {
                "id": f"sess_{self.sessions_opened:04d}",
                "audio": {
                    "input": {"format": {"rate": self._input_rate}},
                    "output": {"format": {"rate": self._output_rate}},
                },
            },
        }))

        try:
            async for raw in ws:
                event = json.loads(raw)
                self.received_events.append(event)
                await self._handle_event(event)
        except Exception:
            pass

    async def _handle_event(self, event: dict) -> None:
        etype = event.get("type", "")

        if etype == "session.update":
            self.session_updates.append(event)

        elif etype == "input_audio_buffer.append":
            pass  # Just accumulate

        elif etype == "input_audio_buffer.commit":
            pass  # Wait for response.create

        elif etype == "response.create":
            await self._send_next_response()

        elif etype == "response.cancel":
            self._cancel_requested = True

        elif etype == "conversation.item.create":
            item = event.get("item", {})
            if item.get("type") == "function_call_output":
                pass  # Tool result received

    async def _send_next_response(self) -> None:
        if self._response_idx >= len(self._responses):
            # No more scripted responses; send empty response
            rid = f"resp_empty_{self._response_idx}"
            await self._ws.send(json.dumps({
                "type": "response.created",
                "response": {"id": rid},
            }))
            await self._ws.send(json.dumps({
                "type": "response.done",
                "response": {"id": rid, "usage": None},
            }))
            return

        resp = self._responses[self._response_idx]
        self._response_idx += 1
        self._cancel_requested = False
        rid = resp.response_id

        # response.created
        await self._ws.send(json.dumps({
            "type": "response.created",
            "response": {"id": rid},
        }))

        # Send tool calls if any
        if resp.tool_calls:
            for tc in resp.tool_calls:
                if self._cancel_requested:
                    break
                await self._ws.send(json.dumps({
                    "type": "response.output_item.added",
                    "output_item": {
                        "type": "function_call",
                        "name": tc.name,
                        "call_id": tc.call_id,
                    },
                }))
                args_json = json.dumps(tc.arguments)
                await self._ws.send(json.dumps({
                    "type": "response.function_call_arguments.delta",
                    "call_id": tc.call_id,
                    "delta": args_json,
                }))
                await self._ws.send(json.dumps({
                    "type": "response.function_call_arguments.done",
                    "call_id": tc.call_id,
                    "arguments": args_json,
                    "name": tc.name,
                }))

        # Send text deltas
        if resp.text and not self._cancel_requested:
            # Send in chunks
            chunk_size = 50
            for i in range(0, len(resp.text), chunk_size):
                if self._cancel_requested:
                    break
                chunk = resp.text[i:i + chunk_size]
                await self._ws.send(json.dumps({
                    "type": "response.output_text.delta",
                    "delta": chunk,
                    "response_id": rid,
                }))
                await asyncio.sleep(0.001)

        # Send audio deltas
        if resp.audio_samples > 0 and not self._cancel_requested:
            # Generate silence audio in chunks
            chunk_samples = 2400  # 100ms at 24kHz
            remaining = resp.audio_samples
            while remaining > 0 and not self._cancel_requested:
                n = min(chunk_samples, remaining)
                audio_bytes = struct.pack(f"<{n}h", *([0] * n))
                b64 = base64.b64encode(audio_bytes).decode()
                await self._ws.send(json.dumps({
                    "type": "response.output_audio.delta",
                    "delta": b64,
                    "response_id": rid,
                }))
                remaining -= n
                await asyncio.sleep(0.001)

        # response.done
        done_event: dict[str, Any] = {
            "type": "response.done",
            "response": {"id": rid},
        }
        if resp.usage:
            done_event["response"]["usage"] = resp.usage
        await self._ws.send(json.dumps(done_event))

    def reset(self) -> None:
        """Reset state for next test."""
        self._responses.clear()
        self._response_idx = 0
        self.received_events.clear()
        self.session_updates.clear()
        self.sessions_opened = 0
        self._response_id_counter = 0


class MockTranscriptionServer:
    """Test double for /v1/realtime?intent=transcription."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self._server: Any = None
        self._transcript = "hello buddy"

    def set_transcript(self, text: str) -> None:
        self._transcript = text

    async def start(self) -> int:
        self._server = await ws_serve(self._handle, self.host, self.port)
        for sock in self._server.sockets:
            self.port = sock.getsockname()[1]
            break
        return self.port

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: Any) -> None:
        async for raw in ws:
            event = json.loads(raw)
            if event.get("type") == "input_audio_buffer.commit":
                if event.get("final"):
                    # Send transcription
                    words = self._transcript.split()
                    for i, w in enumerate(words):
                        await ws.send(json.dumps({
                            "type": "conversation.item.input_audio_transcription.delta",
                            "delta": w + (" " if i < len(words) - 1 else ""),
                        }))
                    await ws.send(json.dumps({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "transcript": self._transcript,
                    }))
