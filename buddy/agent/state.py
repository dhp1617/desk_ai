"""Buddy agent state machine and real-time audio orchestration."""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from buddy.audio.capture import AudioCapture
from buddy.audio.player import AudioPlayer
from buddy.audio.vad import VoiceActivityDetector
from buddy.client.realtime import RealtimeSession, ResponseCollector
from buddy.config import Config
from buddy.memory.store import MemoryStore
from buddy.tools.registry import ToolRegistry
from buddy.wake.base import WakeDetector

if TYPE_CHECKING:
    from buddy.agent.context import ContextManager

logger = logging.getLogger(__name__)


class AgentState(enum.Enum):
    IDLE = "IDLE"
    WAKE_LISTEN = "WAKE_LISTEN"
    ALWAYS_LISTEN = "ALWAYS_LISTEN"
    CAPTURE = "CAPTURE"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"
    MUTED = "MUTED"
    ERROR = "ERROR"


class BuddyAgent:
    """Core state machine driving voice interaction and model streaming."""

    def __init__(
        self,
        config: Config,
        session: RealtimeSession,
        capture: AudioCapture,
        player: AudioPlayer,
        wake: WakeDetector,
        vad: VoiceActivityDetector,
        tools: ToolRegistry,
        memory: MemoryStore,
        context: ContextManager,
        on_state_change: Callable[[AgentState, float], None] | None = None,
    ):
        self.config = config
        self.session = session
        self.capture = capture
        self.player = player
        self.wake = wake
        self.vad = vad
        self.tools = tools
        self.memory = memory
        self.context = context
        self.on_state_change = on_state_change

        self._state = AgentState.WAKE_LISTEN
        self._audio_level: float = 0.0
        self._always_mode: bool = False
        self._muted: bool = False
        self._state_before_mute: AgentState = AgentState.WAKE_LISTEN

        self._last_active_time: float = time.monotonic()
        self._running: bool = True
        self._shutdown_event = asyncio.Event()

        # Barge-in and response cancellation tracking
        self._active_collector: ResponseCollector | None = None
        self._last_user_transcript: str = ""

        # Concurrency locks
        self._state_lock = asyncio.Lock()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def audio_level(self) -> float:
        return self._audio_level

    @property
    def is_always_mode(self) -> bool:
        return self._always_mode

    @property
    def is_muted(self) -> bool:
        return self._muted

    def is_idle(self) -> bool:
        return self._state in (AgentState.IDLE, AgentState.WAKE_LISTEN, AgentState.ALWAYS_LISTEN, AgentState.FOLLOW_UP)

    STATE_BANNERS = {
        AgentState.WAKE_LISTEN: "\n" + "═" * 58 + "\n[😴 SLEEPING / WAKE MODE] Say 'Buddy' to wake me up.\n" + "═" * 58,
        AgentState.ALWAYS_LISTEN: "\n" + "═" * 58 + "\n[⚡ ALWAYS LISTENING] Mic open. Speak anytime!\n" + "═" * 58,
        AgentState.CAPTURE: "\n" + "─" * 58 + "\n[👂 LISTENING] I hear you! Speak your question now...\n" + "─" * 58,
        AgentState.THINKING: "[🧠 THINKING] Got your query. Thinking...",
        AgentState.SPEAKING: "[🗣️ ASSISTANT] Speaking (Say 'Buddy' to interrupt):",
        AgentState.FOLLOW_UP: "\n" + "─" * 58 + "\n[💬 FOLLOW-UP] 8s window: ask another question or stay silent to sleep...\n" + "─" * 58,
        AgentState.MUTED: "\n[🔇 MUTED] Assistant paused.\n",
        AgentState.ERROR: "\n[⚠️ ERROR] Recovering...\n",
    }

    def _set_state(self, new_state: AgentState, level: float = 0.0) -> None:
        if self._state != new_state:
            banner = self.STATE_BANNERS.get(new_state)
            if banner:
                print(banner, flush=True)
        self._state = new_state
        self._audio_level = level
        logger.debug("State transition -> %s (level=%.2f)", new_state.value, level)
        if self.on_state_change:
            try:
                self.on_state_change(new_state, level)
            except Exception:
                pass

    async def set_muted(self, muted: bool) -> None:
        async with self._state_lock:
            if self._muted == muted:
                return
            self._muted = muted
            if muted:
                self._state_before_mute = self._state
                self.player.clear()
                self._set_state(AgentState.MUTED)
            else:
                target = AgentState.ALWAYS_LISTEN if self._always_mode else AgentState.WAKE_LISTEN
                self._set_state(target)

    async def toggle_always(self) -> str:
        return await self.set_always_mode(not self._always_mode)

    async def set_always_mode(self, enabled: bool) -> str:
        async with self._state_lock:
            self._always_mode = enabled
            if not self._muted:
                new_state = AgentState.ALWAYS_LISTEN if enabled else AgentState.WAKE_LISTEN
                self._set_state(new_state)
            self._last_active_time = time.monotonic()
            return "always" if enabled else "wake"

    async def shutdown(self) -> None:
        self._running = False
        self._shutdown_event.set()

    async def initialize_session(self) -> None:
        """Send session.update with current instructions and tools before any audio."""
        instructions = self.context.build_instructions()
        tool_schemas = self.tools.get_all_tools()
        await self.session.send_session_update(
            instructions=instructions,
            tools=tool_schemas,
            output_modalities=["text", "audio"],
            max_output_tokens=self.config.agent.max_output_tokens,
            input_audio_rate=self.config.audio.sample_rate,
        )
        logger.info("Session initialized with %d tools", len(tool_schemas))

    async def run(self) -> None:
        """Main agent loop driving speech interactions."""
        self._running = True
        logger.info("Buddy agent loop starting in %s mode", "always" if self._always_mode else "wake")

        # Initialize realtime session parameters
        try:
            await self.initialize_session()
        except Exception as e:
            logger.error("Failed to initialize realtime session: %s", e)
            self._set_state(AgentState.ERROR)
            return

        while self._running:
            if self._shutdown_event.is_set():
                break

            if self._muted:
                await asyncio.sleep(0.1)
                continue

            # Fallback to wake mode after 10 min idle if in always-mode
            if self._always_mode and self._state == AgentState.ALWAYS_LISTEN:
                if (time.monotonic() - self._last_active_time) > self.config.agent.always_timeout_s:
                    logger.info("Always-listening mode timed out after 10m idle; reverting to wake mode")
                    self._always_mode = False
                    self._set_state(AgentState.WAKE_LISTEN)

            # Check context rollover requirement when idle
            if self.is_idle() and self.context.needs_rollover():
                await self._perform_context_rollover()

            # Listen loop based on state
            if self._state in (AgentState.WAKE_LISTEN, AgentState.ALWAYS_LISTEN, AgentState.FOLLOW_UP):
                triggered, pre_roll = await self._listen_for_trigger()
                if triggered:
                    self._last_active_time = time.monotonic()
                    await self._handle_turn(pre_roll)
            else:
                await asyncio.sleep(0.05)

    async def _listen_for_trigger(self) -> tuple[bool, bytes]:
        """Listens for wake word or VAD speech start, respecting follow-up window."""
        start_state = self._state
        follow_up_deadline = (
            time.monotonic() + self.config.agent.follow_up_s
            if start_state == AgentState.FOLLOW_UP
            else None
        )

        speech_frames = 0
        min_speech_frames = int(self.config.audio.min_speech_ms / 32)  # 32ms frames

        async for frame in self.capture.frames():
            if not self._running or self._muted:
                return False, b""

            # Handle follow-up timeout
            if follow_up_deadline and time.monotonic() > follow_up_deadline:
                print("\n[😴 SLEEPING] No follow-up detected. Sleeping now. Say 'Buddy' to talk again!", flush=True)
                target = AgentState.ALWAYS_LISTEN if self._always_mode else AgentState.WAKE_LISTEN
                self._set_state(target)
                return False, b""

            # Always mode or follow-up: speech detection via VAD
            if start_state in (AgentState.ALWAYS_LISTEN, AgentState.FOLLOW_UP):
                is_speech = self.vad.is_speech(frame)
                if is_speech:
                    speech_frames += 1
                    if speech_frames >= min_speech_frames:
                        # Pre-roll: 300 ms
                        pre_roll = self.capture.get_pre_roll(self.config.audio.pre_roll_ms)
                        return True, pre_roll
                else:
                    speech_frames = 0
                continue

            # Wake mode: Vosk wake word detector
            if self.wake.feed(frame):
                logger.info("Wake word detected!")
                # Pre-roll: 1500 ms on wake so 'buddy, what's for dinner?' works
                pre_roll = self.capture.get_pre_roll(self.config.audio.wake_pre_roll_ms)
                return True, pre_roll

        return False, b""

    async def _handle_turn(self, pre_roll: bytes) -> None:
        """Capture user utterance, commit audio, and process model response."""
        self._set_state(AgentState.CAPTURE, level=0.2)

        # Stream capture until end of speech
        speech_started = False
        consecutive_silence = 0
        speech_frames = 0
        min_speech_frames = int(self.config.audio.min_speech_ms / 32)
        silence_threshold_frames = int(self.config.audio.silence_timeout_ms / 32)
        captured_stream: list[bytes] = []

        async for frame in self.capture.frames():
            if not self._running or self._muted:
                return

            captured_stream.append(frame)
            is_speech = self.vad.is_speech(frame)

            if is_speech:
                speech_frames += 1
                consecutive_silence = 0
                self._audio_level = min(1.0, 0.4 + (speech_frames * 0.05))
            else:
                consecutive_silence += 1

            # Ignore blips / silence (require speech or wake pre-roll)
            if consecutive_silence >= silence_threshold_frames:
                if speech_frames < min_speech_frames and not pre_roll:
                    logger.debug("Ignored silence or noise blip (< %d ms)", self.config.audio.min_speech_ms)
                    print("\n[😴 SILENCE] No speech heard. Staying asleep.", flush=True)
                    target = AgentState.ALWAYS_LISTEN if self._always_mode else AgentState.WAKE_LISTEN
                    self._set_state(target)
                    return
                break

        # Send pre-roll audio to session
        if pre_roll:
            await self.session.send_audio(pre_roll)

        # Stream valid speech frames to session
        for f in captured_stream:
            await self.session.send_audio(f)

        # End of user turn -> commit and trigger response
        await self.session.send_commit()
        await self.session.send_response_create()

        # Process model response round(s)
        await self._process_model_response()

    async def _process_model_response(self) -> None:
        """Handle server response, tool execution loops, audio playback, and barge-in."""
        self._set_state(AgentState.THINKING)
        think_timeout = self.config.agent.think_timeout_s

        while self._running and not self._muted:
            collector = ResponseCollector()
            self._active_collector = collector

            # Wait for response with think timeout
            try:
                await asyncio.wait_for(self._stream_response_events(collector), timeout=think_timeout)
            except asyncio.TimeoutError:
                logger.warning("Think timeout (%.1fs) expired waiting for model", think_timeout)
                self._set_state(AgentState.ERROR)
                await asyncio.sleep(0.05)
                target = AgentState.ALWAYS_LISTEN if self._always_mode else AgentState.WAKE_LISTEN
                self._set_state(target)
                return

            if collector.was_cancelled:
                logger.info("Response %s was cancelled due to barge-in", collector.response_id)
                return

            # Check if tools need execution
            if collector.has_tool_calls():
                self._set_state(AgentState.THINKING)
                for tc in collector.tool_calls:
                    print(f"\n[🔧 TOOL] Calling {tc.name}({tc.arguments})...", flush=True)
                    result_str = await self.tools.run_tool(tc.name, tc.arguments)
                    print(f"[📋 RESULT] {result_str}\n", flush=True)
                    await self.session.send_tool_result(tc.call_id, result_str)

                # Request next response after sending tool results
                await self.session.send_response_create()
                continue

            # Record turn history in context manager
            if collector.full_text or collector.full_audio:
                self.context.add_exchange(
                    user_text="(Voice input)",
                    assistant_text=collector.full_text,
                    tool_calls=[{"name": tc.name, "args": tc.arguments} for tc in collector.tool_calls],
                )
                self.context.update_usage(collector.usage)

            # Wait for physical audio playback to finish completely
            await self.player.wait_until_done()

            # Flush any microphone buffer that picked up the speaker's own voice
            self.capture.flush()
            self.wake.reset()
            # Brief delay for room acoustic echo to settle
            await asyncio.sleep(0.3)
            self.capture.flush()

            # Move to follow-up listening window
            self._set_state(AgentState.FOLLOW_UP)
            break

    async def _stream_response_events(self, collector: ResponseCollector) -> None:
        """Stream events for one response, feeding audio to player and monitoring barge-in."""
        barge_in_mode = self.config.agent.barge_in  # 'wake' or 'vad'

        async for event in self.session.recv_iter():
            etype = event.get("type", "")
            resp_id = event.get("response_id") or event.get("response", {}).get("id", "")

            # Ignore events for cancelled response
            if resp_id and self.session.is_cancelled(resp_id):
                continue

            self.session.process_event(event, collector)

            if etype in ("response.output_text.delta", "response.output_audio.delta"):
                if self._state != AgentState.SPEAKING:
                    self._set_state(AgentState.SPEAKING)
                    print()

            if etype == "response.output_text.delta":
                chunk = event.get("delta", "")
                if chunk:
                    print(chunk, end="", flush=True)

            elif etype == "response.output_audio.delta":
                # Feed audio to player
                if collector.audio_parts:
                    latest_audio = collector.audio_parts[-1]
                    self.player.feed(latest_audio)
                    self._audio_level = self.player.amplitude()

                # Check barge-in
                if await self._check_barge_in(barge_in_mode):
                    await self._trigger_barge_in(collector.response_id)
                    collector.was_cancelled = True
                    return

            elif etype == "response.done":
                print(flush=True)
                break

    async def _check_barge_in(self, mode: str) -> bool:
        """Check for user interruption during playback."""
        # Non-blocking check of microphone stream
        if not self.capture._queue or self.capture._queue.empty():
            return False

        try:
            frame = self.capture._queue.get_nowait()
            if mode == "vad":
                return self.vad.is_speech(frame)
            else:
                return self.wake.feed(frame)
        except Exception:
            return False

    async def _trigger_barge_in(self, response_id: str) -> None:
        """Execute barge-in interrupt: clear audio buffer, cancel response, transition to capture."""
        logger.info("Barge-in triggered! Cancelling response %s", response_id)
        self.player.clear()
        await self.session.send_response_cancel()
        pre_roll = self.capture.get_pre_roll(self.config.audio.pre_roll_ms)
        asyncio.create_task(self._handle_turn(pre_roll))

    async def speak_notice(self, text: str) -> bool:
        """Deliver proactive notice as spoken output if idle."""
        if not self.is_idle():
            return False

        logger.info("Speaking proactive notice: %s", text)
        self._set_state(AgentState.THINKING)
        try:
            await self.session.send_text_message(f"[System Notification]: {text}")
            await self.session.send_response_create()
            await self._process_model_response()
            return True
        except Exception as e:
            logger.error("Failed to speak proactive notice: %s", e)
            target = AgentState.ALWAYS_LISTEN if self._always_mode else AgentState.WAKE_LISTEN
            self._set_state(target)
            return False

    async def _perform_context_rollover(self) -> None:
        """Consolidate context and open clean replacement session."""
        logger.info("Starting context 80/20 rollover...")
        to_evict, to_keep = self.context.plan_rollover()
        if not to_evict:
            return

        # Prepare summary of evicted items
        evicted_text = "\n".join(
            f"User: {e.user_text}\nAssistant: {e.assistant_text}" for e in to_evict
        )
        summary_addition = f"Earlier conversation summary:\n{evicted_text[:1000]}"

        # Close old session and connect new session
        await self.session.close()
        self.context.commit_rollover(to_keep, summary_addition)

        try:
            await self.session.connect()
            await self.initialize_session()
            logger.info("Rollover complete; new session connected with %d retained exchanges", len(to_keep))
        except Exception as e:
            logger.error("Rollover reconnect failed: %s", e)
            self._set_state(AgentState.ERROR)
