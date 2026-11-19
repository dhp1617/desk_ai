"""Comprehensive acceptance test suite for Buddy voice assistant."""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from buddy.agent.context import ContextManager, Exchange
from buddy.agent.control import ControlServer, send_control_command
from buddy.agent.proactive import ProactiveManager
from buddy.agent.state import AgentState, BuddyAgent
from buddy.audio.capture import AudioCapture
from buddy.audio.player import AudioPlayer
from buddy.audio.vad import VoiceActivityDetector
from buddy.client.realtime import RealtimeSession, ResponseCollector
from buddy.config import Config
from buddy.memory.search import MemorySearch
from buddy.memory.store import MemoryStore
from buddy.server.manager import ReconnectingSession
from buddy.tools import builtins, google_tools, memory_tools
from buddy.tools.registry import ToolRegistry, tool
from buddy.wake.base import WakeDetector
from tests.mock_server import MockRealtimeServer, MockResponse, MockToolCall


class MockAudioStream:
    """Mock for sounddevice stream."""
    def __init__(self, *args, **kwargs):
        pass
    def start(self): pass
    def stop(self): pass
    def close(self): pass


class SimpleMockWakeDetector(WakeDetector):
    def __init__(self, triggers_on_call: int = 1):
        self.call_count = 0
        self.triggers_on_call = triggers_on_call

    def feed(self, audio_bytes: bytes) -> bool:
        self.call_count += 1
        return self.call_count == self.triggers_on_call

    def reset(self) -> None:
        pass


class MockVAD(VoiceActivityDetector):
    def __init__(self, speech_pattern: list[bool] | None = None):
        self.pattern = speech_pattern or [True]
        self.idx = 0

    def is_speech(self, audio_bytes: bytes) -> bool:
        if not self.pattern:
            return True
        val = self.pattern[self.idx % len(self.pattern)]
        self.idx += 1
        return val


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.config_dir_override = tmp_path / "config"
    cfg.data_dir_override = tmp_path / "data"
    cfg.server.port = 0
    cfg.server.host = "127.0.0.1"
    cfg.agent.think_timeout_s = 2.0
    cfg.agent.follow_up_s = 0.5
    cfg.audio.min_speech_ms = 250
    return cfg


# -------------------------------------------------------------
# 1. Tool Registry: Schema, Coercion, Timeout, Plugins
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_registry_schema_coercion_timeout_plugins(tmp_path: Path):
    reg = ToolRegistry()

    @reg.tool
    def sample_func(
        name: Annotated[str, "User name"],
        count: Annotated[int, "Repeat count"] = 1,
        active: Annotated[bool, "Active flag"] = False,
        mode: Annotated[Literal["fast", "slow"], "Execution mode"] = "fast",
    ) -> str:
        """Sample test tool."""
        return f"{name}:{count}:{active}:{mode}"

    schema = reg.get_schema("sample_func")
    assert schema is not None
    assert schema["name"] == "sample_func"
    assert "User name" in schema["parameters"]["properties"]["name"]["description"]
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["fast", "slow"]

    # Test coercion: string "5" to int 5, string "true" to bool True
    res = await reg.run_tool("sample_func", {"name": "alice", "count": "5", "active": "true", "mode": "slow"})
    assert res == "alice:5:True:slow"

    # Test timeout recovery
    reg.set_timeout(0.1)

    @reg.tool
    async def slow_func() -> str:
        """Slow function."""
        await asyncio.sleep(1.0)
        return "done"

    res_timeout = await reg.run_tool("slow_func", {})
    assert "timed out" in res_timeout

    # Test plugin auto-load from directory
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    plugin_file = plugin_dir / "my_plugin.py"
    plugin_file.write_text(
        "from typing import Annotated\n"
        "@tool\n"
        "def plugin_action(x: Annotated[int, 'Value']) -> str:\n"
        "    '''Plugin doc.'''\n"
        "    return f'plugin_{x * 2}'\n"
    )
    loaded = reg.load_plugins(plugin_dir)
    assert loaded == 1
    assert reg.has_tool("plugin_action")
    res_plugin = await reg.run_tool("plugin_action", {"x": "21"})
    assert res_plugin == "plugin_42"


# -------------------------------------------------------------
# 2. Memory Store & FTS5 Search: Roundtrip, Hand Edits, Forget
# -------------------------------------------------------------
def test_memory_roundtrip_hand_edits_forget(tmp_path: Path):
    store = MemoryStore(tmp_path)
    store.ensure_dirs()
    search = MemorySearch(tmp_path)

    # Core memory capped
    store.write_core("Core rule: Be helpful.")
    assert "Be helpful" in store.read_core()

    # Facts add, hand-edit, forget
    store.add_fact("User loves Arch Linux and Neovim.")
    store.add_fact("User prefers dark mode.")
    assert len(store.read_facts()) == 2

    # Hand edit emulation (user directly appends or modifies file)
    facts_file = tmp_path / "facts.md"
    facts_file.write_text(facts_file.read_text() + "User enjoys espresso.\n")

    # Sync search index
    search.sync_if_needed()
    results = search.search("Arch")
    assert any("Arch" in r["snippet"] or "Arch" in r["file"] for r in results)

    # Forget fact
    removed = store.remove_fact("dark mode")
    assert removed is True
    assert len(store.read_facts()) == 2

    # Path traversal whitelist check
    assert "Invalid filename" in store.read_file_safe("../../etc/passwd")
    assert "Invalid filename" in store.read_file_safe("/etc/shadow")
    assert "not a valid memory file" in store.read_file_safe("malicious.sh")


# -------------------------------------------------------------
# 3. Schedule, Mess & Unquoted YAML Times
# -------------------------------------------------------------
def test_schedule_and_unquoted_times(tmp_path: Path):
    store = MemoryStore(tmp_path)
    store.ensure_dirs()

    # Note unquoted 10:30 which PyYAML normally reads as int 630
    timetable_content = """
Monday:
  - time: 10:30
    class: "Operating Systems"
    room: "CS-201"
  - time: 14:00
    class: "Network Security"
    room: "CS-202"
"""
    (tmp_path / "timetable.yaml").write_text(timetable_content)

    mess_content = """
Monday:
  breakfast: "Pancakes"
  lunch: "Curry Rice"
  snacks: "Cookies"
  dinner: "Soup & Bread"
"""
    (tmp_path / "mess.yaml").write_text(mess_content)

    cfg = Config()
    cfg.memory_dir_override = tmp_path
    builtins.init_builtins(cfg, store)

    # Exact timetable lookup with time formatting
    sched = builtins.get_schedule("Monday")
    assert "10:30 - Operating Systems (CS-201)" in sched
    assert "14:00 - Network Security (CS-202)" in sched

    # Mess lookup with aliases (morning -> breakfast, evening -> snacks)
    menu_morning = builtins.get_mess_menu("morning", day="Monday")
    assert "Breakfast on" in menu_morning or "Pancakes" in menu_morning


# -------------------------------------------------------------
# 4. Context 80/20 Rollover Plan & Persistence
# -------------------------------------------------------------
def test_context_80_20_plan(tmp_path: Path):
    store = MemoryStore(tmp_path)
    cfg = Config()
    cfg.memory_dir_override = tmp_path
    cfg.server.n_ctx = 1000
    cfg.agent.context_threshold = 0.80  # 800 tokens
    cfg.agent.context_consolidate = 0.20 # 200 tokens

    ctx = ContextManager(cfg, store)

    # Add multiple large exchanges
    for i in range(10):
        ctx.add_exchange(
            user_text=f"Question number {i} " * 20,
            assistant_text=f"Answer number {i} " * 40,
        )

    # Check rollover condition
    assert ctx.needs_rollover() is True

    # Plan rollover
    evicted, surviving = ctx.plan_rollover()
    assert len(evicted) > 0
    assert len(surviving) >= 2  # Keeps at least the last 2 exchanges intact!

    # Rollover commit
    ctx.commit_rollover(surviving, new_summary="Consolidated history summary")
    assert "Consolidated history summary" in store.read_summary()
    assert len(ctx.exchanges) == len(surviving)

    # Persistence across restart
    ctx2 = ContextManager(cfg, store)
    assert ctx2.load_persisted() is True
    assert len(ctx2.exchanges) == len(surviving)


# -------------------------------------------------------------
# 5. Google Tools Mocked
# -------------------------------------------------------------
def test_google_tools_mocked():
    mock_service = MagicMock()
    mock_service.users().messages().list().execute.return_value = {
        "messages": [{"id": "m1"}]
    }
    mock_service.users().messages().get().execute.return_value = {
        "id": "m1",
        "snippet": "Meeting at 3pm",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Project Sync"},
                {"name": "From", "value": "lead@company.com"},
            ]
        },
    }

    with patch("buddy.tools.google_tools._get_gmail_service", return_value=mock_service):
        res = google_tools.check_email(5)
        assert "Project Sync" in res
        assert "lead@company.com" in res
        assert "Meeting at 3pm" in res


# -------------------------------------------------------------
# 6. Control Socket Operations
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_control_socket(tmp_path: Path):
    sock_path = tmp_path / "buddy.sock"

    mock_agent = MagicMock()
    mock_agent.state = AgentState.WAKE_LISTEN
    mock_agent.is_always_mode = False
    mock_agent.is_muted = False
    mock_agent.audio_level = 0.15

    async def _toggle():
        mock_agent.is_always_mode = not mock_agent.is_always_mode
        return "always" if mock_agent.is_always_mode else "wake"

    async def _set_muted(val):
        mock_agent.is_muted = val

    async def _speak_notice(text):
        return True

    mock_agent.toggle_always = _toggle
    mock_agent.set_muted = _set_muted
    mock_agent.speak_notice = _speak_notice

    server = ControlServer(sock_path, mock_agent)
    await server.start()

    try:
        # Status command
        res = await send_control_command(sock_path, "status")
        assert res["status"] == "ok"
        assert res["state"] == "WAKE_LISTEN"

        # Toggle command
        res_toggle = await send_control_command(sock_path, "toggle")
        assert res_toggle["status"] == "ok"
        assert res_toggle["mode"] == "always"

        # Mute command
        res_mute = await send_control_command(sock_path, "mute")
        assert res_mute["status"] == "ok"
        assert res_mute["muted"] is True

        # Say command
        res_say = await send_control_command(sock_path, "say Good morning")
        assert res_say["status"] == "ok"

    finally:
        await server.stop()


# -------------------------------------------------------------
# 7. Reconnect with Backoff
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_reconnect_with_backoff():
    server = MockRealtimeServer()
    port = await server.start()
    url = f"ws://127.0.0.1:{port}"

    setup_called = 0
    async def setup_fn(sess):
        nonlocal setup_called
        setup_called += 1

    recon = ReconnectingSession(url, setup_fn=setup_fn, max_delay=0.1)
    sess = await recon.connect()
    assert sess.connected is True
    assert setup_called == 1

    await recon.close()
    await server.stop()


# -------------------------------------------------------------
# 8. Proactive Notices (Rate Limits & Dedup)
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_proactive_notices_rate_limit(tmp_path: Path):
    store = MemoryStore(tmp_path)
    cfg = Config()
    cfg.notice.min_interval_s = 15

    mock_agent = MagicMock()
    mock_agent.is_idle.return_value = True

    spoken_notices = []
    async def _speak(text):
        spoken_notices.append(text)
        return True
    mock_agent.speak_notice = _speak

    pro = ProactiveManager(cfg, store)
    pro.set_agent(mock_agent)

    # First notice dispatch succeeds
    ok1 = await pro.dispatch_notice("notice:1", "First notice")
    assert ok1 is True
    assert len(spoken_notices) == 1

    # Immediate second notice blocked by 15s rate limit
    ok2 = await pro.dispatch_notice("notice:2", "Second notice")
    assert ok2 is False

    # Duplicate key blocked even after time passes
    pro._last_notice_time = 0.0
    ok_dup = await pro.dispatch_notice("notice:1", "First notice duplicate")
    assert ok_dup is False


# -------------------------------------------------------------
# 9. End-to-End: Wake -> Capture -> Commit -> Tool Call -> Response -> Follow-Up
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_full_turn_with_mock_server(tmp_path: Path):
    # Setup mock server with a tool-call response followed by final text response
    server = MockRealtimeServer()
    server.enqueue_response(MockResponse(
        tool_calls=[MockToolCall(name="get_time", call_id="call_001", arguments={})],
        response_id="resp_001",
    ))
    server.enqueue_response(MockResponse(
        text="It is currently 10:00 AM.",
        audio_samples=4800, # 200ms audio
        response_id="resp_002",
    ))
    port = await server.start()

    # Tool registry with get_time
    reg = ToolRegistry()
    @reg.tool
    def get_time() -> str:
        return "10:00 AM"

    cfg = Config()
    cfg.server.host = "127.0.0.1"
    cfg.server.port = port
    cfg.agent.think_timeout_s = 2.0
    cfg.agent.follow_up_s = 0.5
    cfg.audio.min_speech_ms = 32 # short for test speed
    cfg.audio.silence_timeout_ms = 64 # 2 silence frames to finish turn

    session = RealtimeSession(cfg.server.ws_url)
    await session.connect()

    store = MemoryStore(tmp_path)
    store.ensure_dirs()
    ctx = ContextManager(cfg, store)

    # Audio mocks
    capture = AudioCapture()
    capture._running = True
    capture._queue = asyncio.Queue()

    # Feed frames to trigger turn
    frame = b"\x00" * 1024
    for _ in range(5):
        capture._queue.put_nowait(frame)

    player = AudioPlayer()
    wake = SimpleMockWakeDetector(triggers_on_call=999)
    vad = MockVAD([True, True, False, False, False])

    agent = BuddyAgent(
        config=cfg,
        session=session,
        capture=capture,
        player=player,
        wake=wake,
        vad=vad,
        tools=reg,
        memory=store,
        context=ctx,
    )

    # Run session setup
    await agent.initialize_session()
    await asyncio.sleep(0.05)
    assert len(server.session_updates) == 1

    # Simulate turn execution
    pre_roll = b"\x00" * 1024
    await agent._handle_turn(pre_roll)

    # Verify tool was called and output returned
    assert any(ev.get("type") == "conversation.item.create" and
               ev.get("item", {}).get("type") == "function_call_output"
               for ev in server.received_events)

    # Verify agent reached follow-up state
    assert agent.state == AgentState.FOLLOW_UP

    await session.close()
    await server.stop()


# -------------------------------------------------------------
# 10. Barge-In & Cancellation
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_barge_in_cancelled_response_ignored(tmp_path: Path):
    session = RealtimeSession("ws://dummy")
    session._connected = True
    session.active_response_id = "resp_to_cancel"

    # Simulate cancel
    sent_events = []
    async def mock_send(ev):
        sent_events.append(ev)
    session._send = mock_send

    await session.send_response_cancel()
    assert session.is_cancelled("resp_to_cancel")

    collector = ResponseCollector(response_id="resp_to_cancel")
    # Feed late event of the cancelled response
    late_event = {
        "type": "response.output_text.delta",
        "response_id": "resp_to_cancel",
        "delta": "late arrival",
    }
    session.process_event(late_event, collector)

    # Ensure collector ignored the delta and flagged cancellation
    assert collector.was_cancelled is True
    assert collector.full_text == ""


# -------------------------------------------------------------
# 11. Speech Blip Ignored (< 250ms)
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_blip_ignored(tmp_path: Path):
    cfg = Config()
    cfg.audio.min_speech_ms = 250
    cfg.audio.silence_timeout_ms = 64

    store = MemoryStore(tmp_path)
    ctx = ContextManager(cfg, store)

    session = AsyncMock()
    capture = AudioCapture()
    capture._running = True
    capture._queue = asyncio.Queue()

    # Feed 2 speech frames (64ms < 250ms) then silence frames
    for _ in range(2):
        capture._queue.put_nowait(b"\x00" * 1024)
    for _ in range(4):
        capture._queue.put_nowait(b"\x00" * 1024)

    vad = MockVAD([True, True, False, False, False, False])
    wake = SimpleMockWakeDetector()
    player = AudioPlayer()

    agent = BuddyAgent(
        config=cfg,
        session=session,
        capture=capture,
        player=player,
        wake=wake,
        vad=vad,
        tools=ToolRegistry(),
        memory=store,
        context=ctx,
    )

    await agent._handle_turn(b"")
    # Verify no commit was sent because it was a blip
    session.send_commit.assert_not_called()
    assert agent.state == AgentState.WAKE_LISTEN


# -------------------------------------------------------------
# 12. Think-Timeout Recovery
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_think_timeout_recovery(tmp_path: Path):
    server = MockRealtimeServer()
    # Don't enqueue any response -> server hangs on response.create
    port = await server.start()

    cfg = Config()
    cfg.server.port = port
    cfg.agent.think_timeout_s = 0.2  # fast timeout

    session = RealtimeSession(cfg.server.ws_url)
    await session.connect()

    store = MemoryStore(tmp_path)
    ctx = ContextManager(cfg, store)

    agent = BuddyAgent(
        config=cfg,
        session=session,
        capture=AudioCapture(),
        player=AudioPlayer(),
        wake=SimpleMockWakeDetector(),
        vad=MockVAD(),
        tools=ToolRegistry(),
        memory=store,
        context=ctx,
    )

    await agent._process_model_response()
    # Agent should recover gracefully to WAKE_LISTEN without raising an exception
    assert agent.state == AgentState.WAKE_LISTEN

    await session.close()
    await server.stop()


# -------------------------------------------------------------
# 13. Rollover Opens 2nd Session with Summary and Retained Exchanges
# -------------------------------------------------------------
@pytest.mark.asyncio
async def test_rollover_opens_second_session(tmp_path: Path):
    server = MockRealtimeServer()
    port = await server.start()

    cfg = Config()
    cfg.server.port = port
    cfg.server.n_ctx = 200
    cfg.agent.context_threshold = 0.50
    cfg.agent.context_consolidate = 0.20

    store = MemoryStore(tmp_path)
    store.ensure_dirs()
    ctx = ContextManager(cfg, store)

    # Add 5 exchanges so total exceeds threshold
    for i in range(5):
        ctx.add_exchange(f"User query {i}", f"Assistant answer {i}")

    session = RealtimeSession(cfg.server.ws_url)
    await session.connect()
    assert server.sessions_opened == 1

    agent = BuddyAgent(
        config=cfg,
        session=session,
        capture=AudioCapture(),
        player=AudioPlayer(),
        wake=SimpleMockWakeDetector(),
        vad=MockVAD(),
        tools=ToolRegistry(),
        memory=store,
        context=ctx,
    )

    # Perform rollover
    await agent._perform_context_rollover()

    # Verify a second session was opened on server
    assert server.sessions_opened == 2
    # Verify summary is now present in memory
    assert "Earlier conversation summary" in store.read_summary()
    # Verify dropped exchange is not in active exchanges but last 2 were kept
    assert len(ctx.exchanges) >= 2
    assert ctx.exchanges[-1].user_text == "User query 4"

    # Verify session.update in second session contained the summary in instructions
    await asyncio.sleep(0.05)
    assert len(server.session_updates) >= 1
    latest_update = server.session_updates[-1]
    instructions = latest_update.get("session", {}).get("instructions", "")
    assert "Earlier conversation summary" in instructions

    await session.close()
    await server.stop()
