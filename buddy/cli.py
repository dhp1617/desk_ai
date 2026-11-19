"""Buddy command-line interface."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from buddy.config import Config, load_config

logger = logging.getLogger("buddy")


def _init_config_and_examples(cfg: Config) -> None:
    """Initialize configuration directory and sample data."""
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    (cfg.memory_dir / "notes").mkdir(exist_ok=True)
    cfg.plugin_dir.mkdir(parents=True, exist_ok=True)

    config_path = cfg.config_dir / "config.toml"
    if not config_path.exists():
        example_cfg = Path(__file__).resolve().parent.parent / "examples" / "config.toml"
        if example_cfg.exists():
            shutil.copy(example_cfg, config_path)
            print(f"Created config: {config_path}")

    timetable_path = cfg.memory_dir / "timetable.yaml"
    if not timetable_path.exists():
        example_tt = Path(__file__).resolve().parent.parent / "examples" / "timetable.yaml"
        if example_tt.exists():
            shutil.copy(example_tt, timetable_path)
            print(f"Created sample timetable: {timetable_path}")

    mess_path = cfg.memory_dir / "mess.yaml"
    if not mess_path.exists():
        example_mess = Path(__file__).resolve().parent.parent / "examples" / "mess.yaml"
        if example_mess.exists():
            shutil.copy(example_mess, mess_path)
            print(f"Created sample mess menu: {mess_path}")

    core_path = cfg.memory_dir / "core.md"
    if not core_path.exists():
        core_path.write_text("User is a computer science student at university. Prefers concise answers.\n")
        print(f"Created core memory: {core_path}")

    plugin_sample = cfg.plugin_dir / "example_plugin.py"
    if not plugin_sample.exists():
        src_plugin = Path(__file__).resolve().parent.parent / "examples" / "tools" / "example_plugin.py"
        if src_plugin.exists():
            shutil.copy(src_plugin, plugin_sample)
            print(f"Created sample plugin tool: {plugin_sample}")


def cmd_init(args: argparse.Namespace) -> None:
    cfg = load_config()
    _init_config_and_examples(cfg)
    print("Buddy initialized successfully!")


def cmd_doctor(args: argparse.Namespace) -> None:
    cfg = load_config()
    print("=== Buddy System Diagnostics ===")

    # 1. llama-server
    server_bin = shutil.which("llama-server")
    print(f"[*] llama-server: {'FOUND (' + server_bin + ')' if server_bin else 'NOT FOUND (Install llama.cpp b11960+)'}")

    # 2. Audio devices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
        print(f"[*] sounddevice: OK (Default in: #{default_in}, Default out: #{default_out})")
    except Exception as e:
        print(f"[*] sounddevice: ERROR ({e})")

    # 3. Vosk
    try:
        import vosk
        print(f"[*] vosk: OK (module installed)")
    except Exception as e:
        print(f"[*] vosk: NOT AVAILABLE ({e})")

    # 4. VAD
    try:
        from pysilero_vad import SileroVoiceActivityDetector
        vad = SileroVoiceActivityDetector()
        print(f"[*] pysilero-vad: OK")
    except Exception as e:
        print(f"[*] pysilero-vad: NOT AVAILABLE ({e})")

    # 5. GTK4
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import Gtk
        print(f"[*] GTK4: OK")
    except Exception as e:
        print(f"[*] GTK4: NOT AVAILABLE ({e})")

    # 6. Screenshot tool
    has_grim = shutil.which("grim") is not None
    has_maim = shutil.which("maim") is not None
    tool_name = "grim" if has_grim else ("maim" if has_maim else "NONE")
    print(f"[*] Screenshot tool: {tool_name}")

    # 7. Google Auth
    token_file = cfg.google_token_file
    if token_file.exists():
        print(f"[*] Google Auth: Token present at {token_file}")
    else:
        print(f"[*] Google Auth: No token found. Run 'buddy auth-google' to authenticate.")

    if args.probe:
        print("\n--- Running Live Probe against llama-server ---")
        asyncio.run(_run_probe(cfg))


async def _run_probe(cfg: Config) -> None:
    from buddy.client.realtime import RealtimeSession
    from buddy.tools.registry import get_registry

    print(f"Connecting to {cfg.server.ws_url}...")
    session = RealtimeSession(cfg.server.ws_url, debug=True)
    try:
        await session.connect()
        print(f"Connected! Audio input rate: {session.input_rate}Hz, output: {session.output_rate}Hz")
        tools = get_registry().get_all_tools()
        print(f"Sending session.update with {len(tools)} tools...")
        await session.send_session_update(
            instructions="You are Buddy. Test probe session.",
            tools=tools,
        )
        print("Probe session successfully initialized!")
        await session.close()
    except Exception as e:
        print(f"Probe failed: {e}")


def cmd_auth_google(args: argparse.Namespace) -> None:
    """Run Google OAuth installed application flow and save token with 0600 mode."""
    cfg = load_config()
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Error: Google client libraries not installed. Run: pip install 'buddy-assistant[google]'")
        return

    client_secrets = cfg.google.client_secrets_file or str(cfg.config_dir / "client_secrets.json")
    if not os.path.exists(client_secrets):
        print(f"Error: client_secrets.json not found at {client_secrets}.")
        print("Please download OAuth 2.0 Client credentials from Google Cloud Console.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(client_secrets, cfg.google.scopes)
    creds = flow.run_local_server(port=0)

    token_path = cfg.google_token_file
    token_path.parent.mkdir(parents=True, exist_ok=True)

    # Save credentials JSON with 0600 file mode
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(token_path), flags, 0o600)
    with open(fd, "w") as f:
        f.write(creds.to_json())

    print(f"Successfully authenticated! Token saved with 0600 permissions to {token_path}")


def cmd_tools(args: argparse.Namespace) -> None:
    from buddy.tools.registry import get_registry
    import buddy.tools.builtins  # Ensure builtins registered
    import buddy.tools.memory_tools
    import buddy.tools.google_tools

    cfg = load_config()
    registry = get_registry()
    registry.load_plugins(cfg.plugin_dir)

    tools = registry.get_all_tools()
    print(f"Registered Tools ({len(tools)}):")
    for t in tools:
        print(f"\n- {t['name']}: {t.get('description', '')}")
        params = t.get("parameters", {}).get("properties", {})
        for p, pdata in params.items():
            req = " (required)" if p in t.get("parameters", {}).get("required", []) else ""
            print(f"    * {p} [{pdata.get('type')}]{req}: {pdata.get('description', '')}")


def _send_cli_command(cmd_str: str) -> None:
    from buddy.agent.control import send_control_command
    cfg = load_config()
    try:
        res = asyncio.run(send_control_command(cfg.socket_path, cmd_str))
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.debug_events:
        cfg.debug_events = True
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug_events else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if cfg.debug_events:
        for noisy in ("websockets", "httpcore", "httpx", "asyncio", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.INFO)

    _init_config_and_examples(cfg)

    from buddy.audio.capture import AudioCapture
    from buddy.audio.player import AudioPlayer
    from buddy.audio.vad import VoiceActivityDetector
    from buddy.wake.vosk_wake import VoskWakeDetector
    from buddy.memory.store import MemoryStore
    from buddy.memory.search import MemorySearch
    from buddy.agent.context import ContextManager
    from buddy.agent.state import BuddyAgent
    from buddy.agent.proactive import ProactiveManager
    from buddy.agent.control import ControlServer
    from buddy.server.manager import ServerManager
    from buddy.client.realtime import RealtimeSession
    from buddy.tools.registry import get_registry
    from buddy.tools import builtins, memory_tools, google_tools

    # 1. Initialize stores and search
    store = MemoryStore(cfg.memory_dir)
    store.ensure_dirs()
    search = MemorySearch(cfg.memory_dir)
    search.reindex()

    # 2. Initialize tools
    builtins.init_builtins(cfg, store)
    memory_tools.init_memory_tools(store, search)
    google_tools.init_google_tools(cfg)

    registry = get_registry()
    registry.load_plugins(cfg.plugin_dir)

    # 3. Server manager
    server_mgr = ServerManager(cfg.server)
    if not args.no_autostart:
        print(f"Ensuring llama-server is running at {cfg.server.base_url}...")
        try:
            asyncio.run(server_mgr.ensure_running())
        except Exception as e:
            print(f"Warning: Could not connect to llama-server: {e}")

    # 4. Audio, Wake, VAD
    capture = AudioCapture(sample_rate=cfg.audio.sample_rate)
    player = AudioPlayer(sample_rate=cfg.audio.output_rate, jitter_ms=cfg.audio.jitter_buffer_ms)
    vad = VoiceActivityDetector(threshold=cfg.audio.vad_threshold)
    wake = VoskWakeDetector(wake_word=cfg.agent.wake_word)

    # 5. Session and Context
    session = RealtimeSession(cfg.server.ws_url, debug=cfg.debug_events)
    context = ContextManager(cfg, store)
    context.load_persisted()

    # 6. Agent and Proactive
    orb = None
    if not args.no_orb:
        try:
            from buddy.orb.widget import BuddyOrb
            orb = BuddyOrb(size=cfg.orb.size)
        except Exception as e:
            logger.warning("Failed to initialize Orb: %s", e)

    def on_state_change(st, level):
        if orb:
            orb.set_state(st.name, level)

    agent = BuddyAgent(
        config=cfg,
        session=session,
        capture=capture,
        player=player,
        wake=wake,
        vad=vad,
        tools=registry,
        memory=store,
        context=context,
        on_state_change=on_state_change,
    )

    def _safe_run_coro(coro):
        if main_loop and not main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, main_loop)
        else:
            coro.close()

    if orb:
        orb.on_left_click = lambda: _safe_run_coro(agent.toggle_always())
        orb.on_right_click = lambda: _safe_run_coro(agent.set_muted(not agent.is_muted))

    proactive = ProactiveManager(cfg, store)
    proactive.set_agent(agent)

    control = ControlServer(cfg.socket_path, agent)

    async def main_async():
        await capture.start(pre_roll_ms=cfg.audio.wake_pre_roll_ms)
        await player.start()
        await session.connect()
        await control.start()
        await proactive.start()
        await agent.run()

    import threading

    def run_agent_thread():
        nonlocal main_loop
        main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_loop)
        try:
            main_loop.run_until_complete(main_async())
        finally:
            main_loop.run_until_complete(session.close())
            main_loop.run_until_complete(capture.stop())
            main_loop.run_until_complete(player.stop())
            main_loop.run_until_complete(control.stop())
            main_loop.run_until_complete(proactive.stop())
            main_loop.close()

    main_loop = None
    if orb and not args.no_orb:
        t = threading.Thread(target=run_agent_thread, daemon=True)
        t.start()
        orb.run()
    else:
        main_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(main_loop)
        try:
            main_loop.run_until_complete(main_async())
        except KeyboardInterrupt:
            print("\nShutting down Buddy...")
        finally:
            main_loop.run_until_complete(session.close())
            main_loop.run_until_complete(capture.stop())
            main_loop.run_until_complete(player.stop())
            main_loop.run_until_complete(control.stop())
            main_loop.run_until_complete(proactive.stop())
            main_loop.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="buddy", description="Buddy: Speech-to-Speech Desktop Assistant")
    subparsers = parser.add_subparsers(dest="command")

    # run
    p_run = subparsers.add_parser("run", help="Start the Buddy voice assistant")
    p_run.add_argument("--debug-events", action="store_true", help="Log all realtime WebSocket events")
    p_run.add_argument("--no-orb", action="store_true", help="Disable the GTK floating orb UI")
    p_run.add_argument("--no-autostart", action="store_true", help="Do not autostart llama-server")

    # init
    subparsers.add_parser("init", help="Initialize configuration and sample files")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Check system dependencies and audio devices")
    p_doc.add_argument("--probe", action="store_true", help="Run a live test session against llama-server")

    # auth-google
    subparsers.add_parser("auth-google", help="Authenticate with Google APIs (Gmail, Classroom)")

    # tools
    subparsers.add_parser("tools", help="List registered tools and schemas")

    # control commands
    subparsers.add_parser("toggle", help="Toggle between wake word and always-listening mode")
    subparsers.add_parser("always", help="Switch to always-listening mode")
    subparsers.add_parser("wake", help="Switch to wake-word-only mode")
    subparsers.add_parser("mute", help="Mute the assistant")
    subparsers.add_parser("unmute", help="Unmute the assistant")
    subparsers.add_parser("status", help="Get current assistant status")

    p_say = subparsers.add_parser("say", help="Speak a text message through the assistant")
    p_say.add_argument("text", help="Text to speak")

    subparsers.add_parser("quit", help="Stop the running assistant")

    args = parser.parse_args()

    if not args.command or args.command == "run":
        cmd_run(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "auth-google":
        cmd_auth_google(args)
    elif args.command == "tools":
        cmd_tools(args)
    elif args.command in ("toggle", "always", "wake", "mute", "unmute", "status", "quit"):
        _send_cli_command(args.command)
    elif args.command == "say":
        _send_cli_command(f"say {args.text}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
