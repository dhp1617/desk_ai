"""Typed configuration with TOML override at ~/.config/buddy/config.toml."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


def _config_dir() -> Path:
    return Path(os.environ.get("BUDDY_CONFIG_DIR", Path.home() / ".config" / "buddy"))


def _data_dir() -> Path:
    return Path(os.environ.get("BUDDY_DATA_DIR", Path.home() / ".local" / "share" / "buddy"))


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9931
    model_hf: str = "LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M"
    n_ctx: int = 32768
    ngl: int = 99
    extra_args: list[str] = field(default_factory=list)
    autostart: bool = True
    health_timeout: float = 120.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        import urllib.parse
        model_q = urllib.parse.quote(self.model_hf)
        return f"ws://{self.host}:{self.port}/v1/realtime?model={model_q}"

    @property
    def ws_transcription_url(self) -> str:
        import urllib.parse
        model_q = urllib.parse.quote(self.model_hf)
        return f"ws://{self.host}:{self.port}/v1/realtime?intent=transcription&model={model_q}"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    output_rate: int = 24000
    vad_threshold: float = 0.5
    min_speech_ms: int = 250
    pre_roll_ms: int = 300
    wake_pre_roll_ms: int = 1500
    jitter_buffer_ms: int = 200
    silence_timeout_ms: int = 1500


@dataclass
class AgentConfig:
    wake_word: str = "buddy"
    wake_backend: str = "vosk"  # vosk | openwakeword
    follow_up_s: float = 8.0
    always_timeout_s: float = 600.0  # 10 min
    barge_in: str = "wake"  # wake | vad
    think_timeout_s: float = 30.0
    context_threshold: float = 0.80
    context_consolidate: float = 0.20
    system_prompt: str = (
        "You are Buddy, a concise, fast local voice assistant.\n"
        "DIRECT ACTION RULES:\n"
        "1. Never say 'I can get...', 'Would you like me to...', or ask for confirmation. Call the tool IMMEDIATELY.\n"
        "2. For meals, food, dining, lunch, dinner, breakfast, snacks: call `get_mess_menu` immediately.\n"
        "3. For classes, timetable, lectures, schedule: call `get_schedule` or `next_class` immediately.\n"
        "4. For remembering facts: call `remember` immediately.\n"
        "5. For recalling information: call `search_memory` immediately.\n"
        "6. Keep spoken answers brief and natural (1-2 sentences). Never list your capabilities at the end of answers."
    )
    max_output_tokens: int = 4096


@dataclass
class NoticeConfig:
    class_remind_min: int = 10
    meal_remind_min: int = 15
    email_poll_min: int = 10
    min_interval_s: int = 15


@dataclass
class OrbConfig:
    size: int = 48
    use_layer_shell: bool = True
    position: str = "bottom-right"  # top-left|top-right|bottom-left|bottom-right
    margin: int = 20


@dataclass
class GoogleConfig:
    client_secrets_file: str = ""
    token_file: str = ""  # defaults to config_dir/google_token.json
    scopes: list[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/classroom.courses.readonly",
        "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
        "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    ])


@dataclass
class ToolsConfig:
    allowed_apps: list[str] = field(default_factory=lambda: [
        "firefox", "chromium", "code", "nautilus", "thunar", "kitty", "alacritty",
    ])
    plugin_dir: str = ""  # defaults to config_dir/tools


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    notice: NoticeConfig = field(default_factory=NoticeConfig)
    orb: OrbConfig = field(default_factory=OrbConfig)
    google: GoogleConfig = field(default_factory=GoogleConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    debug_events: bool = False

    memory_dir_override: Path | None = None
    config_dir_override: Path | None = None
    data_dir_override: Path | None = None

    @property
    def config_dir(self) -> Path:
        if self.config_dir_override:
            return Path(self.config_dir_override)
        return _config_dir()

    @property
    def data_dir(self) -> Path:
        if self.data_dir_override:
            return Path(self.data_dir_override)
        return _data_dir()

    @property
    def memory_dir(self) -> Path:
        if self.memory_dir_override:
            return Path(self.memory_dir_override)
        return self.data_dir / "memory"

    @property
    def socket_path(self) -> Path:
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
        return runtime / "buddy.sock"

    @property
    def plugin_dir(self) -> Path:
        if self.tools.plugin_dir:
            return Path(self.tools.plugin_dir)
        return self.config_dir / "tools"

    @property
    def google_token_file(self) -> Path:
        if self.google.token_file:
            return Path(self.google.token_file)
        return self.config_dir / "google_token.json"


def _apply_dict(dc: Any, data: dict[str, Any]) -> None:
    """Recursively apply dict values onto a dataclass instance."""
    for f in fields(dc):
        if f.name in data:
            val = data[f.name]
            current = getattr(dc, f.name)
            if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
                _apply_dict(current, val)
            else:
                setattr(dc, f.name, val)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML, falling back to defaults."""
    cfg = Config()
    if path is None:
        path = cfg.config_dir / "config.toml"
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        _apply_dict(cfg, data)
    return cfg
