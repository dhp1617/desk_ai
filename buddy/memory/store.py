"""File-based memory store (OpenClaw-style, hand-editable)."""
from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CORE_MAX_CHARS = 2000

# Whitelist of allowed filenames (no path traversal)
_ALLOWED_FILES = {
    "core.md", "facts.md", "summary.md",
    "timetable.yaml", "mess.yaml",
}
_NOTES_PATTERN = re.compile(r"^notes/\d{4}-\d{2}-\d{2}\.md$")


class MemoryStore:
    """Plain-file memory system."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir

    def ensure_dirs(self) -> None:
        """Create memory directory structure."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "notes").mkdir(exist_ok=True)

    # --- Core memory ---

    def read_core(self) -> str:
        path = self.memory_dir / "core.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_core(self, content: str) -> None:
        if len(content) > CORE_MAX_CHARS:
            content = content[:CORE_MAX_CHARS]
            logger.warning("Core memory truncated to %d chars", CORE_MAX_CHARS)
        path = self.memory_dir / "core.md"
        path.write_text(content, encoding="utf-8")

    # --- Facts ---

    def read_facts(self) -> list[str]:
        path = self.memory_dir / "facts.md"
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def add_fact(self, fact: str) -> None:
        path = self.memory_dir / "facts.md"
        self.ensure_dirs()
        with open(path, "a", encoding="utf-8") as f:
            f.write(fact.rstrip("\n") + "\n")

    def remove_fact(self, pattern: str) -> bool:
        path = self.memory_dir / "facts.md"
        if not path.exists():
            return False
        lines = path.read_text(encoding="utf-8").splitlines()
        pat_lower = pattern.lower()
        new_lines = [l for l in lines if pat_lower not in l.lower()]
        if len(new_lines) == len(lines):
            return False
        path.write_text("\n".join(new_lines) + "\n" if new_lines else "", encoding="utf-8")
        return True

    # --- Summary ---

    def read_summary(self) -> str:
        path = self.memory_dir / "summary.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_summary(self, summary: str) -> None:
        path = self.memory_dir / "summary.md"
        self.ensure_dirs()
        path.write_text(summary, encoding="utf-8")

    # --- Daily logs ---

    def append_log(self, text: str) -> None:
        today = datetime.date.today().isoformat()
        path = self.memory_dir / "notes" / f"{today}.md"
        self.ensure_dirs()
        with open(path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%H:%M")
            f.write(f"[{timestamp}] {text}\n")

    # --- Safe file access ---

    def read_file_safe(self, name: str) -> str:
        """Read a whitelisted file. No path traversal allowed."""
        # Security: prevent path escape
        if ".." in name or name.startswith("/"):
            return "Error: Invalid filename."

        if name not in _ALLOWED_FILES and not _NOTES_PATTERN.match(name):
            return f"Error: '{name}' is not a valid memory file. Allowed: core.md, facts.md, summary.md, timetable.yaml, mess.yaml, notes/YYYY-MM-DD.md"

        path = self.memory_dir / name
        if not path.exists():
            return f"File '{name}' does not exist."
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading '{name}': {e}"

    # --- Structured data ---

    def get_timetable(self) -> dict:
        """Load timetable.yaml with time fixing."""
        return self._load_yaml("timetable.yaml")

    def get_mess_menu(self) -> dict:
        """Load mess.yaml with time fixing."""
        return self._load_yaml("mess.yaml")

    def _load_yaml(self, filename: str) -> dict:
        import yaml

        path = self.memory_dir / filename
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _fix_yaml_times(data)
        return data

    # --- Prompt context ---

    def get_prompt_context(self) -> str:
        """Return core memory for inclusion in system prompt."""
        core = self.read_core()
        if core:
            return f"[Core Memory]\n{core}\n"
        return ""


def _fix_yaml_times(data, _depth: int = 0) -> None:
    """Recursively fix YAML time values: PyYAML reads 10:30 as int 630."""
    if _depth > 10:
        return
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, int) and _is_time_like_int(val):
                # Check if this is in a time-like context
                if isinstance(key, str) and key.lower() in ("time", "start", "end", "from", "to"):
                    data[key] = _int_to_time(val)
            elif isinstance(val, (dict, list)):
                _fix_yaml_times(val, _depth + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, int) and _is_time_like_int(item):
                data[i] = _int_to_time(item)
            elif isinstance(item, (dict, list)):
                _fix_yaml_times(item, _depth + 1)


def _is_time_like_int(val: int) -> bool:
    """Check if an int could be a YAML-mangled time (H*60+M)."""
    return 0 <= val <= 1439


def _int_to_time(val: int) -> str:
    """Convert H*60+M int back to HH:MM string."""
    hours = val // 60
    minutes = val % 60
    return f"{hours:02d}:{minutes:02d}"
