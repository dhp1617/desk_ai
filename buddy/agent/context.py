"""Rolling context manager and session rollover."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from buddy.config import Config
from buddy.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class Exchange:
    user_text: str = ""
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Exchange:
        return cls(
            user_text=data.get("user_text", ""),
            assistant_text=data.get("assistant_text", ""),
            tool_calls=data.get("tool_calls", []),
            timestamp=data.get("timestamp", 0.0),
        )

    def estimated_tokens(self) -> int:
        total_len = len(self.user_text) + len(self.assistant_text)
        for tc in self.tool_calls:
            total_len += len(str(tc))
        return max(1, total_len // 4)


class ContextManager:
    """Tracks session dialogue history and plans 80/20 rolling consolidations."""

    def __init__(self, config: Config, memory: MemoryStore):
        self.config = config
        self.memory = memory
        self.exchanges: list[Exchange] = []
        self._persisted_file = memory.memory_dir / ".active_session.json"
        self._last_server_usage: dict[str, int] | None = None

    def update_usage(self, usage: dict[str, int] | None) -> None:
        if usage:
            self._last_server_usage = usage

    def add_exchange(self, user_text: str, assistant_text: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        import time
        ex = Exchange(
            user_text=user_text,
            assistant_text=assistant_text,
            tool_calls=tool_calls or [],
            timestamp=time.time(),
        )
        self.exchanges.append(ex)
        self.persist()

    def estimated_tokens(self) -> int:
        """Return total estimated tokens across all active session items."""
        if self._last_server_usage and "total_tokens" in self._last_server_usage:
            return self._last_server_usage["total_tokens"]

        # Base prompt + memory tokens
        base_tokens = len(self.config.agent.system_prompt) // 4
        core_tokens = len(self.memory.read_core()) // 4
        summary_tokens = len(self.memory.read_summary()) // 4
        facts_tokens = sum(len(f) for f in self.memory.read_facts()) // 4

        exchanges_tokens = sum(ex.estimated_tokens() for ex in self.exchanges)
        return base_tokens + core_tokens + summary_tokens + facts_tokens + exchanges_tokens

    def needs_rollover(self) -> bool:
        """Triggered when usage reaches threshold (default 80%) of n_ctx."""
        threshold = int(self.config.agent.context_threshold * self.config.server.n_ctx)
        return self.estimated_tokens() >= threshold

    def plan_rollover(self) -> tuple[list[Exchange], list[Exchange]]:
        """
        Split exchanges into:
        - to_consolidate: oldest items roughly totaling 20% of n_ctx
        - to_keep: surviving items (never split an exchange, keep at least the last 2)
        """
        if len(self.exchanges) <= 2:
            return [], list(self.exchanges)

        target_evict_tokens = int(self.config.agent.context_consolidate * self.config.server.n_ctx)

        evicted: list[Exchange] = []
        # We must keep at least the last 2 exchanges
        max_evict_idx = max(0, len(self.exchanges) - 2)
        accum_tokens = 0

        for i in range(max_evict_idx):
            ex = self.exchanges[i]
            evicted.append(ex)
            accum_tokens += ex.estimated_tokens()
            if accum_tokens >= target_evict_tokens:
                break

        surviving = self.exchanges[len(evicted):]
        return evicted, surviving

    def commit_rollover(self, surviving_exchanges: list[Exchange], new_summary: str, new_facts: list[str] | None = None) -> None:
        """Update memory store and surviving exchanges after consolidation."""
        if new_summary:
            existing = self.memory.read_summary()
            combined = f"{existing}\n\n{new_summary}".strip() if existing else new_summary
            self.memory.write_summary(combined)

        if new_facts:
            for fact in new_facts:
                self.memory.add_fact(fact)

        self.exchanges = list(surviving_exchanges)
        self._last_server_usage = None
        self.persist()

    def build_instructions(self) -> str:
        """Construct full system prompt incorporating core memory, summary, and surviving text history."""
        import datetime
        now = datetime.datetime.now()
        now_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

        parts = [self.config.agent.system_prompt]
        parts.append(
            f"\n[Current Time & Date]\n"
            f"The current local date and time is: {now_str}.\n"
            f"When the user asks for the time or date, answer with this exact time."
        )

        core = self.memory.read_core().strip()
        if core:
            parts.append(f"\n[Core Memory]\n{core}")

        summary = self.memory.read_summary().strip()
        if summary:
            parts.append(f"\n[Context Summary]\n{summary}")

        facts = self.memory.read_facts()
        if facts:
            facts_str = "\n".join(f"- {f}" for f in facts[-20:])
            parts.append(f"\n[Known Facts]\n{facts_str}")

        if self.exchanges:
            history_lines = ["\n[Recent Conversation History]"]
            for ex in self.exchanges:
                if ex.user_text:
                    history_lines.append(f"User: {ex.user_text}")
                if ex.assistant_text:
                    history_lines.append(f"Assistant: {ex.assistant_text}")
            parts.append("\n".join(history_lines))

        return "\n\n".join(parts)

    def persist(self) -> None:
        """Persist session exchanges to disk for survival across restarts."""
        try:
            self.memory.ensure_dirs()
            data = [ex.to_dict() for ex in self.exchanges]
            self._persisted_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to persist active session context: %s", e)

    def load_persisted(self) -> bool:
        """Restore exchanges from disk if available."""
        if not self._persisted_file.exists():
            return False
        try:
            raw = self._persisted_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.exchanges = [Exchange.from_dict(d) for d in data]
            logger.info("Restored %d conversation exchanges from %s", len(self.exchanges), self._persisted_file)
            return True
        except Exception as e:
            logger.warning("Failed to load persisted context: %s", e)
            return False
