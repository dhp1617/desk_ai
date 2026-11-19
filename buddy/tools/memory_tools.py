"""Memory tools using the tool registry."""
from __future__ import annotations

import logging
from typing import Annotated

from buddy.tools.registry import tool

logger = logging.getLogger(__name__)

_store = None  # MemoryStore
_search = None  # MemorySearch


def init_memory_tools(store, search) -> None:
    global _store, _search
    _store = store
    _search = search


@tool
def remember(fact: Annotated[str, "Fact or information to remember long-term"]) -> str:
    """Remember a fact for the long term. Stored in facts.md."""
    if _store is None:
        return "Error: Memory not initialized."
    _store.add_fact(fact)
    if _search:
        _search.sync_if_needed()
    return f"Remembered: {fact}"


@tool
def search_memory(query: Annotated[str, "Search query for memory"]) -> str:
    """Search through all memory files for relevant information."""
    if _search is None:
        return "Error: Memory search not initialized."
    _search.sync_if_needed()
    results = _search.search(query)
    if not results:
        return "No matching memories found."
    lines = ["Memory search results:"]
    for r in results:
        lines.append(f"  [{r['file']}] {r['snippet']}")
    return "\n".join(lines)


@tool
def forget(pattern: Annotated[str, "Text pattern to match for removal from facts"]) -> str:
    """Remove facts matching a pattern from long-term memory."""
    if _store is None:
        return "Error: Memory not initialized."
    removed = _store.remove_fact(pattern)
    if removed:
        if _search:
            _search.sync_if_needed()
        return f"Removed facts matching '{pattern}'."
    return f"No facts matching '{pattern}' found."


@tool
def read_memory(
    filename: Annotated[str, "File to read: core.md, facts.md, summary.md, timetable.yaml, mess.yaml, or notes/YYYY-MM-DD.md"],
) -> str:
    """Read a specific memory file."""
    if _store is None:
        return "Error: Memory not initialized."
    return _store.read_file_safe(filename)
