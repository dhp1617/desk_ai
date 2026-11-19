"""SQLite FTS5 full-text search index for memory files."""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class MemorySearch:
    """Full-text search over memory files using SQLite FTS5."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self._db_path = memory_dir / ".search.db"
        self._conn: sqlite3.Connection | None = None
        self._file_mtimes: dict[str, float] = {}

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_tables()
        return self._conn

    def _init_tables(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(file, content, tokenize='porter unicode61')
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_meta (
                file TEXT PRIMARY KEY,
                mtime REAL
            )
        """)
        conn.commit()

    def reindex(self) -> None:
        """Full reindex of all .md files."""
        conn = self._get_conn()

        # Collect all indexable files
        files: list[Path] = []
        for path in self.memory_dir.rglob("*.md"):
            if path.name.startswith("."):
                continue
            files.append(path)

        # Clear and rebuild
        conn.execute("DELETE FROM memory_fts")
        conn.execute("DELETE FROM file_meta")

        for path in files:
            rel = str(path.relative_to(self.memory_dir))
            try:
                content = path.read_text(encoding="utf-8")
                mtime = path.stat().st_mtime
                conn.execute(
                    "INSERT INTO memory_fts(file, content) VALUES (?, ?)",
                    (rel, content),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO file_meta(file, mtime) VALUES (?, ?)",
                    (rel, mtime),
                )
                self._file_mtimes[rel] = mtime
            except Exception:
                logger.exception("Failed to index %s", path)

        conn.commit()
        logger.info("Indexed %d memory files", len(files))

    def sync_if_needed(self) -> None:
        """Check for changed files and reindex only those."""
        conn = self._get_conn()

        # Load stored mtimes
        if not self._file_mtimes:
            rows = conn.execute("SELECT file, mtime FROM file_meta").fetchall()
            self._file_mtimes = {r[0]: r[1] for r in rows}

        changed = False
        current_files: set[str] = set()

        for path in self.memory_dir.rglob("*.md"):
            if path.name.startswith("."):
                continue
            rel = str(path.relative_to(self.memory_dir))
            current_files.add(rel)

            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue

            if rel not in self._file_mtimes or self._file_mtimes[rel] < mtime:
                # File changed or new
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception:
                    continue

                conn.execute("DELETE FROM memory_fts WHERE file = ?", (rel,))
                conn.execute(
                    "INSERT INTO memory_fts(file, content) VALUES (?, ?)",
                    (rel, content),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO file_meta(file, mtime) VALUES (?, ?)",
                    (rel, mtime),
                )
                self._file_mtimes[rel] = mtime
                changed = True

        # Remove deleted files
        for rel in list(self._file_mtimes.keys()):
            if rel not in current_files:
                conn.execute("DELETE FROM memory_fts WHERE file = ?", (rel,))
                conn.execute("DELETE FROM file_meta WHERE file = ?", (rel,))
                del self._file_mtimes[rel]
                changed = True

        if changed:
            conn.commit()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search indexed files. Returns list of {file, snippet, rank}."""
        conn = self._get_conn()

        try:
            rows = conn.execute(
                """
                SELECT file, snippet(memory_fts, 1, '>>>', '<<<', '...', 64), rank
                FROM memory_fts
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS search error: %s", e)
            # Fallback to simple LIKE search
            rows = conn.execute(
                """
                SELECT file, substr(content, 1, 200), 0
                FROM memory_fts
                WHERE content LIKE ?
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()

        return [
            {"file": r[0], "snippet": r[1], "rank": r[2]}
            for r in rows
        ]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
