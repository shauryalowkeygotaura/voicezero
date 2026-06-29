"""
voicezero cross-call memory: a tiny, stdlib-only SQLite store keyed by caller_hash.

Why this exists
---------------
A live agent that recognizes a repeat caller ("welcome back, last time we booked
you for a Tuesday cleaning") feels human and saves the caller from repeating
themselves. To do that we need a place to remember the last context of each
caller across separate calls, keyed by a stable but privacy-preserving id.

Privacy + key-space parity
--------------------------
We never store the raw phone number. The key is the SHA-256 of the number plus a
per-deployment salt, truncated to 16 hex chars. This is the EXACT same primitive
used by the dental-receptionist webhook (webhook/_lib/parser.py:hash_caller_number),
so a caller_hash produced here lands in the same key space as the one the dental
telemetry already records. The salt comes from CALLER_NUMBER_SALT (Doppler);
the default fallback string is kept identical to dental-receptionist on purpose,
so the two repos agree even before Doppler injects an override.

Cost + dependencies
-------------------
Pure standard library (sqlite3 + hashlib). No pip install, no network, no cost.
Fits the $0.00 per minute guarantee of the rest of voicezero.

Public API
----------
    hash_caller_number(raw_number) -> caller_hash ("" for empty input)
    store(caller_hash, summary=..., outcome=..., extra=...) -> bool
    remember(raw_number, ...) -> caller_hash         # hash + store in one call
    recall(caller_hash) -> dict | None               # None for unknown caller
    recall_by_number(raw_number) -> dict | None
    greeting_note(caller_hash) -> str                # short spoken-friendly line
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

# Same default as dental-receptionist (webhook/_lib/parser.py). Keep these two in
# lock-step: identical default + identical CALLER_NUMBER_SALT (from Doppler) is
# what makes a caller_hash from either repo refer to the same caller.
_DEFAULT_PHONE_SALT = "dental-receptionist-v1"
_PHONE_SALT = os.getenv("CALLER_NUMBER_SALT", _DEFAULT_PHONE_SALT)

# Tracks whether we've already warned about the default salt, so a live agent
# logs the privacy notice exactly once instead of on every hash.
_default_salt_warned = False


def _warn_if_default_salt() -> None:
    """
    Log a single, clear WARNING when CALLER_NUMBER_SALT resolves to the shared
    default. The default is a publicly known string, so caller hashes built with
    it are guessable: anyone can hash a candidate phone number with the default
    salt and confirm whether it matches a stored hash. The per-deployment salt is
    what makes the hash a one-way, privacy-preserving id, so the warning tells the
    operator to set a real secret salt (kept in Doppler / a gitignored .env, never
    committed). We never log the salt value itself.
    """
    global _default_salt_warned
    if _default_salt_warned or _PHONE_SALT != _DEFAULT_PHONE_SALT:
        return
    _default_salt_warned = True
    logger.warning(
        "CALLER_NUMBER_SALT is unset, so caller-hash privacy is using the shared "
        "default salt. Caller hashes are therefore guessable and NOT a real privacy "
        "guarantee. Set a unique secret CALLER_NUMBER_SALT (via Doppler or a "
        "gitignored .env) before handling real caller numbers."
    )

# A caller_hash is exactly the first 16 chars of a sha256 hexdigest. We accept
# only that shape (whitelist, not character-stripping): anything else is treated
# as "no caller" and quietly ignored, never sanitized-then-used.
_HASH_RE = re.compile(r"^[0-9a-f]{16}$")

# DB lives next to the agent by default; override for tests / a shared server.
DB_PATH = Path(os.getenv("VOICEZERO_MEMORY_DB", str(HERE / "caller_memory.db")))


def hash_caller_number(raw_number: str) -> str:
    """
    SHA-256(number + salt), truncated to 16 hex chars. Byte-for-byte identical to
    dental-receptionist's hash_caller_number so the key spaces match. Empty input
    yields "" (no caller), never a hash of the empty string.
    """
    if not raw_number:
        return ""
    _warn_if_default_salt()
    h = hashlib.sha256()
    h.update((raw_number + _PHONE_SALT).encode("utf-8"))
    return h.hexdigest()[:16]


def _valid_hash(caller_hash: str) -> bool:
    return bool(caller_hash) and bool(_HASH_RE.match(caller_hash))


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS caller_memory (
            caller_hash TEXT PRIMARY KEY,
            first_seen  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            call_count  INTEGER NOT NULL DEFAULT 1,
            summary     TEXT NOT NULL DEFAULT '',
            outcome     TEXT NOT NULL DEFAULT '',
            extra       TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    return conn


def store(
    caller_hash: str,
    summary: str = "",
    outcome: str = "",
    extra: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> bool:
    """
    Upsert one caller's last context. Returns True if written, False if the hash
    is not a valid caller_hash (then it is a no-op, never an exception, so a live
    call never crashes on a missing or malformed number).

    On a repeat caller we bump call_count and refresh last_seen, keeping the very
    first first_seen. Parameterized SQL throughout: caller data is never
    concatenated into a statement.
    """
    if not _valid_hash(caller_hash):
        return False
    now = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(extra or {}, ensure_ascii=False)
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO caller_memory
                    (caller_hash, first_seen, last_seen, call_count, summary, outcome, extra)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(caller_hash) DO UPDATE SET
                    last_seen  = excluded.last_seen,
                    call_count = caller_memory.call_count + 1,
                    summary    = excluded.summary,
                    outcome    = excluded.outcome,
                    extra      = excluded.extra
                """,
                (caller_hash, now, now, summary, outcome, payload),
            )
        return True
    finally:
        conn.close()


def remember(
    raw_number: str,
    summary: str = "",
    outcome: str = "",
    extra: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> str:
    """Hash a raw number then store its context. Returns the caller_hash, or ""
    if the number was empty (nothing stored)."""
    caller_hash = hash_caller_number(raw_number)
    if caller_hash:
        store(caller_hash, summary=summary, outcome=outcome, extra=extra, db_path=db_path)
    return caller_hash


def recall(caller_hash: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Return a known caller's last context, or None for an unknown/invalid hash.
    `extra` is decoded back to a dict; a corrupt blob degrades to {} rather than
    raising, so a bad row can never break a greeting."""
    if not _valid_hash(caller_hash):
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM caller_memory WHERE caller_hash = ?", (caller_hash,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        extra = json.loads(row["extra"])
        if not isinstance(extra, dict):
            extra = {}
    except (json.JSONDecodeError, TypeError):
        extra = {}
    return {
        "caller_hash": row["caller_hash"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "call_count": row["call_count"],
        "summary": row["summary"],
        "outcome": row["outcome"],
        "extra": extra,
    }


def recall_by_number(raw_number: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Convenience: hash a raw number, then recall its stored context."""
    return recall(hash_caller_number(raw_number), db_path=db_path)


def greeting_note(caller_hash: str, db_path: Path | None = None) -> str:
    """
    A short, spoken-friendly sentence the agent can prepend to its first message
    for a returning caller, or "" for a first-time/unknown caller. Kept to one
    line with no lists or symbols so it drops straight into the voice path.
    """
    rec = recall(caller_hash, db_path=db_path)
    if not rec:
        return ""
    summary = (rec.get("summary") or "").strip()
    if summary:
        return f"Welcome back. Last time we spoke about: {summary}"
    return "Welcome back, good to hear from you again."
