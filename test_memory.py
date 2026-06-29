"""
Unit tests for voicezero cross-call memory (stdlib unittest, no external deps).

Run:  python -m unittest test_memory -v
"""
import hashlib
import importlib
import os
import tempfile
import unittest
from pathlib import Path


class MemoryTest(unittest.TestCase):
    def setUp(self):
        # Pin a known salt BEFORE importing memory so the module-level default is
        # deterministic, and point the store at a throwaway db per test.
        self._tmp = tempfile.TemporaryDirectory()
        self._db = Path(self._tmp.name) / "mem.db"
        os.environ["CALLER_NUMBER_SALT"] = "test-salt-v1"
        os.environ["VOICEZERO_MEMORY_DB"] = str(self._db)
        import memory  # noqa: WPS433
        self.memory = importlib.reload(memory)

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("CALLER_NUMBER_SALT", None)
        os.environ.pop("VOICEZERO_MEMORY_DB", None)

    def _expected_hash(self, number: str, salt: str) -> str:
        # Reference re-implementation of the EXACT dental-receptionist primitive
        # (SHA-256 of number + salt, truncated to 16 hex chars).
        return hashlib.sha256((number + salt).encode("utf-8")).hexdigest()[:16]

    def test_hash_matches_dental_primitive_exactly(self):
        number = "+919876543210"
        self.assertEqual(
            self.memory.hash_caller_number(number),
            self._expected_hash(number, "test-salt-v1"),
        )
        # 16-char lowercase hex, the dental key shape.
        h = self.memory.hash_caller_number(number)
        self.assertEqual(len(h), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_default_salt_parity_with_dental(self):
        # With no CALLER_NUMBER_SALT override, the default MUST equal the
        # dental-receptionist default so the two repos share a key space.
        os.environ.pop("CALLER_NUMBER_SALT", None)
        import memory
        m = importlib.reload(memory)
        number = "+919876543210"
        self.assertEqual(
            m.hash_caller_number(number),
            self._expected_hash(number, "dental-receptionist-v1"),
        )
        # restore for remaining setUp/tearDown symmetry
        os.environ["CALLER_NUMBER_SALT"] = "test-salt-v1"

    def test_empty_number_is_no_caller(self):
        self.assertEqual(self.memory.hash_caller_number(""), "")
        self.assertIsNone(self.memory.recall_by_number(""))

    def test_store_and_recall_round_trip(self):
        h = self.memory.remember(
            "+919811112222",
            summary="booked a Tuesday cleaning",
            outcome="booked",
            extra={"preferred_time": "Tuesday morning"},
        )
        self.assertTrue(self.memory._valid_hash(h))
        rec = self.memory.recall(h)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["summary"], "booked a Tuesday cleaning")
        self.assertEqual(rec["outcome"], "booked")
        self.assertEqual(rec["extra"]["preferred_time"], "Tuesday morning")
        self.assertEqual(rec["call_count"], 1)

    def test_repeat_caller_bumps_count_keeps_first_seen(self):
        number = "+919811113333"
        h = self.memory.remember(number, summary="asked about prices")
        first = self.memory.recall(h)["first_seen"]
        self.memory.remember(number, summary="booked a checkup", outcome="booked")
        rec = self.memory.recall(h)
        self.assertEqual(rec["call_count"], 2)
        self.assertEqual(rec["first_seen"], first)       # first_seen preserved
        self.assertEqual(rec["summary"], "booked a checkup")  # latest context wins

    def test_unknown_caller_returns_none(self):
        # Valid shape but never stored.
        self.assertIsNone(self.memory.recall("0123456789abcdef"))

    def test_invalid_hash_is_rejected_not_sanitized(self):
        # Whitelist over character-removal: a bad hash is a no-op, never cleaned.
        for bad in ("", "nothex", "DEADBEEFDEADBEEF", "0123", "0123456789abcdef0"):
            self.assertFalse(self.memory.store(bad, summary="x"))
            self.assertIsNone(self.memory.recall(bad))

    def test_greeting_note(self):
        self.assertEqual(self.memory.greeting_note("0123456789abcdef"), "")
        h = self.memory.remember("+919800009999", summary="a root canal follow up")
        note = self.memory.greeting_note(h)
        self.assertIn("Welcome back", note)
        self.assertIn("root canal follow up", note)


if __name__ == "__main__":
    unittest.main()
