import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from paimon import compaction
from paimon.session import Session


class CompactionHelpersTest(unittest.TestCase):
    def test_threshold_requires_known_window_and_exceeds_reserve(self) -> None:
        self.assertFalse(compaction.should_compact(90, None, 10))
        self.assertFalse(compaction.should_compact(90, 100, 10))
        self.assertTrue(compaction.should_compact(91, 100, 10))

    def test_cut_never_starts_at_tool_result(self) -> None:
        messages = [
            {"role": "user", "content": "do work"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        with patch("paimon.compaction.count_tokens", return_value=10):
            self.assertEqual(compaction.find_cut_index(messages, 15, "test/model"), 1)


class CompactTest(unittest.IsolatedAsyncioTestCase):
    async def test_compact_summarizes_prefix_and_keeps_suffix(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="## Goal\nKeep going"))]
        )
        messages = [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent request"},
        ]
        with (
            patch("paimon.compaction.count_tokens", return_value=10),
            patch("paimon.compaction.litellm.acompletion", new=AsyncMock(return_value=response)),
        ):
            result = await compaction.compact(
                messages,
                model="test/model",
                api_base=None,
                api_key=None,
                keep_recent_tokens=15,
                tokens_before=100,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.summary, "## Goal\nKeep going")
        self.assertEqual(result.kept_messages, messages[1:])


class SessionCompactionTest(unittest.TestCase):
    def test_session_replays_from_latest_compaction_but_keeps_old_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.jsonl"
            session = Session(path, "session-id", Path(directory))
            old = {"role": "user", "content": "old"}
            recent = {"role": "assistant", "content": "recent"}
            after = {"role": "user", "content": "after"}
            session.append_message(old)
            session.append_message(recent)
            session.append_compaction("checkpoint", [recent], 100)
            session.append_message(after)

            self.assertEqual(
                session.messages(),
                [compaction.summary_message("checkpoint"), recent, after],
            )
            records = Session._read_records(path)
            self.assertIn(old, [record.get("message") for record in records])
            self.assertEqual([record.get("type") for record in records].count("compaction"), 1)


if __name__ == "__main__":
    unittest.main()
