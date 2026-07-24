import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paimon import compaction
from paimon.session import Session


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _assistant(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


class CompactionHelpersTest(unittest.TestCase):
    def test_threshold_requires_known_window_and_exceeds_reserve(self) -> None:
        self.assertFalse(compaction.should_compact(90, None, 10))
        self.assertFalse(compaction.should_compact(90, 100, 10))
        self.assertTrue(compaction.should_compact(91, 100, 10))

    def test_cut_never_starts_at_tool_result(self) -> None:
        messages = [
            _user("do work"),
            ModelResponse(parts=[ToolCallPart(tool_name="bash", args="{}", tool_call_id="call-1")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="bash", content="result", tool_call_id="call-1")]),
            _assistant("done"),
        ]
        with patch("paimon.compaction.count_tokens", return_value=10):
            self.assertEqual(compaction.find_cut_index(messages, 15), 1)


class CompactTest(unittest.IsolatedAsyncioTestCase):
    async def test_compact_summarizes_prefix_and_keeps_suffix(self) -> None:
        response = ModelResponse(parts=[TextPart(content="## Goal\nKeep going")])
        messages = [
            _user("old request"),
            _assistant("old answer"),
            _user("recent request"),
        ]
        with (
            patch("paimon.compaction.count_tokens", return_value=10),
            patch("paimon.compaction.model_request", new=AsyncMock(return_value=response)),
        ):
            result = await compaction.compact(
                messages,
                model=object(),
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
            old = _user("old")
            recent = _assistant("recent")
            after = _user("after")
            session.append_message(old)
            session.append_message(recent)
            session.append_compaction("checkpoint", [recent], 100)
            session.append_message(after)

            replayed = session.messages()
            self.assertEqual(len(replayed), 3)
            self.assertTrue(compaction.is_summary_message(replayed[0]))
            self.assertIn("checkpoint", replayed[0].parts[0].content)
            self.assertEqual(replayed[1], recent)
            self.assertEqual(replayed[2], after)

            records = Session._read_records(path)
            stored_texts = [str(record.get("message")) for record in records if record.get("type") == "message"]
            self.assertTrue(any("old" in text for text in stored_texts))
            self.assertEqual([record.get("type") for record in records].count("compaction"), 1)


if __name__ == "__main__":
    unittest.main()
