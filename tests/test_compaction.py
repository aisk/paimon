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
from paimon.model_windows import CONTEXT_WINDOWS
from paimon.session import Session, agents_message, is_summary_message


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _assistant(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


class ContextWindowTest(unittest.TestCase):
    """Windows come from the generated table, with the fragments as a backstop."""

    def test_override_beats_the_tables(self) -> None:
        self.assertEqual(compaction.context_window("zai:glm-4.6", 42_000), 42_000)

    def test_the_generated_table_answers_by_name(self) -> None:
        # Asserted against the table rather than a literal, so regenerating it
        # against a newer catalogue does not break the test.
        self.assertEqual(compaction.context_window("zai:glm-4.6"),
                         CONTEXT_WINDOWS["glm-4.6"])
        self.assertEqual(compaction.context_window("anthropic:claude-sonnet-4-5"),
                         CONTEXT_WINDOWS["claude-sonnet-4-5"])
        # Everything past the provider prefix is looked up as one name, so
        # Bedrock's region prefix and its trailing ":0" revision survive.
        name = "us.anthropic.claude-opus-4-1-20250805-v1:0"
        self.assertEqual(compaction.context_window(f"bedrock:{name}"), CONTEXT_WINDOWS[name])

    def test_an_exact_name_beats_its_family_fragment(self) -> None:
        # The 'glm' fragment would coarsely answer 128k for every GLM.
        self.assertGreater(compaction.context_window("zai:glm-4.6"), 128_000)

    def test_models_the_table_never_heard_of_fall_back_to_fragments(self) -> None:
        self.assertNotIn("glm-9-experimental", CONTEXT_WINDOWS)
        self.assertEqual(compaction.context_window("zai:glm-9-experimental"), 128_000)
        self.assertNotIn("claude-unreleased-9", CONTEXT_WINDOWS)
        self.assertEqual(compaction.context_window("anthropic:claude-unreleased-9"), 200_000)

    def test_unknown_model_and_no_override_disable_compaction(self) -> None:
        self.assertIsNone(compaction.context_window("acme:mystery-1"))
        self.assertIsNone(compaction.context_window(None))
        self.assertIsNone(compaction.context_window("acme:mystery-1", 0))


class CompactionHelpersTest(unittest.TestCase):
    def test_count_includes_the_system_prompt(self) -> None:
        messages = [_user("hi")]
        bare = compaction.count_tokens(messages)
        with_prompt = compaction.count_tokens(messages, system_prompt="x" * 4_000)
        self.assertGreaterEqual(with_prompt - bare, 900)

    def test_threshold_requires_known_window_and_exceeds_reserve(self) -> None:
        self.assertFalse(compaction.should_compact(90, None, 10))
        self.assertFalse(compaction.should_compact(90, 100, 10))
        self.assertTrue(compaction.should_compact(91, 100, 10))

    def test_cut_never_starts_at_tool_result(self) -> None:
        messages = [
            _user("do work"),
            ModelResponse(parts=[ToolCallPart(tool_name="shell", args="{}", tool_call_id="call-1")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="shell", content="result", tool_call_id="call-1")]),
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


class BoundedSummaryInputTest(unittest.TestCase):
    """COMPACT-1: the summary request must itself fit the model's window."""

    def test_small_input_passes_through(self) -> None:
        self.assertEqual(compaction._bounded("line1\nline2", 128_000), "line1\nline2")

    def test_oversized_input_keeps_head_and_tail(self) -> None:
        lines = [f"msg-{i:06d} " + "x" * 90 for i in range(2_000)]
        serialized = "\n".join(lines)
        window = 8_192
        bounded = compaction._bounded(serialized, window)
        budget_chars = max(8_000, (window - 2_048 - 2_000) * 4)
        self.assertLessEqual(len(bounded), budget_chars + 200)
        self.assertIn("msg-000000", bounded, "the goal at the start survives")
        self.assertIn("msg-001999", bounded, "the recent work at the end survives")
        self.assertIn("omitted", bounded)

    def test_unknown_window_uses_the_default_budget(self) -> None:
        bounded = compaction._bounded("y" * 1_000_000, None)
        self.assertLess(len(bounded), 60_000 * 4 + 500)

    def test_compact_sends_a_bounded_prompt(self) -> None:
        import asyncio

        response = ModelResponse(parts=[TextPart(content="## Goal\nsummary")])
        request_mock = AsyncMock(return_value=response)
        messages = [_user("goal statement " + "x" * 500_000),
                    _assistant("middle answer"),
                    _user("recent request")]
        with (
            patch("paimon.compaction.count_tokens", return_value=10),
            patch("paimon.compaction.model_request", new=request_mock),
        ):
            asyncio.run(compaction.compact(
                messages, model=object(), keep_recent_tokens=15,
                tokens_before=100, window=8_192,
            ))
        sent = request_mock.await_args.args[1][0].parts[1].content
        self.assertLess(len(sent), 30_000)
        self.assertIn("omitted", sent)


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
            self.assertTrue(is_summary_message(replayed[0]))
            self.assertIn("checkpoint", replayed[0].parts[0].content)
            self.assertEqual(replayed[1], recent)
            self.assertEqual(replayed[2], after)

            records = Session._read_records(path)
            stored_texts = [str(record.get("message")) for record in records if record.get("type") == "message"]
            self.assertTrue(any("old" in text for text in stored_texts))
            self.assertEqual([record.get("type") for record in records].count("compaction"), 1)


if __name__ == "__main__":
    unittest.main()


class AgentStatusExclusionTest(unittest.TestCase):
    def test_agent_status_lines_never_reach_the_summary_prompt(self) -> None:
        """The prompt asks for current status, so a checkpoint would carry an
        hours-old "a1f2 finished" forward for the rest of the session."""
        serialized = compaction._serialize_messages([
            _user("real request"),
            agents_message("a1f2 finished"),
            _assistant("answer"),
        ])

        self.assertNotIn("a1f2 finished", serialized)
        self.assertEqual(len(serialized.splitlines()), 2)
