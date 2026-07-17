import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from paimon import compaction
from paimon.agent import Agent
from paimon.mentions import MentionExpander
from paimon.session import Session


class AgentSystemPromptTest(unittest.TestCase):
    @staticmethod
    def _session(cwd: Path) -> Session:
        session = Session(cwd / "session.jsonl", "session-id", cwd)
        session.append({
            "type": "session",
            "version": 1,
            "id": "session-id",
            "cwd": str(cwd),
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        return session

    def test_system_prompt_is_generated_once_then_loaded_from_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent._system_prompt", return_value="snapshot") as generate,
            ):
                first = Agent(cwd=cwd)

            self.assertEqual(first.messages[0], {"role": "system", "content": "snapshot"})
            self.assertEqual(session.system_prompt(), "snapshot")
            generate.assert_called_once_with(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                resumed = Agent(cwd=cwd, session=session)

            self.assertEqual(resumed.messages[0], {"role": "system", "content": "snapshot"})
            generate.assert_not_called()

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                with self.assertRaisesRegex(RuntimeError, "persisted system prompt"):
                    Agent(cwd=cwd, session=session)

            generate.assert_not_called()


class MentionExpanderTest(unittest.TestCase):
    def test_expands_first_version_and_references_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("hello\n")
            expander = MentionExpander(cwd)

            first = expander.expand("review @hello.txt")
            second = expander.expand("review @hello.txt")

            self.assertIn('exposure="full"', first)
            self.assertIn("hello", first)
            self.assertIn('status="previously_mentioned"', second)
            self.assertNotIn("\nhello\n", second)

    def test_changed_file_is_a_new_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("first\n")
            expander = MentionExpander(cwd)
            first = expander.expand("@hello.txt")
            target.write_text("second\n")
            second = expander.expand("@hello.txt")

            self.assertIn("first", first)
            self.assertIn("second", second)
            self.assertNotIn('status="previously_mentioned"', second)

    def test_escaped_space_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "with space.txt").write_text("contents")
            expander = MentionExpander(cwd)

            expanded = expander.expand("@with\\ space.txt @missing.txt")

            self.assertIn("contents", expanded)
            self.assertIn('requested="missing.txt" status="not_found"', expanded)

    def test_restores_versions_from_persisted_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("hello")
            original = MentionExpander(cwd).expand("@hello.txt")

            resumed = MentionExpander(cwd, [{"role": "user", "content": original}])
            repeated = resumed.expand("@hello.txt")

            self.assertIn('status="previously_mentioned"', repeated)


class MentionAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_content_is_persisted_in_session_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = AgentSystemPromptTest._session(cwd)
            session.append_system_prompt("snapshot")

            async def completion(**_kwargs):
                async def stream():
                    delta = SimpleNamespace(content=None, tool_calls=[], reasoning_content=None)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

                return stream()

            with patch("paimon.agent.litellm.acompletion", new=completion):
                agent = Agent(cwd=cwd, session=session)
                _events = [event async for event in agent.run("review @hello.txt")]

            user_messages = [message for message in session.messages() if message.get("role") == "user"]
            self.assertEqual(len(user_messages), 1)
            self.assertIn('<mentioned_file data-paimon-mention="1"', user_messages[0]["content"])
            self.assertIn("hello", user_messages[0]["content"])

    async def test_compaction_forgets_mentions_removed_from_effective_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = AgentSystemPromptTest._session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent(cwd=cwd, session=session)
            agent._append_message({"role": "user", "content": agent.mentions.expand("@hello.txt")})

            self.assertIn('status="previously_mentioned"', agent.mentions.expand("@hello.txt"))
            result = compaction.CompactionResult("checkpoint", [], 100, 0)
            with (
                patch("paimon.agent.config.COMPACTION_ENABLED", True),
                patch("paimon.agent.compaction.context_window", return_value=100),
                patch("paimon.agent.compaction.count_tokens", side_effect=[100, 10]),
                patch("paimon.agent.compaction.compact", new=AsyncMock(return_value=result)),
            ):
                await agent._maybe_compact()

            expanded = agent.mentions.expand("@hello.txt")
            self.assertIn("hello", expanded)
            self.assertNotIn('status="previously_mentioned"', expanded)

if __name__ == "__main__":
    unittest.main()
