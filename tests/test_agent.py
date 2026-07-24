import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from helpers import make_session, stub_model
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paimon import compaction
from paimon.agent import (
    Agent,
    CompactionNotice,
    ReasoningDelta,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    UserInput,
    replay_events,
)
from paimon.config import Config


def _config() -> Config:
    return Config(model="test:stub")


class AgentSystemPromptTest(unittest.TestCase):
    def test_system_prompt_is_generated_once_then_loaded_from_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent._system_prompt", return_value="snapshot") as generate,
            ):
                first = Agent(cwd=cwd, config=_config())

            self.assertEqual(first.system_prompt, "snapshot")
            self.assertEqual(session.system_prompt(), "snapshot")
            generate.assert_called_once_with(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                resumed = Agent(cwd=cwd, session=session, config=_config())

            self.assertEqual(resumed.system_prompt, "snapshot")
            generate.assert_not_called()

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                with self.assertRaisesRegex(RuntimeError, "persisted system prompt"):
                    Agent(cwd=cwd, session=session, config=_config())

            generate.assert_not_called()


class MentionAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_content_is_persisted_in_session_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = Agent(cwd=cwd, session=session, config=_config())
                _events = [event async for event in agent.run("review @hello.txt")]

            user_texts = [part.content
                          for message in session.messages() if isinstance(message, ModelRequest)
                          for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(len(user_texts), 1)
            self.assertIn('<mentioned_file data-paimon-mention="1"', user_texts[0])
            self.assertIn("hello", user_texts[0])

    async def test_compaction_forgets_mentions_removed_from_effective_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent(cwd=cwd, session=session, config=_config())
            agent._append_message(ModelRequest(parts=[UserPromptPart(content=agent.mentions.expand("@hello.txt"))]))

            self.assertIn('status="previously_mentioned"', agent.mentions.expand("@hello.txt"))
            result = compaction.CompactionResult("checkpoint", [], 100, 0)
            with (
                patch("paimon.agent.build_model", return_value=stub_model()),
                patch("paimon.agent.compaction.context_window", return_value=100),
                patch("paimon.agent.compaction.count_tokens", side_effect=[100, 10]),
                patch("paimon.agent.compaction.compact", new=AsyncMock(return_value=result)),
            ):
                await agent._maybe_compact()

            expanded = agent.mentions.expand("@hello.txt")
            self.assertIn("hello", expanded)
            self.assertNotIn('status="previously_mentioned"', expanded)


class PermissionModeTest(unittest.IsolatedAsyncioTestCase):
    """The agent consults the gate per tool call: allow skips the confirm hook,
    confirm awaits it. The gate's full decision table is covered in test_tools."""

    @staticmethod
    def _agent(cwd: Path, **kwargs) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent(cwd=cwd, session=session, config=_config(), **kwargs)

    async def _run_tool_turn(self, agent: Agent, name: str, arguments: str) -> ToolEnd:
        agent._cached_model = None  # the agent caches per config; each turn gets a fresh stub
        with patch("paimon.agent.build_model", return_value=stub_model(name, arguments)):
            events = [event async for event in agent.run("go")]
        return next(event for event in events if isinstance(event, ToolEnd))

    async def test_edit_mode_auto_approves_writes_in_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            confirm = AsyncMock(return_value=False)
            agent = self._agent(cwd, confirm=confirm, mode="edit")

            end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')

            confirm.assert_not_awaited()
            self.assertFalse(end.denied)
            self.assertEqual((cwd / "a.txt").read_text(), "hi")

    async def test_mode_switch_applies_to_the_next_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            confirm = AsyncMock(return_value=False)
            agent = self._agent(cwd, confirm=confirm, mode="read")

            end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')
            confirm.assert_awaited_once()
            self.assertTrue(end.denied)
            self.assertFalse((cwd / "a.txt").exists())

            agent.mode = "yolo"
            end = await self._run_tool_turn(agent, "write_file", '{"path": "a.txt", "content": "hi"}')
            confirm.assert_awaited_once()
            self.assertFalse(end.denied)
            self.assertEqual((cwd / "a.txt").read_text(), "hi")


class ReplayEventsTest(unittest.TestCase):
    """History replays as the same event sequence a live run would yield."""

    def test_messages_replay_as_live_events(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="do it")]),
            ModelResponse(parts=[
                ThinkingPart(content="planning"),
                TextPart(content="ok"),
                ToolCallPart(tool_name="bash", args='{"command": "ls"}', tool_call_id="c1"),
                ToolCallPart(tool_name="write_todos",
                             args='{"todos": [{"content": "x", "status": "pending"}]}', tool_call_id="c2"),
            ]),
            ModelRequest(parts=[
                ToolReturnPart(tool_name="bash", content="a.py", tool_call_id="c1"),
                ToolReturnPart(tool_name="write_todos", content="[ ] x", tool_call_id="c2"),
            ]),
            ModelResponse(parts=[TextPart(content="done")]),
        ]

        events = replay_events(messages)

        self.assertEqual(
            [type(event) for event in events],
            [UserInput, ReasoningDelta, TextDelta, ToolStart, TodosUpdate, ToolEnd, TextDelta],
        )
        tool_end = events[5]
        self.assertEqual((tool_end.id, tool_end.name, tool_end.result), ("c1", "bash", "a.py"))

    def test_compaction_summary_becomes_notice(self) -> None:
        events = replay_events([
            compaction.summary_message("checkpoint"),
            ModelRequest(parts=[UserPromptPart(content="hi")]),
        ])
        self.assertEqual([type(event) for event in events], [CompactionNotice, UserInput])


if __name__ == "__main__":
    unittest.main()
