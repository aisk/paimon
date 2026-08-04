import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from helpers import make_session, stub_model
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paimon import compaction, tools
from paimon.agent import (
    Agent,
    CompactionNotice,
    ReasoningDelta,
    SessionHandoff,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    replay_events,
)
from paimon.config import Config
from paimon.session import is_summary_message, summary_message


def _config() -> Config:
    return Config(model="test:stub")


class AgentSystemPromptTest(unittest.TestCase):
    def test_system_prompt_is_generated_once_then_loaded_from_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="snapshot") as generate,
            ):
                first = Agent.open(cwd=cwd, config=_config())

            self.assertEqual(first.system_prompt, "snapshot")
            self.assertEqual(session.system_prompt(), "snapshot")
            generate.assert_called_once_with(cwd)

            with patch("paimon.agent.build_system_prompt") as generate:
                resumed = Agent.open(cwd=cwd, session=session, config=_config())

            self.assertEqual(resumed.system_prompt, "snapshot")
            generate.assert_not_called()

    def test_append_system_prompt_extends_a_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="base"),
            ):
                agent = Agent.open(cwd=cwd, config=_config(),
                                   append_system_prompt="  You are a reviewer.  ")

            self.assertEqual(agent.system_prompt, "base\n\nYou are a reviewer.")
            self.assertEqual(session.system_prompt(), "base\n\nYou are a reviewer.")

    def test_append_system_prompt_on_resume_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with self.assertRaisesRegex(ValueError, "new session"):
                Agent.open(cwd=cwd, session=session, config=_config(),
                           append_system_prompt="role")
            self.assertEqual(session.system_prompt(), "snapshot")

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent.build_system_prompt") as generate:
                with self.assertRaisesRegex(RuntimeError, "persisted system prompt"):
                    Agent.open(cwd=cwd, session=session, config=_config())

            generate.assert_not_called()


class MentionAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_content_is_persisted_in_session_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                _events = [event async for event in agent.run("review @hello.txt")]

            user_texts = [part.content
                          for message in session.messages() if isinstance(message, ModelRequest)
                          for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(len(user_texts), 1)
            self.assertIn('<mentioned_file path=', user_texts[0])
            self.assertIn("hello", user_texts[0])


class PermissionModeTest(unittest.IsolatedAsyncioTestCase):
    """The agent consults the gate per tool call: allow skips the confirm hook,
    confirm awaits it. The gate's full decision table is covered in test_tools."""

    @staticmethod
    def _agent(cwd: Path, **kwargs) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=_config(), **kwargs)

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

    async def test_read_only_commands_run_unless_the_config_says_strict(self) -> None:
        """The safe_commands setting has to reach run_tool, not just the gate."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            confirm = AsyncMock(return_value=False)
            agent = self._agent(cwd, confirm=confirm, mode="read")

            end = await self._run_tool_turn(agent, "shell", '{"command": "pwd"}')
            confirm.assert_not_awaited()
            self.assertFalse(end.denied)

            agent.config.safe_commands = False
            end = await self._run_tool_turn(agent, "shell", '{"command": "pwd"}')
            confirm.assert_awaited_once()
            self.assertTrue(end.denied)


class TodosEventShapeTest(unittest.IsolatedAsyncioTestCase):
    async def test_write_todos_yields_only_a_todos_update(self) -> None:
        """No ToolStart/ToolEnd for write_todos, matching what replay produces —
        renderers can then treat every ToolStart/ToolEnd the same way."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            with patch("paimon.agent.build_model", return_value=stub_model("write_todos", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go")]

            self.assertFalse([e for e in events if isinstance(e, (ToolStart, ToolEnd))])
            todos = next(e for e in events if isinstance(e, TodosUpdate))
            self.assertEqual(todos.todos, [{"content": "x", "status": "pending"}])
            self.assertEqual(agent.todos, todos.todos)


class AgentToolsetTest(unittest.IsolatedAsyncioTestCase):
    """A per-agent toolset narrows both what the model sees and what may run."""

    async def test_only_the_toolset_is_offered_and_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config(), mode="yolo",
                               toolset={"read_file": tools.REGISTRY["read_file"]})

            offered: list[list[str]] = []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                offered.append([tool.name for tool in info.function_tools])
                if requests == 1:
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")]

            self.assertEqual(offered[0], ["read_file"])
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("unknown tool", end.result)
            self.assertIsInstance(events[-1], TurnEnd)


class SessionHandoffTest(unittest.IsolatedAsyncioTestCase):
    """start_new_session ends the turn on approval without another model
    request; without a confirm hook it is denied even in yolo mode."""

    @staticmethod
    def _agent(cwd: Path, **kwargs) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=_config(), **kwargs)

    @staticmethod
    async def _run(agent: Agent, arguments: str = '{"prompt": "next phase"}') -> list:
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", arguments)):
            return [event async for event in agent.run("go")]

    async def test_approval_ends_the_turn_with_a_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=True)
            agent = self._agent(cwd, confirm=confirm, mode="yolo")

            events = await self._run(agent)

            confirm.assert_awaited_once()
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertFalse(end.denied)
            self.assertIsInstance(events[-1], SessionHandoff)
            self.assertEqual(events[-1].prompt, "next phase")
            self.assertFalse([e for e in events if isinstance(e, (TurnEnd, TextDelta))],
                             "no second model request after the handoff")
            returns = [part for message in agent.session.messages()
                       if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, ToolReturnPart)]
            self.assertIn("Handoff accepted", returns[-1].content)

    async def test_denial_continues_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=False)
            agent = self._agent(cwd, confirm=confirm, mode="read")

            events = await self._run(agent)

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertTrue(end.denied)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertTrue([e for e in events if isinstance(e, TextDelta)])
            self.assertIsInstance(events[-1], TurnEnd)

    async def test_without_confirm_hook_denied_even_in_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            agent = self._agent(cwd, mode="yolo")

            events = await self._run(agent)

            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertTrue(end.denied)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertIsInstance(events[-1], TurnEnd)

    async def test_empty_prompt_is_an_error_without_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            confirm = AsyncMock(return_value=True)
            agent = self._agent(cwd, confirm=confirm, mode="read")

            events = await self._run(agent, '{"prompt": "  "}')

            confirm.assert_not_awaited()
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("prompt is required", end.result)
            self.assertFalse([e for e in events if isinstance(e, SessionHandoff)])
            self.assertIsInstance(events[-1], TurnEnd)


class ReplayEventsTest(unittest.TestCase):
    """History replays as the same event sequence a live run would yield."""

    def test_messages_replay_as_live_events(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="do it")]),
            ModelResponse(parts=[
                ThinkingPart(content="planning"),
                TextPart(content="ok"),
                ToolCallPart(tool_name="shell", args='{"command": "ls"}', tool_call_id="c1"),
                ToolCallPart(tool_name="write_todos",
                             args='{"todos": [{"content": "x", "status": "pending"}]}', tool_call_id="c2"),
            ]),
            ModelRequest(parts=[
                ToolReturnPart(tool_name="shell", content="a.py", tool_call_id="c1"),
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
        self.assertEqual((tool_end.id, tool_end.name, tool_end.result), ("c1", "shell", "a.py"))

    def test_compaction_summary_becomes_notice(self) -> None:
        events = replay_events([
            summary_message("checkpoint"),
            ModelRequest(parts=[UserPromptPart(content="hi")]),
        ])
        self.assertEqual([type(event) for event in events], [CompactionNotice, UserInput])


class ManualCompactionTest(unittest.IsolatedAsyncioTestCase):
    async def test_compact_now_ignores_the_toggle_and_the_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="snapshot"),
            ):
                # A tiny history under a disabled auto-compaction config: the
                # automatic path would decline on both counts.
                agent = Agent.open(cwd=cwd, config=Config(model="test:stub", compaction_enabled=False))
            old = ModelRequest(parts=[UserPromptPart(content="old")])
            recent = ModelResponse(parts=[TextPart(content="recent")])
            agent._append_message(old)
            agent._append_message(recent)

            result = compaction.CompactionResult("checkpoint", [recent], 100, 0)
            with (
                patch("paimon.agent.Agent._model", return_value=object()),
                patch("paimon.compaction.compact", new=AsyncMock(return_value=result)) as compact,
            ):
                self.assertIsNone(await agent._maybe_compact())
                returned = await agent.compact_now()

            self.assertIs(returned, result)
            compact.assert_awaited_once()
            self.assertTrue(is_summary_message(agent.history[0]))
            self.assertEqual(agent.history[1:], [recent])
            # the checkpoint is persisted, so a resume replays the same context
            replayed = session.messages()
            self.assertEqual(len(replayed), 2)
            self.assertIn("checkpoint", replayed[0].parts[0].content)


if __name__ == "__main__":
    unittest.main()
