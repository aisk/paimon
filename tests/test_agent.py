import asyncio
import gc
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

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

from paimon import compaction, lockfile, tools
from paimon.agent import (
    Agent,
    AgentsNotice,
    CompactionNotice,
    ContextCompactionFailed,
    ReasoningDelta,
    RequestStats,
    SessionHandoff,
    TextDelta,
    TodosUpdate,
    ToolBudgetExhausted,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    replay_events,
)
from paimon.config import Config
from paimon.session import (
    Session,
    SessionIncompleteError,
    is_agents_message,
    is_summary_message,
    summary_message,
)


def _config() -> Config:
    return Config(model="test:stub")


def _records(session: Session) -> list[dict]:
    """Every raw record in the session log, in order."""
    return [json.loads(line) for line in
            session.path.read_text(encoding="utf-8").splitlines()]


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

            first.session.unlock()  # a session is only ever open once at a time
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

    def test_prompt_does_not_enumerate_tool_names(self) -> None:
        # Schemas travel with every request; a prose list would drift from a
        # narrowed toolset.
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent.Session.create", return_value=session):
                agent = Agent.open(cwd=cwd, config=_config(),
                                   toolset={"read_file": tools.REGISTRY["read_file"]})

            self.assertNotIn("You have these tools", agent.system_prompt)

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent.build_system_prompt") as generate:
                with self.assertRaisesRegex(SessionIncompleteError, "persisted system prompt"):
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


class RequestStatsTest(unittest.IsolatedAsyncioTestCase):
    async def test_each_request_yields_stats_from_reported_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("read_file", '{"path": "missing.txt"}')):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("hi")]

            stats = [event for event in events if isinstance(event, RequestStats)]
            # one per model request: the tool-call response and the final text
            self.assertEqual(len(stats), 2)
            for stat in stats:
                self.assertGreater(stat.output_tokens, 0)
                self.assertGreater(stat.seconds, 0)


class HistoryToolWiringTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_history_reaches_the_agents_own_session_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("search_history", '{"query": "avocado"}')):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("the codeword is avocado")]

            end = next(event for event in events if isinstance(event, ToolEnd))
            self.assertIn("matching part", end.result)
            self.assertIn("avocado", end.result)


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


    async def test_looping_write_todos_stops_at_the_tool_budget(self) -> None:
        """HEADLESS-1: agent-handled tools count against max_tool_calls, and
        the refusal is persisted explicitly rather than as an interrupted
        placeholder."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            async def stream(messages, info):
                # Calls write_todos on every request, forever.
                yield {0: DeltaToolCall(name="write_todos", json_args=arguments,
                                        tool_call_id=f"call-{len(messages)}")}

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go", max_tool_calls=1)]

            self.assertEqual(len([e for e in events if isinstance(e, TodosUpdate)]), 1)
            budget = [e for e in events if isinstance(e, ToolBudgetExhausted)]
            self.assertEqual([b.limit for b in budget], [1])
            last = session.messages()[-1]
            self.assertIn("Not executed", last.parts[0].content)
            self.assertIn("max_tool_calls=1", last.parts[0].content)

    async def test_zero_budget_refuses_every_tool_including_agent_handled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'

            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_todos", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go", max_tool_calls=0)]

            self.assertFalse([e for e in events if isinstance(e, TodosUpdate)])
            self.assertEqual(agent.todos, [])
            self.assertTrue([e for e in events if isinstance(e, ToolBudgetExhausted)])

    async def test_malformed_todos_are_a_tool_error_the_turn_survives(self) -> None:
        """The model writes these arguments, so the wrong shape must reach it as a
        tool error instead of raising out of the agent loop and killing the turn."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_todos", '{"todos": "oops"}')):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go")]

            self.assertFalse([e for e in events if isinstance(e, TodosUpdate)])
            end = next(e for e in events if isinstance(e, ToolEnd))
            self.assertIn("must be an array", end.result)
            self.assertEqual(agent.todos, [], "the bad list is not adopted")
            self.assertTrue([e for e in events if isinstance(e, TurnEnd)],
                            "the model gets the error and finishes the turn")
            # ...and a resume of that session shows the same failed call.
            replayed = replay_events(session.messages())
            self.assertFalse([e for e in replayed if isinstance(e, TodosUpdate)])
            self.assertIn("must be an array",
                          next(e for e in replayed if isinstance(e, ToolEnd)).result)


class SessionLockReleaseTest(unittest.TestCase):
    """Agent.open takes the session lock early; no failure after that may keep it."""

    def test_unparsable_history_releases_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch.object(Session, "messages", side_effect=ValueError("corrupt")):
                with self.assertRaises(ValueError):
                    Agent.open(cwd=cwd, session=session, config=_config())

            self.assertFalse(lockfile.held(session.path))

    def test_prompt_build_failure_releases_a_new_sessions_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", side_effect=OSError("disk full")),
            ):
                with self.assertRaises(OSError):
                    Agent.open(cwd=cwd, config=_config())

            self.assertFalse(lockfile.held(session.path))

    def test_a_missing_snapshot_still_releases_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with self.assertRaises(SessionIncompleteError):
                Agent.open(cwd=cwd, session=session, config=_config())

            self.assertFalse(lockfile.held(session.path))


class AgentCloseTest(unittest.TestCase):
    """An agent holds its session until it is closed, one way or another."""

    def _session(self, cwd: Path) -> Session:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return session

    def test_with_block_releases_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with Agent.open(cwd=cwd, session=session, config=_config()) as agent:
                self.assertIs(agent.session, session)
                self.assertTrue(lockfile.held(session.path))

            self.assertFalse(lockfile.held(session.path))

    def test_closing_twice_leaves_a_later_holder_alone(self) -> None:
        """Claims are refcounted per process: a stale close must not drop one."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)
            first = Agent.open(cwd=cwd, session=session, config=_config())
            first.close()

            resumed = Session(session.path, session.id, cwd)
            second = Agent.open(cwd=cwd, session=resumed, config=_config())
            self.addCleanup(second.close)

            first.close()
            self.assertTrue(lockfile.held(resumed.path), "the second agent still holds it")

    def test_a_dropped_agent_releases_its_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)
            agent = Agent.open(cwd=cwd, session=session, config=_config())
            self.assertTrue(lockfile.held(session.path))

            del agent
            gc.collect()
            self.assertFalse(lockfile.held(session.path))


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


class PendingMessagesTest(unittest.IsolatedAsyncioTestCase):
    """Messages queued mid-turn reach the model at the next request, not the
    next turn."""

    async def test_queued_message_is_injected_before_the_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config(), mode="yolo")

            queue = ["use uv instead"]

            def take() -> list[str]:
                texts, queue[:] = list(queue), []
                return texts

            agent.pending = take
            seen: list[list[object]] = []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                seen.append(list(messages))
                if requests == 1:
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            # Queued before the turn even started, so the first request already
            # carries it, ahead of the tool call it goes on to make.
            self.assertEqual(
                [type(event) for event in events],
                [UserInput, ToolStart, ToolEnd, TextDelta, TurnEnd],
            )
            self.assertEqual(events[0].text, "use uv instead")

            prompts = [part.content for message in seen[0] if isinstance(message, ModelRequest)
                       for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(prompts, ["go", "use uv instead"])

    async def test_a_message_queued_during_a_tool_call_lands_after_its_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config(), mode="yolo")

            queue: list[str] = []
            agent.pending = lambda: [queue.pop(0)] if queue else []
            requests = 0

            async def stream(messages, info):
                nonlocal requests
                requests += 1
                if requests == 1:
                    # Typed while the tool below is running.
                    queue.append("stop, wrong file")
                    yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                            tool_call_id="call-1")}
                else:
                    yield "done"

            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            self.assertEqual(
                [type(event) for event in events],
                [ToolStart, ToolEnd, UserInput, TextDelta, TurnEnd],
            )
            # The injected request follows the tool results rather than
            # replacing them, so no tool_call_id is left unanswered.
            requests_only = [m for m in agent.history if isinstance(m, ModelRequest)]
            self.assertTrue(any(isinstance(part, ToolReturnPart)
                                for part in requests_only[-2].parts))
            self.assertEqual([part.content for part in requests_only[-1].parts],
                             ["stop, wrong file"])

    async def test_without_the_hook_nothing_is_injected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config(), mode="yolo")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("go")
                          if not isinstance(event, RequestStats)]

            self.assertEqual([type(event) for event in events], [TextDelta, TurnEnd])


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

        # SESSION-5: each call is followed by its own result, as live serial
        # execution orders them, not all starts of a batch first.
        self.assertEqual(
            [type(event) for event in events],
            [UserInput, ReasoningDelta, TextDelta, ToolStart, ToolEnd, TodosUpdate, TextDelta],
        )
        tool_end = events[4]
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


class CompactionTokenCountTest(unittest.IsolatedAsyncioTestCase):
    """Counting tokens serializes the whole history, so it stays off the loop."""

    @staticmethod
    def _agent(cwd: Path, **settings) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=Config(**settings))

    async def test_an_unknown_window_counts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # "test:stub" matches no entry in the window table and there is no
            # override, so auto-compaction is off and counting is wasted work.
            agent = self._agent(Path(directory), model="test:stub")
            with patch("paimon.compaction.count_tokens") as count:
                self.assertIsNone(await agent._maybe_compact())
            count.assert_not_called()

    async def test_the_count_runs_in_a_worker_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory), model="test:stub",
                                compaction_context_window=1_000,
                                compaction_reserve_tokens=0)
            threads: list[threading.Thread] = []

            def count(*args, **kwargs) -> int:
                threads.append(threading.current_thread())
                return 1  # well under the window, so nothing is compacted

            with patch("paimon.compaction.count_tokens", side_effect=count):
                self.assertIsNone(await agent._maybe_compact())

            self.assertEqual(len(threads), 1)
            self.assertIsNot(threads[0], threading.main_thread())


class ModelOverrideTest(unittest.TestCase):
    """Agents share one Config, so a per-agent model cannot live in it."""

    def test_the_override_wins_and_the_shared_config_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            config = Config(model="test:stub",
                            providers={"test": {"api_base": "https://example/v1",
                                                "api_key": "k"}})
            plain = Agent(make_session(cwd), "snapshot", config=config)
            overridden = Agent(make_session(cwd), "snapshot", config=config,
                               model_override="test:other")

            self.assertEqual(plain.model_name, "test:stub")
            self.assertEqual(overridden.model_name, "test:other")
            self.assertEqual(config.model, "test:stub")

            with patch("paimon.agent.build_model", side_effect=lambda *key: key) as build:
                self.assertEqual(plain._model(), ("test:stub", "https://example/v1", "k"))
                self.assertEqual(overridden._model(), ("test:other", "https://example/v1", "k"))
            self.assertEqual(build.call_count, 2)

    def test_the_override_picks_its_own_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(make_session(Path(directory)), "snapshot",
                          config=Config(model="test:stub"), model_override="claude-x")
            self.assertEqual(compaction.context_window(agent.model_name), 200_000)


class LiveReplayParityTest(unittest.IsolatedAsyncioTestCase):
    """SESSION-5: resumed history replays with the live event order and the
    live denied styling."""

    @staticmethod
    def _tool_events(events: list) -> list[tuple]:
        return [(type(e).__name__, e.id, getattr(e, "denied", None))
                for e in events if isinstance(e, (ToolStart, ToolEnd))]

    async def test_a_serial_tool_batch_replays_in_live_order(self) -> None:
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            if requests == 1:
                yield {0: DeltaToolCall(name="shell", json_args='{"command": "true"}',
                                        tool_call_id="c1"),
                       1: DeltaToolCall(name="shell", json_args='{"command": "true"}',
                                        tool_call_id="c2")}
            else:
                yield "done"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with patch("paimon.agent.build_model",
                       return_value=FunctionModel(stream_function=stream)):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                live = [event async for event in agent.run("go")]

            live_order = self._tool_events(live)
            self.assertEqual([item[:2] for item in live_order],
                             [("ToolStart", "c1"), ("ToolEnd", "c1"),
                              ("ToolStart", "c2"), ("ToolEnd", "c2")])
            replayed = replay_events(session.messages())
            self.assertEqual(self._tool_events(replayed), live_order)

    async def test_a_denied_tool_still_replays_as_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            arguments = '{"path": "a.txt", "content": "hi"}'
            confirm = AsyncMock(return_value=False)
            with patch("paimon.agent.build_model",
                       return_value=stub_model("write_file", arguments)):
                agent = Agent.open(cwd=cwd, session=session, config=_config(),
                                   confirm=confirm, mode="read")
                live = [event async for event in agent.run("go")]

            live_end = next(e for e in live if isinstance(e, ToolEnd))
            self.assertTrue(live_end.denied)
            replayed_end = next(e for e in replay_events(session.messages())
                                if isinstance(e, ToolEnd))
            self.assertTrue(replayed_end.denied,
                            "the denied state survives persistence and replay")


class TurnOutcomeTest(unittest.IsolatedAsyncioTestCase):
    """SESSION-1: every turn leaves a terminal turn_end record, and failures
    with partial output are persisted without entering the LLM context."""

    def _open(self, cwd: Path, model) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        patcher = patch("paimon.agent.build_model", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Agent.open(cwd=cwd, session=session, config=_config())

    async def test_success_records_a_turn_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), stub_model())
            [event async for event in agent.run("go")]
            last = _records(agent.session)[-1]
            self.assertEqual((last["type"], last["outcome"]), ("turn_end", "success"))

    async def test_zero_output_failure_records_the_error(self) -> None:
        async def explode(messages, info):
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this an async generator

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=explode))
            with self.assertRaises(RuntimeError):
                [event async for event in agent.run("go")]
            last = _records(agent.session)[-1]
            self.assertEqual(last["outcome"], "error")
            self.assertEqual(last["error"], "RuntimeError: boom")
            self.assertNotIn("partial_text", last)
            # No error assistant enters the context: the history ends on the
            # user request, which is still resumable.
            self.assertIsInstance(agent.session.messages()[-1], ModelRequest)

    async def test_failure_after_partial_output_persists_the_partial(self) -> None:
        async def stream(messages, info):
            yield "half an answer"
            raise RuntimeError("dropped")

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=stream))
            with self.assertRaises(RuntimeError):
                [event async for event in agent.run("go")]
            last = _records(agent.session)[-1]
            self.assertEqual(last["outcome"], "error")
            self.assertEqual(last["partial_text"], "half an answer")
            for message in agent.session.messages():
                if isinstance(message, ModelResponse):
                    for part in message.parts:
                        self.assertNotIn("half an answer", getattr(part, "content", ""),
                                         "the partial must not replay as a normal answer")

    async def test_transient_retries_are_recorded(self) -> None:
        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("dropped")
            yield "done"

        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.asyncio.sleep"):
                agent = self._open(Path(directory), FunctionModel(stream_function=stream))
                [event async for event in agent.run("go")]
            records = _records(agent.session)
            retries = [r for r in records if r["type"] == "model_retry"]
            self.assertEqual([r["attempt"] for r in retries], [1])
            self.assertEqual(records[-1]["outcome"], "success")

    async def test_cancellation_records_interrupted(self) -> None:
        async def stream(messages, info):
            yield "partial answer"
            raise asyncio.CancelledError

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=stream))
            with self.assertRaises(asyncio.CancelledError):
                [event async for event in agent.run("go")]
            ends = [r for r in _records(agent.session) if r["type"] == "turn_end"]
            self.assertEqual([r["outcome"] for r in ends], ["interrupted"])
            self.assertEqual(ends[0]["partial_text"], "partial answer")
            # SESSION-2: no truncated answer is persisted as a completed one.
            self.assertIsInstance(agent.session.messages()[-1], ModelRequest)
            self.assertEqual(agent.history, agent.session.messages())

    async def test_budget_stop_records_max_tool_calls(self) -> None:
        arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), stub_model("write_todos", arguments))
            [event async for event in agent.run("go", max_tool_calls=0)]
            self.assertEqual(_records(agent.session)[-1]["outcome"], "max_tool_calls")

    async def test_compaction_failure_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.Agent._maybe_compact",
                       new=AsyncMock(side_effect=[httpx.ConnectError("dropped"), None])):
                arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
                agent = self._open(Path(directory), stub_model("write_todos", arguments))
                [event async for event in agent.run("go")]
            failures = [r for r in _records(agent.session)
                        if r["type"] == "compaction_failed"]
            self.assertEqual(len(failures), 1)
            self.assertIn("dropped", failures[0]["error"])


class UsageAnchorTest(unittest.IsolatedAsyncioTestCase):
    """COMPACT-1: provider-reported usage drives the context count when
    available; the chars/4 heuristic only covers what came after it."""

    def _agent(self, cwd: Path) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=_config())

    async def test_count_prefers_the_anchor_and_estimates_only_the_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent._append_message(ModelRequest(parts=[UserPromptPart(content="x" * 100_000)]))
            agent._usage_anchor = (len(agent.history), 5_000)

            self.assertEqual(await agent.count_context_tokens(), 5_000)

            agent._append_message(ModelRequest(parts=[UserPromptPart(content="y" * 400)]))
            count = await agent.count_context_tokens()
            self.assertGreater(count, 5_000)
            self.assertLess(count, 5_600, "only the appended suffix is estimated")

    async def test_a_completed_request_sets_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = self._agent(Path(directory))
                [event async for event in agent.run("go")]
            self.assertIsNotNone(agent._usage_anchor)
            self.assertEqual(agent._usage_anchor[0], len(agent.history))

    async def test_compaction_drops_the_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent._append_message(ModelRequest(parts=[UserPromptPart(content="old")]))
            recent = ModelResponse(parts=[TextPart(content="recent")])
            agent._append_message(recent)
            agent._usage_anchor = (2, 9_000)
            result = compaction.CompactionResult("checkpoint", [recent], 100, 0)
            with patch("paimon.agent.build_model", return_value=stub_model()), \
                    patch("paimon.agent.compaction.compact", new=AsyncMock(return_value=result)):
                await agent.compact_now()
            self.assertIsNone(agent._usage_anchor)


class OverflowRecoveryTest(unittest.IsolatedAsyncioTestCase):
    """COMPACT-1: a provider context-overflow compacts and retries once."""

    async def test_overflow_compacts_and_retries_once(self) -> None:
        from pydantic_ai.exceptions import ModelHTTPError

        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ModelHTTPError(400, "stub", {"message": "context length exceeded"})
            yield "done"

        result = compaction.CompactionResult("checkpoint", [], 100, 10)

        async def fake_compact(force: bool = False):
            return result if force else None

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with (
                patch("paimon.agent.build_model",
                      return_value=FunctionModel(stream_function=stream)),
                patch("paimon.agent.Agent._maybe_compact", new=AsyncMock(side_effect=fake_compact)),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go")]

            self.assertEqual(attempts, 2, "the request is retried after compaction")
            self.assertEqual("".join(e.text for e in events if isinstance(e, TextDelta)), "done")
            records = _records(session)
            self.assertTrue([r for r in records if r["type"] == "context_overflow"])
            self.assertEqual(records[-1]["outcome"], "success")

    async def test_a_second_overflow_is_raised_not_looped(self) -> None:
        from pydantic_ai.exceptions import ModelHTTPError

        attempts = 0

        async def stream(messages, info):
            nonlocal attempts
            attempts += 1
            raise ModelHTTPError(400, "stub", {"message": "context length exceeded"})
            yield  # pragma: no cover - makes this an async generator

        result = compaction.CompactionResult("checkpoint", [], 100, 10)

        async def fake_compact(force: bool = False):
            return result if force else None

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            with (
                patch("paimon.agent.build_model",
                      return_value=FunctionModel(stream_function=stream)),
                patch("paimon.agent.Agent._maybe_compact", new=AsyncMock(side_effect=fake_compact)),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                with self.assertRaises(ModelHTTPError):
                    [event async for event in agent.run("go")]

            self.assertEqual(attempts, 2, "exactly one retry, then the error surfaces")


class CompactionFailureTest(unittest.IsolatedAsyncioTestCase):
    """A failed compaction must not silently disable the safety net for the turn."""

    async def _run_with(self, failure: Exception) -> tuple[list, AsyncMock]:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            # write_todos needs no confirmation, so the turn takes two model
            # requests and therefore two compaction checks.
            arguments = '{"todos": [{"content": "x", "status": "pending"}]}'
            compact = AsyncMock(side_effect=[failure, None])
            with (
                patch("paimon.agent.build_model",
                      return_value=stub_model("write_todos", arguments)),
                patch("paimon.agent.Agent._maybe_compact", new=compact),
            ):
                agent = Agent.open(cwd=cwd, session=session, config=_config())
                events = [event async for event in agent.run("go")]
            return events, compact

    async def test_transient_failure_is_retried_on_the_next_step(self) -> None:
        events, compact = await self._run_with(httpx.ConnectError("dropped"))

        self.assertEqual(compact.await_count, 2)
        self.assertEqual(len([e for e in events if isinstance(e, ContextCompactionFailed)]), 1)

    async def test_a_failure_that_will_not_fix_itself_stops_for_the_turn(self) -> None:
        events, compact = await self._run_with(ValueError("no context window"))

        self.assertEqual(compact.await_count, 1)
        self.assertEqual(len([e for e in events if isinstance(e, ContextCompactionFailed)]), 1)


if __name__ == "__main__":
    unittest.main()


class AgentToolsTest(unittest.IsolatedAsyncioTestCase):
    """The four supervised tools as the agent loop sees them."""

    @staticmethod
    def _agent(cwd: Path, **kwargs) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return Agent.open(cwd=cwd, session=session, config=_config(), **kwargs)

    async def test_without_a_supervisor_they_refuse_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            with patch("paimon.agent.build_model",
                       return_value=stub_model("spawn_agent", '{"prompt": "go"}')):
                events = [event async for event in agent.run("do it")]

            end = next(event for event in events if isinstance(event, ToolEnd))
            self.assertIn("only works in the interactive UI", end.result)

    async def test_a_narrowed_toolset_disables_them_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            toolset = tools.without(tools.REGISTRY, tools.SUBAGENT_DENIED)
            agent = self._agent(Path(directory), toolset=toolset)
            agent.supervisor = _FakeSupervisor()
            with patch("paimon.agent.build_model",
                       return_value=stub_model("spawn_agent", '{"prompt": "go"}')):
                events = [event async for event in agent.run("do it")]

            end = next(event for event in events if isinstance(event, ToolEnd))
            self.assertIn("unknown tool", end.result)
            names = [schema["function"]["name"] for schema in agent.tool_schemas]
            self.assertNotIn("spawn_agent", names, "and the model is never offered it")

    async def test_a_call_is_handed_to_the_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.supervisor = supervisor = _FakeSupervisor()
            with patch("paimon.agent.build_model",
                       return_value=stub_model("read_job", '{"job_id": "a1f2"}')):
                events = [event async for event in agent.run("do it")]

            self.assertEqual(supervisor.calls, [("read_job", {"job_id": "a1f2"})])
            end = next(event for event in events if isinstance(event, ToolEnd))
            self.assertEqual(end.result, "handled")


class AgentStatusInjectionTest(unittest.IsolatedAsyncioTestCase):
    """What the agents this session started did reaches the model as history."""

    async def test_a_status_line_opens_the_turn_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config())
            agent.supervisor = _FakeSupervisor(summary="a1f2 finished")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            notices = [event for event in events if isinstance(event, AgentsNotice)]
            self.assertEqual([notice.text for notice in notices], ["a1f2 finished"])
            self.assertTrue(is_agents_message(agent.history[0]),
                            "it goes in ahead of the user's own message")
            self.assertFalse(is_agents_message(agent.history[1]))

            # It survives a reload, and replays as a notice rather than as
            # something the user typed.
            replayed = replay_events(session.messages())
            self.assertEqual([type(event) for event in replayed][:2], [AgentsNotice, UserInput])

    async def test_nothing_is_injected_without_news(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")
            agent = Agent.open(cwd=cwd, session=session, config=_config())
            agent.supervisor = _FakeSupervisor(summary=None)

            with patch("paimon.agent.build_model", return_value=stub_model()):
                events = [event async for event in agent.run("what now")]

            self.assertFalse([event for event in events if isinstance(event, AgentsNotice)])
            self.assertFalse(any(is_agents_message(message) for message in agent.history))


class _FakeSupervisor:
    def __init__(self, summary=None) -> None:
        self.summary = summary
        self.calls: list = []

    def status_summary(self, caller) -> object:
        return self.summary

    async def handle(self, name: str, args: dict, *, caller) -> str:
        self.calls.append((name, args))
        return "handled"
