import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static
from textual.worker import WorkerState

from helpers import SILENT_EVENTS, agent_events, stub_model
from paimon import lockfile
from paimon.agent import Agent, ReasoningDelta
from paimon.app import PaimonApp, _EventRenderer, _session_label
from paimon.config import Config
from paimon.login import PickerScreen
from paimon.session import Session
from paimon.ui import AssistantMessage, ConfirmPanel, PromptInput, ToolResult, UserMessage


class AppTestCase(unittest.IsolatedAsyncioTestCase):
    """Pilot-driven TUI tests against an isolated data dir and a stub model."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_DATA_HOME": tmp.name,
                                        "PAIMON_CONFIG_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def make_app(self, *, session: Session | None = None, mode: str = "read",
                 config: Config | None = None, pick_session: bool = False) -> PaimonApp:
        agent = Agent.open(session=session, mode=mode, config=config or Config(model="test-model"))
        return PaimonApp(agent, resumed=session is not None, pick_session=pick_session)


class ConfirmPanelTest(AppTestCase):
    @staticmethod
    async def _open(app: PaimonApp, pilot, tool: str = "shell", args: dict | None = None) -> asyncio.Future:
        task = asyncio.ensure_future(app._confirm(tool, args or {"command": "echo hi"}))
        await pilot.pause()
        return task

    async def test_enter_allows_and_restores_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            task = await self._open(app, pilot)
            panel = app.query_one("#confirm-panel", ConfirmPanel)
            self.assertFalse(prompt.display, "prompt hidden while confirming")
            self.assertIs(app.focused, panel)
            await pilot.press("enter")
            self.assertTrue(await task)
            await pilot.pause()
            self.assertFalse(app.query("#confirm-panel"))
            self.assertTrue(prompt.display)

    async def test_escape_denies(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, args={"command": "rm -rf x"})
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_number_shortcut_denies(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, "write_file", {"path": "c.py", "content": "x"})
            await pilot.press("2")
            self.assertFalse(await task)

    async def test_start_new_session_detail_shows_the_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, "start_new_session", {"prompt": "carry on"})
            detail = str(app.query_one("#confirm-detail Static", Static).render())
            self.assertIn("carry on", detail)
            self.assertIn("fresh one", detail)
            await pilot.press("escape")
            self.assertFalse(await task)


class ModeCycleTest(AppTestCase):
    async def test_shift_tab_cycles_mode_and_updates_indicators(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            self.assertEqual(app.mode, "read")
            prompt = app.query_one(PromptInput)
            self.assertEqual(prompt.border_title, " read ")

            await pilot.press("shift+tab")
            self.assertEqual(app.mode, "edit")
            self.assertEqual(app.agent.mode, "edit")
            self.assertEqual(prompt.border_title, " edit ")
            self.assertIn("edit mode", str(app.query_one("#statusbar", Static).render()))

            await pilot.press("shift+tab", "shift+tab")
            self.assertEqual(app.mode, "read")

    async def test_new_session_keeps_current_mode(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("shift+tab")
            app.action_new_session()
            self.assertEqual(app.agent.mode, "edit")

    async def test_shift_tab_while_confirm_panel_open_keeps_pending_future(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = asyncio.ensure_future(app._confirm("shell", {"command": "echo hi"}))
            await pilot.pause()
            await pilot.press("shift+tab")
            self.assertEqual(app.mode, "edit")
            self.assertTrue(app.query("#confirm-panel"), "panel survives a mode switch")
            await pilot.press("enter")
            self.assertTrue(await task)


class ResumeSessionTest(AppTestCase):
    @staticmethod
    def _old_session(content: str = "hello there") -> Session:
        session = Session.create(Path.cwd())
        session.append_system_prompt("sys")
        session.append_message(ModelRequest(parts=[UserPromptPart(content=content)]))
        return session

    def _log_text(self, app: PaimonApp) -> str:
        return " ".join(str(widget.render()) for widget in app.query_one("#log").children)

    async def test_palette_resume_swaps_agent_and_renders_history(self) -> None:
        old = self._old_session()
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_cycle_mode()  # read -> edit, must survive the resume
            app.action_resume_session()
            await pilot.pause()
            self.assertIsInstance(app.screen, PickerScreen)
            app.screen.dismiss(_session_label(old))
            await pilot.pause()
            self.assertEqual(app.agent.session.id, old.id)
            self.assertEqual(app.agent.mode, "edit")
            self.assertTrue(app.query(UserMessage), "history re-rendered")
            self.assertIn("Resumed session", self._log_text(app))

    async def test_noop_while_turn_is_running(self) -> None:
        self._old_session()
        app = self.make_app()
        async with app.run_test() as pilot:
            before = app.agent
            app._turn = SimpleNamespace(is_running=True)
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertIs(app.agent, before)

    async def test_no_sessions_shows_notice(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertIn("No sessions to resume", self._log_text(app))

    async def test_constructor_session_param_resumes_on_mount(self) -> None:
        old = self._old_session()
        app = self.make_app(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(app.agent.session.id, old.id)
            self.assertTrue(app.query(UserMessage))
            self.assertIn("Resumed session", self._log_text(app))

    async def test_history_replays_assistant_and_tool_widgets(self) -> None:
        session = self._old_session("do it")
        session.append_message(ModelResponse(parts=[
            TextPart(content="working"),
            ToolCallPart(tool_name="shell", args='{"command": "ls"}', tool_call_id="c1"),
        ]))
        session.append_message(ModelRequest(parts=[
            ToolReturnPart(tool_name="shell", content="a.py", tool_call_id="c1"),
        ]))
        session.append_message(ModelResponse(parts=[TextPart(content="done")]))
        app = self.make_app(session=session)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(len(app.query(UserMessage)), 1)
            self.assertEqual(len(app.query(AssistantMessage)), 2)
            results = app.query(ToolResult)
            self.assertEqual(len(results), 1)
            self.assertIn("a.py", results.first()._full)


class ForkSessionTest(AppTestCase):
    async def test_fork_swaps_agent_and_keeps_the_log(self) -> None:
        old = ResumeSessionTest._old_session("keep this")
        app = self.make_app(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_fork_session()
            await pilot.pause()
            self.assertNotEqual(app.agent.session.id, old.id)
            self.assertEqual(app.agent.history, old.messages())
            self.assertTrue(app.query(UserMessage), "log survives the fork")
            log_text = " ".join(str(w.render()) for w in app.query_one("#log").children)
            self.assertIn("Forked session", log_text)
            self.assertNotIn(str(old.path.resolve()), lockfile._held)
            self.assertIn(str(app.agent.session.path.resolve()), lockfile._held)

    async def test_source_session_stays_resumable(self) -> None:
        old = ResumeSessionTest._old_session()
        app = self.make_app(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_fork_session()
            await pilot.pause()
            resumed = Agent.open(session=old)
            self.assertEqual(resumed.history, app.agent.history)
            resumed.session.unlock()

    async def test_noop_while_turn_is_running(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            before = app.agent
            app._turn = SimpleNamespace(is_running=True)
            app.action_fork_session()
            await pilot.pause()
            self.assertIs(app.agent, before)


class SessionLabelTest(AppTestCase):
    def test_label_has_local_time_short_id_and_flattened_preview(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message(ModelRequest(parts=[UserPromptPart(content="fix the\nbug " + "x" * 50)]))

        label = _session_label(session)

        when = datetime.fromisoformat(session.created_at()).astimezone().strftime("%m-%d %H:%M")
        preview = " ".join(("fix the\nbug " + "x" * 50).split())
        self.assertEqual(label, f"{when} · {session.id[:8]} · {preview[:40]}…")


class StatusLineTest(AppTestCase):
    async def test_pinned_status_layout_and_toggle(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            status = app.query_one("#response-status", Horizontal)
            self.assertFalse(status.display, "status hidden when idle")
            ids = [child.id for child in app.query_one("#workspace").children]
            self.assertEqual(ids, ["log", "response-status", "queued", "prompt", "statusbar"])

            app._set_status(True, " Counting mora… 3s")
            await pilot.pause()
            self.assertTrue(status.display)
            self.assertIn("3s", str(status.query_one(".status-label", Static).render()))

            app._set_status(False)
            await pilot.pause()
            self.assertFalse(status.display)


class TodoPanelTest(AppTestCase):
    @staticmethod
    def _plan(*statuses: str) -> list[dict]:
        return [{"content": f"step {i}", "status": s} for i, s in enumerate(statuses)]

    async def test_burst_collapses_but_a_panel_with_output_under_it_stays(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            log = app.query_one("#log", VerticalScroll)
            app._show_todos(self._plan("in_progress", "pending"))
            app._show_todos(self._plan("completed", "in_progress"))
            await pilot.pause()
            panels = app.query(".todos")
            self.assertEqual(len(panels), 1, "consecutive revisions share one panel")
            self.assertIn("1/2", str(panels.first().render()))

            app._add_tool_result("output")
            app._show_todos(self._plan("completed", "completed"))
            await pilot.pause()
            panels = app.query(".todos")
            self.assertEqual(len(panels), 2, "the earlier plan is left as a snapshot")
            self.assertIn("1/2", str(panels.first().render()))
            self.assertIn("2/2", str(panels.last().render()))
            self.assertIs(log.children[-1], panels.last())

    async def test_clearing_removes_the_panel(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app._show_todos(self._plan("pending"))
            app._show_todos([])
            await pilot.pause()
            self.assertEqual(len(app.query(".todos")), 0)
            self.assertIsNone(app._todo_panel)


class EventCoverageTest(AppTestCase):
    """The TUI renderer is held to the same event list as the headless ones.

    An event the renderer has no branch for would silently vanish from the
    conversation log, so each one has to put something in it.
    """

    async def test_every_event_puts_something_in_the_log(self) -> None:
        app = self.make_app(config=Config(model="test-model", show_reasoning=True))
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app)
            log = app.query_one("#log", VerticalScroll)
            for event in agent_events():
                name = type(event).__name__
                before = len(log.children)
                await renderer.handle(event)
                await pilot.pause()
                if name in SILENT_EVENTS:
                    continue
                self.assertGreater(len(log.children), before, f"{name} rendered nothing")
            await renderer.close()


class ReasoningDisplayTest(AppTestCase):
    async def test_reasoning_rendered_when_enabled(self) -> None:
        app = self.make_app(config=Config(model="test-model", show_reasoning=True))
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app)
            await renderer.handle(ReasoningDelta("thinking hard"))
            await pilot.pause()
            widgets = app.query(".reasoning")
            self.assertEqual(len(widgets), 1)
            self.assertIn("thinking hard", str(widgets.first().render()))

    async def test_reasoning_hidden_by_default(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app)
            await renderer.handle(ReasoningDelta("thinking hard"))
            await pilot.pause()
            self.assertFalse(app.query(".reasoning"))

    async def test_toggle_flips_and_persists(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_toggle_reasoning()
            await pilot.pause()
            self.assertTrue(Config.load().show_reasoning)
            self.assertTrue(app.config.show_reasoning)
            app.action_toggle_reasoning()
            self.assertFalse(Config.load().show_reasoning, "persisted to config.json")


class QueueTest(AppTestCase):
    async def test_queue_flush_and_cancel(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            queued = app.query_one("#queued", Static)
            self.assertFalse(prompt.disabled, "prompt stays enabled during turns")

            # prompts submitted while a (fake) turn runs are queued and shown
            app._turn = SimpleNamespace(is_running=True)
            app.handle_submit(PromptInput.Submitted("first message"))
            app.handle_submit(PromptInput.Submitted("second message"))
            await pilot.pause()
            self.assertEqual(app._queue, ["first message", "second message"])
            self.assertTrue(queued.display)

            # a finished turn flushes the queue into the next turn
            started: list[str] = []
            app._start_turn = started.append
            app.on_worker_state_changed(SimpleNamespace(worker=app._turn, state=WorkerState.SUCCESS))
            await pilot.pause()
            self.assertEqual(started, ["first message\n\nsecond message"])
            self.assertFalse(app._queue)
            self.assertFalse(queued.display)

            # an interrupted turn hands the queue back to the input instead
            app.handle_submit(PromptInput.Submitted("queued later"))
            prompt.load_text("half-typed draft")
            app.on_worker_state_changed(SimpleNamespace(worker=app._turn, state=WorkerState.CANCELLED))
            await pilot.pause()
            self.assertEqual(prompt.text, "queued later\nhalf-typed draft")
            self.assertFalse(app._queue)
            self.assertEqual(started, ["first message\n\nsecond message"], "cancel must not auto-submit")


class HandoffTest(AppTestCase):
    """start_new_session in the TUI: confirm (even in yolo), switch, resume hint."""

    @staticmethod
    def _log_text(app: PaimonApp) -> str:
        return " ".join(str(widget.render()) for widget in app.query_one("#log").children)

    @staticmethod
    async def _wait_for(pilot, condition) -> None:
        for _ in range(200):
            await pilot.pause()
            if condition():
                return
        raise AssertionError("condition not reached")

    async def test_approved_handoff_switches_to_a_new_session(self) -> None:
        app = self.make_app(mode="yolo")
        old = app.agent.session
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", '{"prompt": "carry on"}')):
            async with app.run_test() as pilot:
                app.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.query("#confirm-panel"))
                await pilot.press("enter")
                await self._wait_for(pilot, lambda: app.agent.session.id != old.id
                                     and app._turn is not None and not app._turn.is_running)

                log = self._log_text(app)
                self.assertIn("Started new session", log)
                self.assertIn("Handed off", log)
                self.assertIn(old.id[:8], log, "resume hint names the old session")
                self.assertIn("carry on", log, "handoff prompt submitted as the first message")
                self.assertTrue(old.path.exists())
                self.assertIsNone(app._pending_handoff)

    async def test_denied_handoff_keeps_the_session(self) -> None:
        app = self.make_app(mode="yolo")
        old_id = app.agent.session.id
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", '{"prompt": "carry on"}')):
            async with app.run_test() as pilot:
                app.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.query("#confirm-panel"))
                await pilot.press("escape")
                await self._wait_for(pilot, lambda: app._turn is not None
                                     and not app._turn.is_running)

                self.assertEqual(app.agent.session.id, old_id)
                self.assertIsNone(app._pending_handoff)
                self.assertTrue(app.query(AssistantMessage), "turn continued after the denial")

    async def test_queued_messages_return_to_input_on_handoff(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            worker = SimpleNamespace(is_running=True)
            app._turn = worker
            app.handle_submit(PromptInput.Submitted("for the old context"))
            worker.is_running = False
            app._pending_handoff = "next phase"
            started: list[str] = []
            app._start_turn = started.append
            old_id = app.agent.session.id

            app.on_worker_state_changed(SimpleNamespace(worker=worker, state=WorkerState.SUCCESS))
            await pilot.pause()

            self.assertEqual(started, ["next phase"])
            self.assertNotEqual(app.agent.session.id, old_id)
            self.assertEqual(app.query_one(PromptInput).text, "for the old context")
            self.assertFalse(app._queue)
            self.assertIsNone(app._pending_handoff)

    async def test_failed_turn_clears_pending_handoff_without_switching(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            worker = SimpleNamespace(is_running=False)
            app._turn = worker
            app._pending_handoff = "next phase"
            started: list[str] = []
            app._start_turn = started.append
            old_id = app.agent.session.id

            app.on_worker_state_changed(SimpleNamespace(worker=worker, state=WorkerState.ERROR))
            await pilot.pause()

            self.assertIsNone(app._pending_handoff)
            self.assertFalse(started)
            self.assertEqual(app.agent.session.id, old_id)


class ProfileSwitchTest(AppTestCase):
    @staticmethod
    def _write_profile(name: str, **data) -> None:
        directory = Path(os.environ["PAIMON_CONFIG_HOME"]) / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps(data), encoding="utf-8")

    async def test_switch_reloads_config_and_statusbar(self) -> None:
        self._write_profile("work", model="test:work")
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_switch_profile()
            await pilot.pause()
            self.assertIsInstance(app.screen, PickerScreen)
            app.screen.dismiss("work")
            await pilot.pause()
            self.assertEqual(app.config.profile, "work")
            self.assertEqual(app.config.model, "test:work")
            self.assertIs(app.agent.config, app.config)
            self.assertIn("profile work", str(app.query_one("#statusbar", Static).render()))

    async def test_unconfigured_profile_opens_login_and_cancel_reverts(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_switch_profile()
            await pilot.pause()
            # An unlisted typed name switches to a not-yet-existing profile,
            # which has no model, so the login flow opens. Cancel it.
            app.screen.dismiss("fresh")
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            self.assertEqual(app.config.profile, "default")
            self.assertEqual(app.config.model, "test-model")
            self.assertIs(app.agent.config, app.config)

    async def test_noop_while_turn_is_running(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app._turn = SimpleNamespace(is_running=True)
            app.action_switch_profile()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertEqual(app.config.profile, "default")


if __name__ == "__main__":
    unittest.main()
