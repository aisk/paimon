import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from textual.widgets import Static

from paimon import lockfile
from paimon.agent import Agent
from paimon.app import PaimonApp
from paimon.jobs import AgentJob, Outcome
from paimon.login import LoginScreen, PickerScreen
from paimon.pane import _session_label
from paimon.ui import (
    AssistantMessage,
    ConfirmPanel,
    PromptInput,
    ToolResult,
    UserMessage,
)
from tests.support.agent import stub_model
from tests.support.app import AppTestCase, end_turn, hold_turn


class ResumeSessionTest(AppTestCase):

    def _log_text(self, app: PaimonApp) -> str:
        return " ".join(str(widget.render())
                         for widget in app.query_one("#log").walk_children())

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
            self.assertEqual(app.pane.agent.session.id, old.id)
            self.assertEqual(app.pane.agent.mode, "edit")
            self.assertTrue(app.query(UserMessage), "history re-rendered")
            self.assertIn("Resumed session", self._log_text(app))

    async def test_noop_while_turn_is_running(self) -> None:
        self._old_session()
        app = self.make_app()
        async with app.run_test() as pilot:
            before = app.pane.agent
            hold_turn(app.pane)
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertIs(app.pane.agent, before)

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
            self.assertEqual(app.pane.agent.session.id, old.id)
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
        old = self._old_session("keep this")
        app = self.make_app(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_fork_session()
            await pilot.pause()
            self.assertNotEqual(app.pane.agent.session.id, old.id)
            self.assertEqual(app.pane.agent.history, old.messages())
            self.assertTrue(app.query(UserMessage), "log survives the fork")
            log_text = " ".join(str(w.render()) for w in app.query_one("#log").children)
            self.assertIn("Forked session", log_text)
            self.assertNotIn(str(old.path.resolve()), lockfile._held)
            self.assertIn(str(app.pane.agent.session.path.resolve()), lockfile._held)

    async def test_source_session_stays_resumable(self) -> None:
        old = self._old_session()
        app = self.make_app(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_fork_session()
            await pilot.pause()
            resumed = Agent.open(session=old)
            self.assertEqual(resumed.history, app.pane.agent.history)
            resumed.session.unlock()

    async def test_noop_while_turn_is_running(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            before = app.pane.agent
            hold_turn(app.pane)
            app.action_fork_session()
            await pilot.pause()
            self.assertIs(app.pane.agent, before)


class HandoffTest(AppTestCase):
    """start_new_session in the TUI: confirm (even in yolo), switch, resume hint."""

    @staticmethod
    def _log_text(app: PaimonApp) -> str:
        return " ".join(str(widget.render())
                         for widget in app.query_one("#log").walk_children())

    async def test_approved_handoff_switches_to_a_new_session(self) -> None:
        app = self.make_app(mode="yolo")
        old = app.pane.agent.session
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", '{"prompt": "carry on"}')):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.query(ConfirmPanel))
                await pilot.press("enter")
                await self._wait_for(pilot, lambda: app.pane.agent.session.id != old.id
                                     and not app.pane.is_busy)

                log = self._log_text(app)
                self.assertIn("Started new session", log)
                self.assertIn("Handed off", log)
                self.assertIn(old.id[:8], log, "resume hint names the old session")
                self.assertIn("carry on", log, "handoff prompt submitted as the first message")
                self.assertTrue(old.path.exists())
                self.assertIsNone(app.pane._pending_handoff)

    async def test_denied_handoff_keeps_the_session(self) -> None:
        app = self.make_app(mode="yolo")
        old_id = app.pane.agent.session.id
        with patch("paimon.agent.build_model",
                   return_value=stub_model("start_new_session", '{"prompt": "carry on"}')):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.query(ConfirmPanel))
                await pilot.press("escape")
                await self._wait_for(pilot, lambda: not app.pane.is_busy)

                self.assertEqual(app.pane.agent.session.id, old_id)
                self.assertIsNone(app.pane._pending_handoff)
                self.assertTrue(app.query(AssistantMessage), "turn continued after the denial")

    async def test_queued_messages_return_to_input_on_handoff(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            app.pane.handle_submit(PromptInput.Submitted("for the old context"))
            app.pane._pending_handoff = "next phase"
            old_id = app.pane.agent.session.id

            # Patched on the class, not the instance: the handoff opens a new
            # session, and with it the new job the prompt actually lands in.
            started: list[str] = []
            with patch.object(AgentJob, "submit",
                              lambda self, text: started.append(text) or True):
                await end_turn(app.pane)
                await pilot.pause()

            self.assertEqual(started, ["next phase"])
            self.assertNotEqual(app.pane.agent.session.id, old_id)
            self.assertEqual(app.query_one(PromptInput).text, "for the old context")
            self.assertFalse(app.pane._queue)
            self.assertIsNone(app.pane._pending_handoff)

    async def test_failed_turn_clears_pending_handoff_without_switching(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.pane._pending_handoff = "next phase"
            started: list[str] = []
            app.pane.job.submit = started.append
            old_id = app.pane.agent.session.id

            await end_turn(app.pane, Outcome.FAILED, error="provider failed")
            await pilot.pause()

            self.assertIsNone(app.pane._pending_handoff)
            self.assertFalse(started)
            self.assertEqual(app.pane.agent.session.id, old_id)


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
            self.assertIs(app.pane.agent.config, app.config)
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
            self.assertIs(app.pane.agent.config, app.config)

    async def test_noop_while_turn_is_running(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            app.action_switch_profile()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertEqual(app.config.profile, "default")

    async def test_login_is_refused_while_a_turn_is_running(self) -> None:
        """Login rewrites the model every running turn re-reads at each step."""
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            app.action_login()
            await pilot.pause()
            self.assertEqual(self._login_screens(app), [])

    async def test_login_opens_when_no_turn_is_running(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_login()
            await pilot.pause()
            # LoginScreen immediately pushes its provider picker on top, so
            # look down the stack rather than at the active screen.
            self.assertEqual(len(self._login_screens(app)), 1)

    @staticmethod
    def _login_screens(app: PaimonApp) -> list[LoginScreen]:
        return [screen for screen in app.screen_stack if isinstance(screen, LoginScreen)]


class PaneSessionLockTest(AppTestCase):
    async def test_resume_hides_a_session_open_in_another_pane(self) -> None:
        old = self._old_session()
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            app.action_resume_session()
            await pilot.pause()
            app.screen.dismiss(_session_label(old))
            await pilot.pause()
            self.assertEqual(app.pane.agent.session.id, old.id)

            await pilot.press("ctrl+pageup")
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen,
                                     "the session is already open in the other pane")
            self.assertIn("No sessions to resume", self._log_text(app.pane))


class AgentCwdTest(AppTestCase):
    """Switching sessions keeps the agent's cwd, rather than falling back to
    the process cwd, so the permission boundary cannot drift."""

    async def test_new_and_forked_sessions_inherit_the_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            elsewhere = Path(directory).resolve()
            app = self.make_app()
            async with app.run_test():
                app.pane.agent.cwd = elsewhere

                app.action_new_session()
                self.assertEqual(app.pane.agent.cwd, elsewhere)

                app.action_fork_session()
                self.assertEqual(app.pane.agent.cwd, elsewhere)
