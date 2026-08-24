import asyncio
import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import RichLog, Static
from textual.widgets.markdown import MarkdownBlock
from helpers import SILENT_EVENTS, agent_events, stub_model
from paimon import agents, aside, lockfile, tools
from paimon.agent import Agent, ReasoningDelta, RequestStats, ToolEnd, ToolStart
from paimon.app import MAX_PANES, PaimonApp
from paimon.jobs import AgentJob, Outcome, Result
from paimon.pane import SessionPane, _EventRenderer, _session_label
from paimon import skills
from paimon.config import Config
from paimon.login import LoginScreen, PickerScreen
from paimon.session import Session, is_agents_message
from paimon.tabs import PaneTab
from paimon.commandpane import CommandPane
from paimon.ui import (
    AssistantMessage,
    ConfirmPanel,
    EditCall,
    PromptInput,
    RecapMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


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
        task = asyncio.ensure_future(app.pane._confirm(tool, args or {"command": "echo hi"}))
        await pilot.pause()
        return task

    async def test_enter_allows_and_restores_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            task = await self._open(app, pilot)
            panel = app.query_one(ConfirmPanel)
            self.assertFalse(prompt.display, "prompt hidden while confirming")
            self.assertIs(app.focused, panel)
            await pilot.press("enter")
            self.assertTrue(await task)
            await pilot.pause()
            self.assertFalse(app.query(ConfirmPanel))
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

    async def test_stale_panel_is_swept_before_mounting(self) -> None:
        # remove() is asynchronous, so a leftover panel from an interrupted
        # confirm can still be mounted when the next one arrives; it must not
        # collide on the fixed widget ID.
        app = self.make_app()
        async with app.run_test() as pilot:
            leftover = ConfirmPanel("shell", {"command": "old"},
                                    asyncio.get_running_loop().create_future())
            await app.pane.mount(leftover)
            task = await self._open(app, pilot, args={"command": "new"})
            self.assertEqual(len(app.query(ConfirmPanel)), 1)
            await pilot.press("enter")
            self.assertTrue(await task)

    async def test_quit_while_blocked_on_confirm_does_not_crash(self) -> None:
        # Quitting cancels the turn worker only after the DOM is torn down,
        # so the cancel handler must not touch widgets anymore.
        app = self.make_app()
        with patch("paimon.agent.build_model",
                   return_value=stub_model("shell", '{"command": "rm x"}')):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                for _ in range(200):
                    await pilot.pause()
                    if app.query(ConfirmPanel):
                        break
                else:
                    raise AssertionError("confirm panel never appeared")
                await pilot.press("ctrl+c")

    async def test_long_command_shows_head_and_tail(self) -> None:
        """UI-2: what gets approved is the whole operation — a dangerous
        suffix must never be hidden behind a prefix clip."""
        command = "echo start " + "x" * 60_000 + " && rm -rf tail-danger"
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, args={"command": command})
            detail = str(app.query_one("#confirm-detail Static", Static).render())
            self.assertIn("echo start", detail)
            self.assertIn("rm -rf tail-danger", detail)
            self.assertIn("not shown", detail, "the elision is named, not silent")
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_write_preview_resolves_against_the_agent_cwd(self) -> None:
        from rich.console import Group
        app = self.make_app()
        async with app.run_test() as pilot:
            with tempfile.TemporaryDirectory() as directory:
                other = Path(directory).resolve()
                (other / "w.txt").write_text("old body\n")
                app.pane.agent.cwd = other
                task = await self._open(app, pilot, "write_file",
                                        {"path": "w.txt", "content": "new body\n"})
                panel = app.query_one(ConfirmPanel)
                self.assertIsInstance(
                    panel._detail(), Group,
                    "the existing file is found via the agent's cwd, so the "
                    "preview is a diff against it")
                await pilot.press("escape")
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


class _HeldTurn:
    """A stand-in for the task a real turn runs in."""

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        pass


def hold_turn(pane) -> None:
    """Make a pane look busy without a model behind it.

    The driver is parked on its inbox, so a stand-in for the turn task is all
    is_busy needs; submitting a prompt would really run one.
    """
    pane.job._turn = _HeldTurn()


async def end_turn(pane, outcome: Outcome = Outcome.SUCCESS, error: str = "") -> None:
    """Finish the held turn the way the driver would."""
    pane.job._turn = None
    await pane._end_turn(Result(outcome, error=error))


class BackgroundPaneTest(AppTestCase):
    """Guards for panes that are mounted but not on screen.

    Widget.focusable only looks at visibility, which is unrelated to display,
    so a hidden pane really can take the keyboard away from the visible one —
    and answer, with the user's next keystroke, a confirmation they never saw.
    """

    async def _background_pane(self, app: PaimonApp) -> SessionPane:
        pane = SessionPane(Agent.open(config=app.config), job_id="bg01", id="pane-2")
        self.addCleanup(pane.agent.session.unlock)
        await app.mount(pane)
        pane.display = False
        return pane

    async def test_confirming_in_a_hidden_pane_leaves_focus_alone(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            other = await self._background_pane(app)
            prompt = app.pane.query_one(PromptInput)
            task = asyncio.ensure_future(other._confirm("shell", {"command": "rm x"}))
            await pilot.pause()

            self.assertEqual(len(other.query(ConfirmPanel)), 1, "the panel is up in its own pane")
            self.assertIs(app.focused, prompt, "the visible pane keeps the keyboard")
            await pilot.press("y")
            self.assertFalse(task.done(), "keystrokes must not answer an unseen confirmation")
            self.assertEqual(prompt.text, "y")

            other.query_one(ConfirmPanel)._resolve("deny")
            self.assertFalse(await task)

    async def test_a_confirmation_elsewhere_survives_a_new_one(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            other = await self._background_pane(app)
            waiting = asyncio.ensure_future(other._confirm("shell", {"command": "rm x"}))
            await pilot.pause()
            mine = await ConfirmPanelTest._open(app, pilot)

            self.assertEqual(len(other.query(ConfirmPanel)), 1, "the sweep is pane-scoped")
            await pilot.press("enter")
            self.assertTrue(await mine)
            self.assertFalse(waiting.done())

            other.query_one(ConfirmPanel)._resolve("deny")
            self.assertFalse(await waiting)

    async def test_stray_typing_stays_in_the_visible_pane(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            other = await self._background_pane(app)
            app.pane.query_one("#log", VerticalScroll).focus()
            await pilot.press("h", "i")
            self.assertEqual(app.pane.query_one(PromptInput).text, "hi")
            self.assertEqual(other.query_one(PromptInput).text, "")


class StrayTypingTest(AppTestCase):
    async def test_typing_with_log_focused_lands_in_the_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.query_one("#log", VerticalScroll).focus()
            await pilot.press("h", "i")
            prompt = app.query_one(PromptInput)
            self.assertIs(app.focused, prompt)
            self.assertEqual(prompt.text, "hi")

    async def test_confirm_panel_keeps_the_keyboard(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await ConfirmPanelTest._open(app, pilot, args={"command": "rm x"})
            await pilot.press("x")  # a key the panel does not handle
            self.assertIsInstance(app.focused, ConfirmPanel)
            self.assertEqual(app.query_one(PromptInput).text, "")
            await pilot.press("escape")
            self.assertFalse(await task)


class ModeCycleTest(AppTestCase):
    async def test_shift_tab_cycles_mode_and_updates_indicators(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            self.assertEqual(app.pane.mode, "read")
            prompt = app.query_one(PromptInput)
            self.assertEqual(prompt.border_title, " read ")

            await pilot.press("shift+tab")
            self.assertEqual(app.pane.mode, "edit")
            self.assertEqual(app.pane.agent.mode, "edit")
            self.assertEqual(prompt.border_title, " edit ")
            self.assertIn("edit mode", str(app.query_one("#statusbar", Static).render()))

            await pilot.press("shift+tab", "shift+tab")
            self.assertEqual(app.pane.mode, "read")

    async def test_new_session_keeps_current_mode(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("shift+tab")
            app.action_new_session()
            self.assertEqual(app.pane.agent.mode, "edit")

    async def test_shift_tab_while_confirm_panel_open_keeps_pending_future(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = asyncio.ensure_future(app.pane._confirm("shell", {"command": "echo hi"}))
            await pilot.pause()
            await pilot.press("shift+tab")
            self.assertEqual(app.pane.mode, "edit")
            self.assertTrue(app.query(ConfirmPanel), "panel survives a mode switch")
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
        old = ResumeSessionTest._old_session("keep this")
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
        old = ResumeSessionTest._old_session()
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
            ids = [child.id for child in app.pane.children]
            self.assertEqual(ids, ["log", "response-status", "queued", "prompt"])
            # the status bar is app-wide, so it sits outside the pane
            self.assertEqual(app.query_one("#statusbar", Static).parent, app.screen)

            app.pane._set_status(True, " Counting mora… 3s")
            await pilot.pause()
            self.assertTrue(status.display)
            self.assertIn("3s", str(status.query_one(".status-label", Static).render()))

            app.pane._set_status(False)
            await pilot.pause()
            self.assertFalse(status.display)


class CacheHitStatusTest(AppTestCase):
    async def test_the_rate_accumulates_across_requests(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await app.pane._on_event(RequestStats(120, 2.5, 2000, 1600, 300))
            await pilot.pause()
            self.assertIn("cache hit 80%", str(app.query_one("#statusbar", Static).render()))

            # (1600 + 1500) / (2000 + 3000): the session total, not the last request
            await app.pane._on_event(RequestStats(100, 2.0, 3000, 1500, 500))
            await pilot.pause()
            self.assertIn("cache hit 62%", str(app.query_one("#statusbar", Static).render()))

    async def test_a_new_session_starts_a_fresh_count(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await app.pane._on_event(RequestStats(120, 2.5, 2000, 1600, 300))
            await pilot.pause()
            app.pane.new_session()
            await pilot.pause()
            self.assertNotIn("cache", str(app.query_one("#statusbar", Static).render()))

    async def test_no_rate_is_shown_when_the_provider_reports_none(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await app.pane._on_event(RequestStats(120, 2.5, 2000))
            await pilot.pause()
            bar = str(app.query_one("#statusbar", Static).render())
            self.assertNotIn("cache", bar)
            self.assertIn("tokens per second", bar)


class TodoPanelTest(AppTestCase):
    @staticmethod
    def _plan(*statuses: str) -> list[dict]:
        return [{"content": f"step {i}", "status": s} for i, s in enumerate(statuses)]

    async def test_burst_collapses_but_a_panel_with_output_under_it_stays(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            log = app.query_one("#log", VerticalScroll)
            app.pane._show_todos(self._plan("in_progress", "pending"))
            app.pane._show_todos(self._plan("completed", "in_progress"))
            await pilot.pause()
            panels = app.query(".todos")
            self.assertEqual(len(panels), 1, "consecutive revisions share one panel")
            self.assertIn("1/2", str(panels.first().render()))

            app.pane._add_tool_result("output")
            app.pane._show_todos(self._plan("completed", "completed"))
            await pilot.pause()
            panels = app.query(".todos")
            self.assertEqual(len(panels), 2, "the earlier plan is left as a snapshot")
            self.assertIn("1/2", str(panels.first().render()))
            self.assertIn("2/2", str(panels.last().render()))
            self.assertIs(log.children[-1], panels.last())

    async def test_clearing_removes_the_panel(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.pane._show_todos(self._plan("pending"))
            app.pane._show_todos([])
            await pilot.pause()
            self.assertEqual(len(app.query(".todos")), 0)
            self.assertIsNone(app.pane._todo_panel)


class EventCoverageTest(AppTestCase):
    """The TUI renderer is held to the same event list as the headless ones.

    An event the renderer has no branch for would silently vanish from the
    conversation log, so each one has to put something in it.
    """

    async def test_every_event_puts_something_in_the_log(self) -> None:
        app = self.make_app(config=Config(model="test-model", show_reasoning=True))
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
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


class ToolRenderingTest(AppTestCase):
    async def test_multiline_command_folds_to_its_first_line(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ToolStart("c1", "shell", {"command": "echo one\necho two"}))
            await pilot.pause()
            widget = app.query(ToolCall).first()
            body = str(widget.render())
            self.assertIn("echo one", body)
            self.assertNotIn("echo two", body)
            self.assertIn("+1 lines", body)
            widget.on_click()
            body = str(widget.render())
            self.assertIn("echo two", body)
            self.assertIn("click to collapse", body)

    async def test_single_line_command_has_no_fold(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ToolStart("c1", "shell", {"command": "ls"}))
            await pilot.pause()
            body = str(app.query(ToolCall).first().render())
            self.assertIn("ls", body)
            self.assertNotIn("click to expand", body)

    async def test_edit_call_shows_diff_expanded_by_default(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ToolStart("c1", "edit_file", {
                "path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}))
            await pilot.pause()
            widget = app.query(EditCall).first()
            header = widget.query_one(".edit-call-header", Static)
            diff = widget.query_one(".edit-call-diff", Static)
            self.assertIn("a.py", str(header.render()))
            self.assertTrue(diff.display)
            widget.on_click()
            self.assertFalse(diff.display)
            self.assertIn("click to expand", str(header.render()))
            widget.on_click()
            self.assertTrue(diff.display)

    async def test_folded_result_names_its_call(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ToolStart("c1", "shell", {"command": "ls src"}))
            await renderer.handle(ToolEnd("c1", "shell", "a.py\nb.py\nc.py"))
            await pilot.pause()
            body = str(app.query(ToolResult).first().render())
            self.assertIn("shell ls src", body)
            self.assertIn("3 lines", body)
            self.assertNotIn("a.py", body)


class ReasoningDisplayTest(AppTestCase):
    async def test_reasoning_rendered_when_enabled(self) -> None:
        app = self.make_app(config=Config(model="test-model", show_reasoning=True))
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ReasoningDelta("thinking hard"))
            await pilot.pause()
            widgets = app.query(".reasoning")
            self.assertEqual(len(widgets), 1)
            self.assertIn("thinking hard", str(widgets.first().render()))

    async def test_reasoning_folded_by_default(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ReasoningDelta("line one\nline two\nline three"))
            await pilot.pause()
            body = str(app.query(".reasoning").first().render())
            self.assertIn("reasoning", body)
            self.assertIn("3 lines", body)
            self.assertNotIn("line one", body)

    async def test_live_reasoning_folds_when_the_block_ends(self) -> None:
        app = self.make_app(config=Config(model="test-model", show_reasoning=True))
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            await renderer.handle(ReasoningDelta("line one\nline two"))
            await pilot.pause()
            widget = app.query(".reasoning").first()
            self.assertIn("line one", str(widget.render()))
            await renderer.close()
            await pilot.pause()
            self.assertNotIn("line one", str(widget.render()))

    async def test_toggle_flips_and_persists(self) -> None:
        app = self.make_app()
        async with app.run_test():
            app.action_toggle_reasoning()
            self.assertTrue(app.config.show_reasoning, "flips before the write lands")
            await app.workers.wait_for_complete()  # the save runs on a thread
            self.assertTrue(Config.load().show_reasoning)
            app.action_toggle_reasoning()
            await app.workers.wait_for_complete()
            self.assertFalse(Config.load().show_reasoning, "persisted to config.json")

    async def test_toggle_recap_flips_persists_and_drops_an_armed_recap(self) -> None:
        app = self.make_app()
        async with app.run_test():
            self.assertTrue(app.config.recap_enabled, "on by default")
            app.pane._used_tools = True
            app.pane._arm_recap()
            self.assertIsNotNone(app.pane._recap_timer)
            app.action_toggle_recap()
            self.assertFalse(app.config.recap_enabled)
            self.assertIsNone(app.pane._recap_timer, "an armed recap is dropped")
            await app.workers.wait_for_complete()  # the save runs on a thread
            self.assertFalse(Config.load().recap_enabled)
            app.action_toggle_recap()
            await app.workers.wait_for_complete()
            self.assertTrue(Config.load().recap_enabled)
            # Turning it back on arms nothing by itself: the next turn does.
            self.assertIsNone(app.pane._recap_timer)


class QueueTest(AppTestCase):
    async def test_queue_flush_and_cancel(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            queued = app.query_one("#queued", Static)
            self.assertFalse(prompt.disabled, "prompt stays enabled during turns")

            # prompts submitted while a (fake) turn runs are queued and shown
            hold_turn(app.pane)
            app.pane.handle_submit(PromptInput.Submitted("first message"))
            app.pane.handle_submit(PromptInput.Submitted("second message"))
            await pilot.pause()
            self.assertEqual(app.pane._queue, ["first message", "second message"])
            self.assertTrue(queued.display)

            # a finished turn flushes the queue into the next turn
            started: list[str] = []
            app.pane.job.submit = started.append
            await end_turn(app.pane)
            await pilot.pause()
            self.assertEqual(started, ["first message\n\nsecond message"])
            self.assertFalse(app.pane._queue)
            self.assertFalse(queued.display)

            # an interrupted turn hands the queue back to the input instead
            hold_turn(app.pane)
            app.pane.handle_submit(PromptInput.Submitted("queued later"))
            prompt.load_text("half-typed draft")
            await end_turn(app.pane, Outcome.INTERRUPTED)
            await pilot.pause()
            self.assertEqual(prompt.text, "queued later\nhalf-typed draft")
            self.assertFalse(app.pane._queue)
            self.assertEqual(started, ["first message\n\nsecond message"], "cancel must not auto-submit")


class InjectedQueueTest(AppTestCase):
    """A queued message joins the turn already running, at its next request."""

    async def test_the_queue_is_injected_rather_than_held_to_the_next_turn(self) -> None:
        app = self.make_app(mode="yolo")
        seen: list[list[object]] = []
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            requests += 1
            seen.append(list(messages))
            if requests == 1:
                # The user types while the model is still working.
                app.pane.handle_submit(PromptInput.Submitted("actually use uv"))
                yield {0: DeltaToolCall(name="shell", json_args='{"command": "echo hi"}',
                                        tool_call_id="call-1")}
            else:
                yield "done"

        with patch("paimon.agent.build_model", return_value=FunctionModel(stream_function=stream)):
            async with app.run_test() as pilot:
                queued = app.query_one("#queued", Static)
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await InterruptTest._wait_for(pilot, lambda: not app.pane.is_busy)

                self.assertEqual(requests, 2, "the tool result went back to the model")
                prompts = [part.content for message in seen[1] if isinstance(message, ModelRequest)
                           for part in message.parts if isinstance(part, UserPromptPart)]
                self.assertEqual(prompts, ["go", "actually use uv"])
                self.assertFalse(app.pane._queue)
                self.assertFalse(queued.display)
                self.assertEqual(
                    [str(widget.render()) for widget in app.pane.query(UserMessage)],
                    ["go", "actually use uv"],
                    "the injected message is drawn like any other user turn",
                )


class InterruptTest(AppTestCase):
    """Escape stops the turn in flight and leaves the conversation usable.

    Worth its own test because interrupting is no longer a Textual worker
    being cancelled: the pane asks its job, which cancels the task one turn
    runs in while the driver behind it stays parked on its inbox, ready for
    the next prompt.
    """

    @staticmethod
    def _endless_model() -> FunctionModel:
        async def stream(messages, info):
            await asyncio.sleep(30)
            yield "never gets here"  # pragma: no cover

        return FunctionModel(stream_function=stream)

    @staticmethod
    async def _wait_for(pilot, condition) -> None:
        for _ in range(200):
            await pilot.pause()
            if condition():
                return
        raise AssertionError("condition not reached")

    async def test_escape_stops_the_turn_and_the_pane_still_works(self) -> None:
        app = self.make_app()
        with patch("paimon.agent.build_model", return_value=self._endless_model()):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.pane.is_busy)

                await pilot.press("escape")
                await self._wait_for(pilot, lambda: not app.pane.is_busy)
                self.assertIs(app.pane.job.result.outcome, Outcome.INTERRUPTED)
                self.assertIn("Paimon stopped", MultiPaneTest._log_text(app.pane))

                # The driver survived, so the pane takes the next prompt.
                app.pane.handle_submit(PromptInput.Submitted("again"))
                await self._wait_for(pilot, lambda: app.pane.is_busy)
                self.assertIn("again", MultiPaneTest._log_text(app.pane),
                              "the second prompt is rendered from the job's own event")
                app.pane.interrupt()
                await self._wait_for(pilot, lambda: not app.pane.is_busy)


class FailedTurnQueueTest(AppTestCase):
    """A turn that errored is not a turn that finished: queued input stays put."""

    @staticmethod
    def _failing_model() -> FunctionModel:
        async def stream(messages, info):
            raise RuntimeError("provider failed")
            yield  # pragma: no cover - only here to make this a generator

        return FunctionModel(stream_function=stream)

    async def test_a_model_error_is_logged_rather_than_raised_at_the_app(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            with patch("paimon.agent.build_model", return_value=self._failing_model()):
                app.pane.job.submit("go")
                for _ in range(200):
                    await pilot.pause()
                    if not app.pane.is_busy:
                        break
            self.assertIs(app.pane.job.result.outcome, Outcome.FAILED)
            self.assertIn("provider failed", MultiPaneTest._log_text(app.pane),
                          "the error is shown in the log, not raised out of the driver")

    async def test_queue_returns_to_the_input_after_an_error(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            hold_turn(app.pane)
            app.pane.handle_submit(PromptInput.Submitted("queued while it ran"))
            await pilot.pause()

            started: list[str] = []
            app.pane.job.submit = started.append
            await end_turn(app.pane, Outcome.FAILED, error="provider failed")
            await pilot.pause()

            self.assertFalse(started, "a failed turn must not fire the queue at the model")
            self.assertEqual(prompt.text, "queued while it ran")
            self.assertFalse(app.pane._queue)


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


if __name__ == "__main__":
    unittest.main()


class MultiPaneTest(AppTestCase):
    """Opening, switching and closing panes."""

    @staticmethod
    def _log_text(pane: SessionPane) -> str:
        return " ".join(str(widget.render()) for widget in pane.query_one("#log").children)

    @staticmethod
    def _tab_text(app, pane: SessionPane) -> str:
        """A tab's drawn text with its frame and padding taken back out."""
        tab = app.query_one(f"#tab-{pane.id}", PaneTab)
        lines = str(tab.render()).splitlines()
        return " ".join(line.strip("╭╮╰╯┬┴│─ ") for line in lines).strip()

    async def test_new_pane_opens_a_second_session_and_shows_the_strip(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.pane
            self.assertFalse(app._tabs.display, "one pane needs no strip")
            await pilot.press("ctrl+t")
            await pilot.pause()

            second = app.pane
            self.assertEqual(app.panes, [first, second])
            self.assertNotEqual(second.agent.session.id, first.agent.session.id)
            self.assertTrue(app._tabs.display)
            self.assertFalse(first.display, "only the current pane is shown")
            self.assertTrue(second.display)
            self.assertIs(app.focused, second.query_one(PromptInput))

    async def test_new_pane_inherits_cwd_and_mode(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("shift+tab")  # read -> edit
            await pilot.press("ctrl+t")
            await pilot.pause()
            self.assertEqual(app.pane.mode, "edit")
            self.assertEqual(app.pane.agent.mode, "edit")
            self.assertEqual(app.pane.agent.cwd, app.panes[0].agent.cwd)

    async def test_cycling_wraps_in_both_directions(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.press("ctrl+t")
            await pilot.pause()
            self.assertIs(app.pane, app.panes[2])

            await pilot.press("ctrl+pagedown")
            self.assertIs(app.pane, app.panes[0])
            await pilot.press("ctrl+pageup")
            self.assertIs(app.pane, app.panes[2])
            await pilot.press("ctrl+pageup")
            self.assertIs(app.pane, app.panes[1])

    async def test_clicking_a_tab_switches_to_it(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.pane
            await pilot.press("ctrl+t")
            await pilot.pause()
            await pilot.click(app.query_one(f"#tab-{first.id}", PaneTab))
            await pilot.pause()
            self.assertIs(app.pane, first)
            self.assertTrue(first.display)

    async def test_tab_labels_number_the_panes_and_show_their_title(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            app.panes[0]._title = "fix   the parser"
            app._sync_panes()
            await pilot.pause()
            labels = [self._tab_text(app, pane) for pane in app.panes]
            self.assertEqual(labels, ["1 fix the parser", "2 new session"])

    async def test_closing_a_pane_unlocks_it_and_falls_back_to_a_neighbour(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.pane
            await pilot.press("ctrl+t")
            await pilot.pause()
            closed = app.pane.agent.session.path
            self.assertTrue(lockfile.held(closed))

            await pilot.press("ctrl+w")
            await pilot.pause()
            self.assertFalse(lockfile.held(closed), "a closed pane releases its session")
            self.assertEqual(app.panes, [first])
            self.assertIs(app.pane, first)
            self.assertTrue(first.display)
            self.assertFalse(app._tabs.display)
            self.assertEqual([tab.pane for tab in app.query(PaneTab)], [first],
                             "the strip drops the closed tab")

    async def test_closing_a_pane_mid_turn_leaves_nothing_behind(self) -> None:
        # The turn is cancelled while its widgets are being removed, so the
        # worker must unwind without touching them.
        app = self.make_app()
        with patch("paimon.agent.build_model",
                   return_value=stub_model("shell", '{"command": "rm x"}')):
            async with app.run_test() as pilot:
                await pilot.press("ctrl+t")
                await pilot.pause()
                pane = app.pane
                pane.handle_submit(PromptInput.Submitted("go"))
                for _ in range(200):
                    await pilot.pause()
                    if pane.needs_confirm:
                        break
                else:
                    raise AssertionError("confirm panel never appeared")

                await pilot.press("ctrl+w")
                await pilot.pause()
                self.assertEqual(len(app.panes), 1)
                self.assertFalse(lockfile.held(pane.agent.session.path))
                self.assertNotIn("awaiting confirmation",
                                 str(app.query_one("#statusbar", Static).render()))

    async def test_the_last_pane_stays_open(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+w")
            await pilot.pause()
            self.assertEqual(len(app.panes), 1)
            self.assertIn("last pane stays open", self._log_text(app.pane))

    async def test_pane_count_is_capped(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            for _ in range(MAX_PANES + 1):
                await pilot.press("ctrl+t")
            await pilot.pause()
            self.assertEqual(len(app.panes), MAX_PANES)
            self.assertIn(f"Already at {MAX_PANES} panes", self._log_text(app.pane))


class PaneAttentionTest(AppTestCase):
    """A pane waiting for permission is the one thing the user must not miss."""

    async def test_a_confirmation_elsewhere_is_announced_and_reachable(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.panes[0]
            await pilot.press("ctrl+t")
            await pilot.pause()

            task = asyncio.ensure_future(first._confirm("shell", {"command": "rm x"}))
            await pilot.pause()
            self.assertTrue(first.needs_confirm)
            self.assertIn("1 awaiting confirmation",
                          str(app.query_one("#statusbar", Static).render()))
            tab = app.query_one(f"#tab-{first.id}", PaneTab)
            self.assertTrue(tab.has_class("-attention"))
            self.assertIn("!", str(tab.render()))

            await pilot.press("ctrl+g")
            await pilot.pause()
            self.assertIs(app.pane, first)
            # The prompt is hidden under the panel, so the panel takes the keys.
            self.assertIs(app.focused, first.query_one(ConfirmPanel))
            await pilot.press("enter")
            self.assertTrue(await task)

            await pilot.pause()
            self.assertFalse(first.needs_confirm)
            self.assertNotIn("awaiting confirmation",
                             str(app.query_one("#statusbar", Static).render()))

    async def test_goto_attention_does_nothing_when_nothing_waits(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.press("ctrl+t")
            await pilot.pause()
            current = app.pane
            await pilot.press("ctrl+g")
            self.assertIs(app.pane, current)


class TabDrawingTest(AppTestCase):
    """The frames the tabs draw for themselves."""

    async def _strip(self, app, pilot, panes: int = 2) -> list:
        for _ in range(panes - 1):
            await pilot.press("ctrl+t")
        await pilot.pause()
        return [str(app.query_one(f"#tab-{pane.id}", PaneTab).render()).splitlines()
                for pane in app.panes]

    async def test_the_current_tab_is_framed_and_meets_the_rule(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first, second = await self._strip(app, pilot)
            self.assertEqual(len(first), 3)
            self.assertEqual(len(second), 3)
            # The second pane is the current one. Its frame opens downwards
            # from the rule the idle tabs carry.
            self.assertTrue(second[0].startswith("┬") and second[0].endswith("┬"))
            self.assertTrue(second[2].startswith("╰") and second[2].endswith("╯"))
            self.assertEqual(first[0], "─" * len(first[0]),
                             "an idle tab contributes plain rule")
            self.assertEqual(len(first[0]), len(second[0]),
                             "every tab is the same width, so the rule lines up")

    async def test_only_a_visible_strip_takes_the_status_bars_bottom_margin(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            bar = app.query_one("#statusbar", Static)
            self.assertFalse(app.screen.has_class("-tabs-bottom"),
                             "one pane means no strip and no rule to sit over")
            self.assertEqual((bar.styles.margin.bottom, bar.styles.margin.left), (1, 2))

            await pilot.press("ctrl+t")
            await pilot.pause()
            self.assertTrue(app.screen.has_class("-tabs-bottom"))
            # The side margins have to survive: they are what lines the bar up
            # with the prompt above it and the strip's rule below.
            self.assertEqual((bar.styles.margin.bottom, bar.styles.margin.left), (0, 2))

            await pilot.press("ctrl+w")
            await pilot.pause()
            self.assertFalse(app.screen.has_class("-tabs-bottom"))
            self.assertEqual((bar.styles.margin.bottom, bar.styles.margin.left), (1, 2))

    async def test_the_fill_stays_last_so_the_rule_reaches_the_edge(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._strip(app, pilot, panes=3)
            self.assertIs(app._tabs.children[-1], app._tabs._fill)

    async def test_tabs_shrink_so_all_eight_fit(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            rows = await self._strip(app, pilot, panes=8)
            widths = [len(row[1]) for row in rows]
            self.assertEqual(len(set(widths)), 1, "all tabs share one width")
            self.assertLessEqual(sum(widths), 76, "the whole strip fits the terminal")


class PaneSessionLockTest(AppTestCase):
    async def test_resume_hides_a_session_open_in_another_pane(self) -> None:
        old = ResumeSessionTest._old_session()
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
            self.assertIn("No sessions to resume", MultiPaneTest._log_text(app.pane))


class SpawnAgentTest(AppTestCase):
    """spawn_agent in the UI: a second pane nobody asked to look at."""

    @staticmethod
    def _spawning_model() -> FunctionModel:
        return stub_model("spawn_agent", '{"prompt": "check the parser"}')

    @staticmethod
    async def _wait_for(pilot, condition) -> None:
        for _ in range(200):
            await pilot.pause()
            if condition():
                return
        raise AssertionError("condition not reached")

    async def _spawn(self, app: PaimonApp, pilot) -> SessionPane:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: len(app.panes) == 2)
        return app.panes[1]

    async def test_the_new_pane_stays_in_the_background(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                prompt = parent.query_one(PromptInput)
                child = await self._spawn(app, pilot)

                self.assertIs(app.pane, parent, "spawning does not switch panes")
                self.assertFalse(child.display)
                self.assertIs(app.focused, prompt, "a pane the user did not open takes no keys")
                self.assertEqual(child.agent.cwd, parent.agent.cwd)
                self.assertEqual(child.mode, parent.mode)
                self.assertIn(child.job.job_id, MultiPaneTest._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertIn(f"{child.job.job_id} check the parser",
                              MultiPaneTest._tab_text(app, child))

    async def test_the_new_agent_cannot_spawn_or_hand_off(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                child = await self._spawn(app, pilot)
                self.assertNotIn("spawn_agent", child.agent.toolset, "depth stays 1")
                self.assertNotIn("start_new_session", child.agent.toolset,
                                 "a handoff would swap the session out from under its id")
                self.assertNotIn("run_background", child.agent.toolset,
                                 "only the conversation the user is in leaves processes behind")
                for name in ("read_job", "wait_for_job", "stop_job", "send_to_agent"):
                    self.assertNotIn(name, child.agent.toolset,
                                     "it can start nothing, so it has nothing to look at")

    async def test_the_new_session_is_a_child_and_stays_out_of_the_listings(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)

                self.assertEqual(child.agent.session.parent_id, parent.agent.session.id)
                listed = [session.id for session in Session.list(parent.agent.cwd)]
                self.assertNotIn(child.agent.session.id, listed)
                self.assertIn(child.agent.session.id,
                              [s.id for s in Session.list(parent.agent.cwd, include_children=True)])

    async def test_a_typed_spawn_narrows_tools_and_appends_the_prompt(self) -> None:
        app = self.make_app(mode="yolo")
        model = stub_model("spawn_agent", '{"prompt": "map the modules", "agent": "explore"}')
        with patch("paimon.agent.build_model", return_value=model):
            async with app.run_test() as pilot:
                app.pane.agent.model_override = "test:override"
                child = await self._spawn(app, pilot)

                expected = {name for name, tool in tools.REGISTRY.items()
                            if tool.access in ("read", "none")
                            and name not in tools.SUBAGENT_DENIED}
                self.assertEqual(set(child.agent.toolset), expected)
                self.assertTrue(child.agent.system_prompt.rstrip().endswith(
                    agents.builtin_types()[0].body),
                    "the type's body ends the child's system prompt")
                self.assertEqual(child.agent.model_override, "test:override",
                                 "with no explicit model the caller's override carries over")

    async def test_a_finished_child_wakes_the_parent(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)

                # The child's stub turn ends on its own; the parent is then
                # woken without anybody typing, reports the news and reacts.
                await self._wait_for(
                    pilot, lambda: f"Agents: {child.job.job_id} finished"
                    in MultiPaneTest._log_text(parent))
                await self._wait_for(pilot, lambda: not parent.is_busy)
                self.assertTrue(any(is_agents_message(message)
                                    for message in parent.agent.history))
                self.assertNotIn(f"{child.job.job_id} finished",
                                 parent.agent.session.first_user_text() or "",
                                 "the wake-up never becomes the session title")

    async def test_an_unknown_type_reports_and_opens_no_pane(self) -> None:
        app = self.make_app(mode="yolo")
        model = stub_model("spawn_agent", '{"prompt": "go", "agent": "nope"}')
        with patch("paimon.agent.build_model", return_value=model):
            async with app.run_test() as pilot:
                parent = app.pane
                parent.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(
                    pilot, lambda: "unknown agent type" in MultiPaneTest._log_text(parent))
                self.assertIn("'nope'", MultiPaneTest._log_text(parent))
                self.assertEqual(len(app.panes), 1)

    async def test_changing_the_parents_session_stops_its_agents(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                agent_id, path = child.job.job_id, child.agent.session.path
                await self._wait_for(pilot, lambda: not parent.is_busy)

                parent.new_session()
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

                self.assertFalse(lockfile.held(path), "the stopped agent released its session")
                log = MultiPaneTest._log_text(parent)
                self.assertIn("Stopped 1 agent", log)
                self.assertIn(agent_id, log)
                self.assertIs(app.pane, parent)

    async def test_closing_an_agents_pane_leaves_its_output_readable(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not child.is_busy)
                app._switch_to(child)

                await pilot.press("ctrl+w")
                await pilot.pause()

                answer = await app._supervisor.handle(
                    "read_job", {"job_id": child.job.job_id, "mode": "all"},
                    caller=parent.agent)
                self.assertIn("killed", answer)
                self.assertIn("done", answer, "what it managed to say survives its pane")

    async def test_closing_the_parent_of_the_only_other_pane_leaves_one_open(self) -> None:
        # Closing a pane stops the agents it started, so both panes can go at
        # once; the app has to be left with a conversation either way.
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                await pilot.press("ctrl+w")
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

                self.assertNotIn(app.pane, (parent, child))
                self.assertTrue(app.pane.display)
                self.assertIs(app.focused, app.pane.query_one(PromptInput))
                self.assertFalse(lockfile.held(child.agent.session.path))

    async def test_the_next_turn_opens_with_what_the_agents_did(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy and not child.is_busy)

                parent.handle_submit(PromptInput.Submitted("anything new?"))
                await self._wait_for(pilot, lambda: not parent.is_busy)

                self.assertIn(f"Agents: {child.job.job_id} finished",
                              MultiPaneTest._log_text(parent))
                self.assertTrue(any(is_agents_message(message)
                                    for message in parent.agent.history),
                                "the model only learns of it through the history")


class BackgroundTaskTest(AppTestCase):
    """run_background in the UI: a process with a tab and no keyboard."""

    COMMAND = "printf 'pid %s\\n' $$; sleep 30"

    def _model(self, command: str | None = None) -> FunctionModel:
        return stub_model("run_background", json.dumps(
            {"command": command or self.COMMAND, "description": "dev server"}))

    @staticmethod
    async def _wait_for(pilot, condition) -> None:
        """Like SpawnAgentTest's, but it lets real time pass.

        A task pane collects its output on an interval, so a condition that
        depends on one has to be given the wall clock, not just message-loop
        round trips.
        """
        for _ in range(200):
            await pilot.pause()
            if condition():
                return
            await asyncio.sleep(0.02)
        raise AssertionError("condition not reached")

    @staticmethod
    def _pid(pane: CommandPane) -> int:
        match = re.search(r"pid (\d+)", pane.command.output.since(0)[0].decode())
        assert match, "the command never printed its pid"
        return int(match.group(1))

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    async def _start(self, app: PaimonApp, pilot) -> CommandPane:
        app.pane.handle_submit(PromptInput.Submitted("run the dev server"))
        await self._wait_for(pilot, lambda: len(app.panes) == 2)
        pane = app.panes[1]
        await self._wait_for(pilot, lambda: pane.command.output.total_bytes > 0)
        self.addCleanup(pane.command.terminate_now)
        return pane

    async def test_the_command_runs_in_a_background_tab(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)

                self.assertIsInstance(task, CommandPane)
                self.assertIs(app.pane, parent, "starting a task does not switch panes")
                self.assertFalse(task.display)
                self.assertIs(app.focused, parent.query_one(PromptInput),
                              "a pane the user did not open takes no keys")
                self.assertIn(f"{task.job.job_id} dev server",
                              MultiPaneTest._tab_text(app, task))
                self.assertIn(task.job.job_id, MultiPaneTest._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertTrue(task.is_running)

    async def test_its_output_reaches_the_agent_and_the_tab_it_opens(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)

                answer = await app._supervisor.handle(
                    "read_job", {"job_id": task.job.job_id}, caller=parent.agent)
                self.assertIn("pid", answer)
                self.assertIn("running", answer)
                self.assertNotIn("pid", self._log_text(task),
                                 "a hidden tab writes nothing; RichLog would defer it all")

                app._switch_to(task)
                await self._wait_for(pilot, lambda: "pid" in self._log_text(task))
                self.assertIn(self.COMMAND.split(";")[0].strip(), self._log_text(task),
                              "the tab opens with the command it is running")

    @staticmethod
    def _log_text(pane: CommandPane) -> str:
        return "\n".join(strip.text for strip in pane.query_one("#log", RichLog).lines)

    async def test_the_status_bar_follows_the_task(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                app._switch_to(task)
                await pilot.pause()

                bar = str(app.query_one("#statusbar", Static).render())
                self.assertIn(f"command {task.job.job_id}", bar)
                self.assertIn("running", bar)

    async def test_closing_the_tab_stops_the_command(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                pid = self._pid(task)
                app._switch_to(task)

                await pilot.press("ctrl+w")
                await self._wait_for(pilot, lambda: len(app.panes) == 1)
                await self._wait_for(pilot, lambda: not self._alive(pid))
                self.assertTrue(task.command.killed)

    async def test_quitting_does_not_leave_the_process_group_behind(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                pid = self._pid(task)
                self.assertTrue(self._alive(pid))

            for _ in range(100):
                if not self._alive(pid):
                    break
                await asyncio.sleep(0.05)
            else:
                self.fail(f"{pid} outlived the app that started it")

    async def test_an_exit_ends_the_tab_without_closing_it(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model",
                   return_value=self._model("printf 'built\\n'; exit 1")):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                await self._wait_for(pilot, lambda: not task.is_running)

                self.assertEqual(len(app.panes), 2, "the tab stays, so the output can be read")
                self.assertEqual(task.status_text, "exited (code 1)")
                app._switch_to(task)
                await self._wait_for(pilot, lambda: "built" in self._log_text(task))
                self.assertIn("exited (code 1)",
                              str(app.query_one("#statusbar", Static).render()))

    async def test_an_agent_can_still_be_started_alongside_a_task(self) -> None:
        # Only a conversation has an agent, and finding the one that asked
        # walks the pane list, task panes included.
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                await self._start(app, pilot)

                answer = await app._supervisor.handle(
                    "spawn_agent", {"prompt": "check the parser"}, caller=parent.agent)
                await self._wait_for(pilot, lambda: len(app.panes) == 3)
                self.assertIn("Started agent", answer)
                self.assertEqual(app.panes[2].agent.cwd, parent.agent.cwd)

    async def test_a_denied_confirmation_starts_nothing(self) -> None:
        app = self.make_app(mode="read")
        with patch("paimon.agent.build_model", return_value=self._model("ls -la")):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("run it"))
                await self._wait_for(pilot, lambda: bool(app.query(ConfirmPanel)))
                self.assertIn("Runs in its own tab",
                              str(app.query_one(ConfirmPanel).query_one("#confirm-detail")
                                  .children[0].render()))

                await pilot.press("escape")
                await self._wait_for(pilot, lambda: not app.pane.is_busy)
                self.assertEqual(len(app.panes), 1, "a safe-looking command is confirmed too")


class RecapTest(AppTestCase):
    """A turn that did some work, then silence: Paimon says where things stand.

    The waits are real, the way the task-pane tests do it: nothing in the app
    fakes a clock, so neither does this.
    """

    RECAP = "读了 missing.txt；下一步把结果写回去"

    @staticmethod
    def _model(*, tool: bool = True, recap: str = RECAP, boom: bool = False) -> FunctionModel:
        """One tool call for the turn, a separate answer for the recap."""
        requests = 0

        async def stream(messages, info):
            nonlocal requests
            content = getattr(messages[-1].parts[-1], "content", "")
            if isinstance(content, str) and aside.RECAP_INSTRUCTIONS in content:
                if boom:
                    raise ModelHTTPError(401, "stub")
                yield recap
                return
            requests += 1
            if tool and requests == 1:
                yield {0: DeltaToolCall(name="read_file", json_args='{"path": "missing.txt"}',
                                        tool_call_id="call-1")}
            else:
                yield "done"

        return FunctionModel(stream_function=stream)

    @staticmethod
    def _config(idle: float = 0.05, enabled: bool = True) -> Config:
        return Config(model="test-model", recap_idle_seconds=idle, recap_enabled=enabled)

    @staticmethod
    async def _wait_for(pilot, condition) -> None:
        """Let real time pass: a recap is scheduled on a timer."""
        for _ in range(200):
            await pilot.pause()
            if condition():
                return
            await asyncio.sleep(0.02)
        raise AssertionError("condition not reached")

    @staticmethod
    async def _stays_away(pilot, condition, rounds: int = 20) -> None:
        for _ in range(rounds):
            await pilot.pause()
            await asyncio.sleep(0.02)
            if condition():
                raise AssertionError("a recap turned up where none was due")

    @staticmethod
    def _recap_text(app: PaimonApp) -> str:
        # A recap is a Markdown container: its words live in the blocks it
        # mounted, not in the container's own render.
        return " ".join(
            str(block.render())
            for recap in app.query(RecapMessage)
            for block in recap.query(MarkdownBlock)
        )

    async def _finish_a_turn(self, app: PaimonApp, pilot) -> None:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: not app.pane.is_busy)

    async def test_a_recap_follows_a_tool_turn_that_goes_quiet(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                await self._wait_for(pilot, lambda: bool(app.query(RecapMessage)))

                self.assertIn(self.RECAP, self._recap_text(app))
                self.assertIn("While you were away", self._recap_text(app))

    async def test_the_recap_is_not_part_of_the_conversation(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                agent = app.pane.agent
                messages = len(agent.history)
                lines = agent.session.path.read_text().count("\n")

                await self._wait_for(pilot, lambda: bool(app.query(RecapMessage)))

                self.assertEqual(len(agent.history), messages)
                self.assertEqual(agent.session.path.read_text().count("\n"), lines)
                self.assertNotIn(self.RECAP, agent.session.path.read_text())

    async def test_a_turn_that_only_answered_gets_no_recap(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model(tool=False)):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)

                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))

    async def test_typing_restarts_the_countdown(self) -> None:
        # A wait long enough that the assertions are about the timer rather
        # than about how fast the machine running them is.
        app = self.make_app(config=self._config(idle=30))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                armed = app.pane._recap_timer
                self.assertIsNotNone(armed)

                await pilot.press("h")
                await pilot.pause()

                self.assertIsNotNone(app.pane._recap_timer)
                self.assertIsNot(app.pane._recap_timer, armed)
                self.assertFalse(app.query(RecapMessage))

    async def test_a_new_turn_drops_a_pending_recap(self) -> None:
        app = self.make_app(config=self._config(idle=30))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                self.assertIsNotNone(app.pane._recap_timer)

                app.pane.handle_submit(PromptInput.Submitted("more"))
                await pilot.pause()

                self.assertIsNone(app.pane._recap_timer)
                await self._wait_for(pilot, lambda: not app.pane.is_busy)

    async def test_a_failed_recap_says_nothing(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model(boom=True)):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)
                # Nothing at all is added, not even a note about the failure:
                # the recap was never asked for, so it does not get to complain.
                shown = len(app.query_one("#log").children)

                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))
                self.assertEqual(len(app.query_one("#log").children), shown)

    async def test_only_the_pane_on_screen_recaps(self) -> None:
        app = self.make_app(config=self._config())
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                first = app.pane
                first.handle_submit(PromptInput.Submitted("go"))
                await app.action_new_pane()
                await self._wait_for(pilot, lambda: not first.is_busy)

                self.assertIsNot(app.pane, first)
                self.assertIsNone(first._recap_timer)
                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))

    async def test_it_can_be_turned_off(self) -> None:
        app = self.make_app(config=self._config(enabled=False))
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                await self._finish_a_turn(app, pilot)

                self.assertIsNone(app.pane._recap_timer)
                await self._stays_away(pilot, lambda: bool(app.query(RecapMessage)))


if __name__ == "__main__":
    unittest.main()


class SkillPaletteTest(AppTestCase):
    async def test_palette_entry_types_the_command_and_replay_folds_the_block(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "demo" / "SKILL.md"
        path.parent.mkdir()
        path.write_text("---\nname: demo\ndescription: Demo skill.\n---\nLine one\nLine two")
        config = Config(model="test-model", skills=[str(path.parent)])
        app = self.make_app(config=config)
        async with app.run_test() as pilot:
            commands = {c.title: c for c in app.get_system_commands(app.screen)}
            self.assertIn("Skill: demo", commands)
            self.assertEqual(commands["Skill: demo"].help, "Demo skill.")
            commands["Skill: demo"].callback()
            await pilot.pause()
            self.assertEqual(app.pane.query_one(PromptInput).text, "/skill:demo ")

            body = skills.expand_skill_command("/skill:demo and more", app.pane.agent.skills)
            app.pane._add_user(body)
            await pilot.pause()
            folded = app.pane.query(".skill-invocation")
            self.assertEqual(len(folded), 1)
            self.assertEqual(str(app.pane.query(UserMessage).last().render()), "and more")
