import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static

from paimon import skills
from paimon.agent import ReasoningDelta, RequestStats, ToolEnd, ToolStart
from paimon.config import Config
from paimon.pane import _EventRenderer, _session_label
from paimon.session import (
    Session,
)
from paimon.ui import (
    EditCall,
    PromptInput,
    ToolCall,
    ToolResult,
    UserMessage,
)
from tests.support.agent import SILENT_EVENTS, agent_events
from tests.support.app import AppTestCase


class SessionLabelTest(unittest.TestCase):
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
