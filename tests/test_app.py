import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textual.containers import Horizontal
from textual.widgets import Static
from textual.worker import WorkerState

from paimon import config
from paimon.app import PaimonApp, _session_label
from paimon.login import PickerScreen
from paimon.session import Session
from paimon.ui import ConfirmPanel, PromptInput, UserMessage


class AppTestCase(unittest.IsolatedAsyncioTestCase):
    """Pilot-driven TUI tests against an isolated data dir and a stub model."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict("os.environ", {"PAIMON_DATA_HOME": tmp.name})
        model = patch.object(config, "MODEL", "test-model")
        for patcher in (env, model):
            patcher.start()
            self.addCleanup(patcher.stop)


class ConfirmPanelTest(AppTestCase):
    @staticmethod
    async def _open(app: PaimonApp, pilot, tool: str = "bash", args: dict | None = None) -> asyncio.Future:
        task = asyncio.ensure_future(app._confirm(tool, args or {"command": "echo hi"}))
        await pilot.pause()
        return task

    async def test_enter_allows_and_restores_prompt(self) -> None:
        app = PaimonApp()
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
        app = PaimonApp()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, args={"command": "rm -rf x"})
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_always_allows_for_the_rest_of_the_session(self) -> None:
        app = PaimonApp()
        async with app.run_test() as pilot:
            task = await self._open(
                app, pilot, "edit_file", {"path": "a.py", "old_string": "a", "new_string": "b"}
            )
            await pilot.press("down", "enter")
            self.assertTrue(await task)
            self.assertIn("edit_file", app._session_allowed)
            self.assertTrue(await app._confirm("edit_file", {"path": "b.py"}))
            self.assertFalse(app.query("#confirm-panel"))

    async def test_number_shortcut_denies(self) -> None:
        app = PaimonApp()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, "write_file", {"path": "c.py", "content": "x"})
            await pilot.press("3")
            self.assertFalse(await task)


class ModeCycleTest(AppTestCase):
    async def test_shift_tab_cycles_mode_and_updates_indicators(self) -> None:
        app = PaimonApp()
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
        app = PaimonApp()
        async with app.run_test() as pilot:
            await pilot.press("shift+tab")
            app.action_new_session()
            self.assertEqual(app.agent.mode, "edit")

    async def test_shift_tab_while_confirm_panel_open_keeps_pending_future(self) -> None:
        app = PaimonApp()
        async with app.run_test() as pilot:
            task = asyncio.ensure_future(app._confirm("bash", {"command": "echo hi"}))
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
        session.append_message({"role": "user", "content": content})
        return session

    def _log_text(self, app: PaimonApp) -> str:
        return " ".join(str(widget.render()) for widget in app.query_one("#log").children)

    async def test_palette_resume_swaps_agent_and_renders_history(self) -> None:
        old = self._old_session()
        app = PaimonApp()
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
        app = PaimonApp()
        async with app.run_test() as pilot:
            before = app.agent
            app._turn = SimpleNamespace(is_running=True)
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertIs(app.agent, before)

    async def test_no_sessions_shows_notice(self) -> None:
        app = PaimonApp()
        async with app.run_test() as pilot:
            app.action_resume_session()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)
            self.assertIn("No sessions to resume", self._log_text(app))

    async def test_constructor_session_param_resumes_on_mount(self) -> None:
        old = self._old_session()
        app = PaimonApp(session=old)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(app.agent.session.id, old.id)
            self.assertTrue(app.query(UserMessage))
            self.assertIn("Resumed session", self._log_text(app))


class SessionLabelTest(AppTestCase):
    def test_label_has_local_time_short_id_and_flattened_preview(self) -> None:
        session = Session.create(Path.cwd())
        session.append_message({"role": "user", "content": "fix the\nbug " + "x" * 50})

        label = _session_label(session)

        when = datetime.fromisoformat(session.created_at()).astimezone().strftime("%m-%d %H:%M")
        preview = " ".join(("fix the\nbug " + "x" * 50).split())
        self.assertEqual(label, f"{when} · {session.id[:8]} · {preview[:40]}…")


class StatusLineTest(AppTestCase):
    async def test_pinned_status_layout_and_toggle(self) -> None:
        app = PaimonApp()
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


class QueueTest(AppTestCase):
    async def test_queue_flush_and_cancel(self) -> None:
        app = PaimonApp()
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


if __name__ == "__main__":
    unittest.main()
