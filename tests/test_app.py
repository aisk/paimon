import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.containers import Horizontal
from textual.widgets import Static
from textual.worker import WorkerState

from paimon import config
from paimon.app import PaimonApp
from paimon.ui import ConfirmPanel, PromptInput


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
    async def test_allow_deny_always_and_shortcuts(self) -> None:
        app = PaimonApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)

            # Enter on the default option allows
            task = asyncio.ensure_future(app._confirm("bash", {"command": "echo hi"}))
            await pilot.pause()
            panel = app.query_one("#confirm-panel", ConfirmPanel)
            self.assertFalse(prompt.display, "prompt hidden while confirming")
            self.assertIs(app.focused, panel)
            await pilot.press("enter")
            self.assertTrue(await task)
            await pilot.pause()
            self.assertFalse(app.query("#confirm-panel"))
            self.assertTrue(prompt.display)

            # Esc denies
            task = asyncio.ensure_future(app._confirm("bash", {"command": "rm -rf x"}))
            await pilot.pause()
            await pilot.press("escape")
            self.assertFalse(await task)

            # Down+Enter picks "always": allowed now and for the session
            task = asyncio.ensure_future(
                app._confirm("edit_file", {"path": "a.py", "old_string": "a", "new_string": "b"})
            )
            await pilot.pause()
            await pilot.press("down", "enter")
            self.assertTrue(await task)
            self.assertIn("edit_file", app._session_allowed)
            self.assertTrue(await app._confirm("edit_file", {"path": "b.py"}))
            self.assertFalse(app.query("#confirm-panel"))

            # Number shortcut: 3 = deny
            task = asyncio.ensure_future(app._confirm("write_file", {"path": "c.py", "content": "x"}))
            await pilot.pause()
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
