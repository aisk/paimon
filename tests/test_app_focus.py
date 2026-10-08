"""Keyboard ownership: user intent wins over asynchronous pane updates."""
import asyncio

from textual.widgets import Input, Static

from paimon.login import PickerScreen
from paimon.ui import ConfirmPanel, PromptInput, QuestionPanel
from tests.support.app import AppTestCase, end_turn, hold_turn


class FocusTest(AppTestCase):
    async def test_turn_end_preserves_reading_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            log = app.pane.transcript
            log.focus()
            await pilot.pause()
            await end_turn(app.pane)
            await pilot.pause()
            self.assertIs(app.focused, log)
            self.assertIn("Focus: reading", str(app.query_one("#statusbar", Static).render()))
            await pilot.press("ctrl+l")
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_tab_switch_restores_reading_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.pane
            first.transcript.focus()
            await pilot.pause()
            await pilot.press("ctrl+t", "ctrl+pageup")
            self.assertIs(app.pane, first)
            self.assertIs(app.focused, first.transcript)

    async def test_confirm_close_restores_prompt(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open_confirm(app, pilot)
            await pilot.press("enter")
            self.assertTrue(await task)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_confirm_arrival_preserves_deliberate_reading_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            app.pane.transcript.focus()
            await pilot.pause()
            task = await self._open_confirm(app, pilot)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.transcript)
            self.assertTrue(app.pane.needs_confirm)
            await pilot.press("ctrl+l")
            self.assertIs(app.focused, app.pane.query_one(ConfirmPanel))
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_confirm_close_preserves_reading_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open_confirm(app, pilot)
            panel = app.pane.query_one(ConfirmPanel)
            app.pane.transcript.focus()
            await pilot.pause()
            panel._resolve("deny")
            self.assertFalse(await task)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.transcript)

    async def test_panel_during_modal_does_not_steal_keyboard(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            dialog = PickerScreen("Choose", ["one", "two"])
            app.push_screen(dialog)
            await pilot.pause()
            focused = app.focused
            task = await self._open_confirm(app, pilot)
            self.assertIs(app.focused, focused)
            self.assertIn(dialog, app.screen_stack)
            await pilot.press("ctrl+t", "ctrl+w", "ctrl+l")
            self.assertEqual(len(app.panes), 1)
            self.assertIs(app.focused, focused)
            app.pane.query_one(ConfirmPanel)._resolve("deny")
            self.assertFalse(await task)
            await pilot.pause()
            self.assertIs(app.focused, focused)
            dialog.dismiss(None)
            await pilot.pause()
            await pilot.press("ctrl+l")
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_modal_return_preserves_reading_with_existing_question(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open_confirm(app, pilot)
            app.pane.transcript.focus()
            await pilot.pause()
            dialog = PickerScreen("Choose", ["one"])
            app.push_screen(dialog)
            await pilot.pause()
            dialog.dismiss(None)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.transcript)
            await pilot.press("ctrl+l", "escape")
            self.assertFalse(await task)

    async def test_modal_return_recovers_previously_missing_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            app.screen.set_focus(None)
            dialog = PickerScreen("Choose", ["one"])
            app.push_screen(dialog)
            await pilot.pause()
            dialog.dismiss(None)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_modal_return_focuses_question_that_arrived_behind_it(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            dialog = PickerScreen("Choose", ["one"])
            app.push_screen(dialog)
            await pilot.pause()
            task = await self._open_confirm(app, pilot)
            dialog.dismiss(None)
            await pilot.pause()
            self.assertIs(app.focused, app.pane.query_one(ConfirmPanel))
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_click_answer_synchronizes_selected_option(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            task = asyncio.create_task(app.pane._ask("Which?", ["A", "B"]))
            await self._wait_for_panel(app, pilot)
            panel = app.pane.query_one(QuestionPanel)
            await pilot.click("#question-input")
            self.assertIsInstance(app.focused, Input)
            self.assertEqual(panel._selected, panel._other)
            self.assertIn("Focus: answer", str(app.query_one("#statusbar", Static).render()))
            await pilot.press("x", "enter")
            self.assertEqual(await task, "x")
            await pilot.pause()
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_question_detail_navigation_does_not_change_answer(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            task = asyncio.create_task(app.pane._ask("line\n" * 30, ["A", "B"]))
            await self._wait_for_panel(app, pilot)
            panel = app.pane.query_one(QuestionPanel)
            detail = panel.query_one("#question-detail")
            detail.focus()
            await pilot.pause()
            await pilot.press("down", "down")
            self.assertIs(app.focused, detail)
            self.assertEqual(panel._selected, 0)
            self.assertGreater(detail.scroll_y, 0)
            await pilot.press("ctrl+l", "escape")
            self.assertIsNone(await task)

    async def test_no_focus_is_visible_and_recoverable(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            app.screen.set_focus(None)
            await pilot.pause()
            self.assertIn("No keyboard focus", str(app.query_one("#statusbar", Static).render()))
            await pilot.press("ctrl+l")
            self.assertIs(app.focused, app.pane.query_one(PromptInput))

    async def test_hidden_widget_cannot_receive_automatic_focus(self):
        app = self.make_app()
        async with app.run_test() as pilot:
            first = app.pane
            await pilot.press("ctrl+t")
            focused = app.focused
            app.focus_pane_widget(first, first.query_one(PromptInput))
            self.assertIs(app.focused, focused)
            focused.display = False
            app.screen.set_focus(None)
            app.focus_pane_widget(app.pane, focused)
            self.assertIsNone(app.focused)
