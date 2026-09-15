import asyncio
from unittest.mock import patch

from textual.containers import VerticalScroll
from textual.widgets import Static

from paimon import lockfile
from paimon.agent import Agent
from paimon.app import MAX_PANES, PaimonApp
from paimon.pane import SessionPane
from paimon.tabs import PaneTab
from paimon.ui import (
    ConfirmPanel,
    PromptInput,
)
from tests.support.agent import stub_model
from tests.support.app import AppTestCase


class MultiPaneTest(AppTestCase):
    """Opening, switching and closing panes."""

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
                self.assertNotIn("waiting on you",
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
            self.assertIn("1 waiting on you",
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
            self.assertNotIn("waiting on you",
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
            mine = await self._open_confirm(app, pilot)

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
            task = await self._open_confirm(app, pilot, args={"command": "rm x"})
            await pilot.press("x")  # a key the panel does not handle
            self.assertIsInstance(app.focused, ConfirmPanel)
            self.assertEqual(app.query_one(PromptInput).text, "")
            await pilot.press("escape")
            self.assertFalse(await task)
