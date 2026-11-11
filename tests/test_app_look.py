"""The look of the app: its themes, the welcome sky and the parts that move."""

from textual.widgets import Static

from paimon.config import Config
from paimon.events import TextDelta, ToolEnd, ToolStart, UserInput
from paimon.sparkle import TAGLINE, Hero, Masthead, Shimmer, Twinkle
from paimon.theme import DEFAULT_THEME, gauge
from paimon.transcript import EventRenderer
from paimon.ui import AssistantHeading, AssistantMessage, PromptInput
from tests.support.app import AppTestCase, hold_turn

_FOLDED = Hero._SKY_ABOVE + Hero._BLOCK


class ThemeTest(AppTestCase):
    async def test_paimons_own_theme_is_the_default_and_is_not_written_down(self) -> None:
        app = self.make_app()
        async with app.run_test():
            self.assertEqual(app.theme, DEFAULT_THEME)
            self.assertIn("paimon-light", app.available_themes)
            await app.workers.wait_for_complete()
            self.assertIsNone(Config.load().theme, "only a theme the user picked is saved")

    async def test_a_configured_theme_wins(self) -> None:
        app = self.make_app(config=Config(model="test-model", theme="nord"))
        async with app.run_test():
            self.assertEqual(app.theme, "nord")

    async def test_every_theme_draws_the_moving_parts(self) -> None:
        # Their colours are blended from the theme's own, ANSI names included.
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            for theme in ("paimon-light", "ansi-dark", "textual-light"):
                app.theme = theme
                await pilot.pause()
                self.assertIn(TAGLINE, str(app.query_one(Hero).render()))
                self.assertIn("Paimon", str(app.query_one(Masthead).render()))


class WelcomeTest(AppTestCase):
    async def test_an_empty_conversation_opens_on_the_sky_and_the_first_words_fold_it(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            hero = app.query_one(Hero)
            log = app.query_one("#log")
            self.assertIn(TAGLINE, str(hero.render()))
            self.assertGreater(hero.size.height, _FOLDED, "alone, it takes the whole log")
            self.assertEqual(log.max_scroll_y, 0, "and no more than the log has")

            app.pane.transcript.add_user("hello")
            await pilot.pause()
            self.assertEqual(hero.size.height, _FOLDED)
            self.assertTrue(hero._resting, "a folded sky holds still")
            self.assertIs(log.children[0], hero, "it stays as the top of the log")

    async def test_a_narrow_terminal_gets_the_name_without_the_letters(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(40, 24)) as pilot:
            await pilot.pause()
            hero = app.query_one(Hero)
            self.assertEqual(hero.size.height, 3)
            self.assertNotIn("█", str(hero.render()))
            self.assertIn("Paimon", str(hero.render()))

    async def test_a_resumed_session_opens_on_its_history_instead(self) -> None:
        app = self.make_app(session=self._old_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertFalse(app.query(Hero))

    async def test_a_new_session_gets_a_welcome_of_its_own(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            app.pane.transcript.add_user("hello")
            app.pane.new_session()
            await pilot.pause()
            self.assertEqual(len(app.query(Hero)), 1)
            self.assertIn("Started new session", self._log_text(app.pane))


class HeadingTest(AppTestCase):
    async def test_the_name_heads_the_first_text_block_of_a_turn_only(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = EventRenderer(app.pane.transcript, app.pane.agent)
            await renderer.handle(UserInput("go"))
            # Streamed in pieces: the heading must still be there afterwards.
            await renderer.handle(TextDelta("one "))
            await renderer.handle(TextDelta("two"))
            await renderer.handle(ToolStart("c1", "shell", {"command": "ls"}))
            await renderer.handle(ToolEnd("c1", "shell", "ok"))
            await renderer.handle(TextDelta("three"))
            await renderer.close()
            await pilot.pause()
            self.assertEqual(len(app.query(AssistantHeading)), 1)
            self.assertEqual(len(app.query(AssistantMessage)), 2)
            self.assertIn("Paimon", str(app.query_one(AssistantHeading).render()))
            log = app.query_one("#log")
            heading = list(log.children).index(app.query_one(AssistantHeading))
            self.assertIs(log.children[heading + 1], app.query(AssistantMessage).first())


class MovingPartsTest(AppTestCase):
    async def test_the_spinner_and_its_label_only_tick_while_shown(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            label, star = app.query_one(Shimmer), app.query_one(Twinkle)
            await pilot.pause(0.3)
            self.assertEqual((label._frame, star._frame), (0, 0), "hidden while idle")

            app.pane._set_status(True, " Counting mora… 3s")
            await self._wait_for(pilot, lambda: label._frame > 0 and star._frame > 0)
            self.assertEqual(str(label.render()), " Counting mora… 3s")

            app.pane._set_status(False)
            await pilot.pause()
            frame = label._frame
            await pilot.pause(0.3)
            self.assertEqual(label._frame, frame)

    async def test_the_masthead_names_the_directory_and_moves_only_for_work(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await pilot.pause()
            masthead = app.query_one(Masthead)
            self.assertIn("Paimon", str(masthead.render()))
            await pilot.pause(0.3)
            self.assertEqual(masthead._frame, 0)

            hold_turn(app.pane)
            app._sync_panes()
            await self._wait_for(pilot, lambda: masthead._frame > 0)

    async def test_the_prompt_wears_its_mode(self) -> None:
        app = self.make_app(mode="read")
        async with app.run_test() as pilot:
            prompt = app.pane.query_one(PromptInput)
            self.assertTrue(prompt.has_class("-read"))
            await pilot.press("shift+tab")
            await pilot.pause()
            self.assertTrue(prompt.has_class("-auto"))
            self.assertFalse(prompt.has_class("-read"))


class GaugeTest(AppTestCase):
    def test_a_gauge_is_always_its_full_width(self) -> None:
        for fraction, filled in ((0, 0), (0.5, 4), (1, 8), (7, 8), (-1, 0)):
            done, rest = gauge(fraction)
            self.assertEqual((len(done), len(done) + len(rest)), (filled, 8))

    async def test_the_status_bar_draws_how_full_the_context_is(self) -> None:
        app = self.make_app(config=Config(model="test-model", compaction_context_window=100_000))
        async with app.run_test() as pilot:
            app.refresh_statusbar(50_000)
            await pilot.pause()
            bar = str(app.query_one("#statusbar", Static).render())
            self.assertIn("context ━━━━━━ 50.0k/100k (50%)", bar)
