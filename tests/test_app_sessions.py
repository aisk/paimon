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
from textual.widgets import Input, Static

from paimon import lockfile
from paimon.agent import Agent
from paimon.app import PaimonApp
from paimon.config import Config
from paimon.login import BrowserLoginScreen, LoginScreen, PickerScreen, PromptScreen
from paimon.pane import _session_label
from paimon.ui import (
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from tests.support.app import AppTestCase, hold_turn


class ResumeSessionTest(AppTestCase):

    def _log_text(self, app: PaimonApp) -> str:
        return " ".join(str(widget.render())
                         for widget in app.query_one("#log").walk_children())

    async def test_palette_resume_swaps_agent_and_renders_history(self) -> None:
        old = self._old_session()
        app = self.make_app()
        async with app.run_test() as pilot:
            app.action_cycle_mode()  # read -> auto, must survive the resume
            app.action_resume_session()
            await pilot.pause()
            self.assertIsInstance(app.screen, PickerScreen)
            app.screen.dismiss(_session_label(old))
            await pilot.pause()
            self.assertEqual(app.pane.agent.session.id, old.id)
            self.assertEqual(app.pane.agent.mode, "auto")
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


class LoginTest(AppTestCase):
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

    async def test_chatgpt_login_asks_for_no_key_and_then_for_a_model(self) -> None:
        pasted: list[str] = []

        async def sign_in(show_url, redirect) -> None:
            show_url("https://auth.example/authorize")
            pasted.append(await redirect)

        app = self.make_app()
        with patch("paimon.chatgpt.login", sign_in), patch("paimon.chatgpt.open_browser") as opened:
            async with app.run_test() as pilot:
                app.action_login()
                await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
                app.screen.dismiss("chatgpt")
                await self._wait_for(pilot, lambda: isinstance(app.screen, BrowserLoginScreen))
                await self._wait_for(pilot, lambda: opened.called)
                opened.assert_called_once_with("https://auth.example/authorize")

                status = app.screen.query_one("#browser-login-status").render()
                self.assertIn("https://auth.example/authorize", {span.style.link for span in status.spans})
                await pilot.click("#browser-login-copy")
                self.assertEqual(app.clipboard, "https://auth.example/authorize")

                app.screen.query_one(Input).value = "http://127.0.0.1:1455/auth/callback?code=x"
                await pilot.press("enter")
                await self._wait_for(pilot, lambda: "Select model" in getattr(app.screen, "_title", ""))
                self.assertEqual(pasted, ["http://127.0.0.1:1455/auth/callback?code=x"])
                self.assertEqual(app.config.model, "test-model", "logging in alone changes no model")
                app.screen.dismiss("chatgpt:gpt-5.5")
                await self._wait_for(pilot, lambda: app.config.model == "chatgpt:gpt-5.5")
                self.assertEqual(app.config.providers, {})

    async def test_antigravity_login_shows_that_it_is_unsupported(self) -> None:
        async def sign_in(show_url, redirect) -> None:
            await redirect

        app = self.make_app()
        with patch("paimon.antigravity.login", sign_in):
            async with app.run_test() as pilot:
                app.action_login()
                await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
                app.screen.dismiss("antigravity")
                await self._wait_for(pilot, lambda: isinstance(app.screen, BrowserLoginScreen))
                notice = app.screen.query_one("#browser-login-notice", Static)
                self.assertIn("at your own risk", str(notice.render()))
                await pilot.press("escape")
                await self._wait_for(pilot, lambda: self._login_screens(app) == [])

    async def test_a_cancelled_chatgpt_login_keeps_the_model(self) -> None:
        async def sign_in(show_url, redirect) -> None:
            await redirect

        app = self.make_app()
        with patch("paimon.chatgpt.login", sign_in):
            async with app.run_test() as pilot:
                app.action_login()
                await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
                app.screen.dismiss("chatgpt")
                await self._wait_for(pilot, lambda: isinstance(app.screen, BrowserLoginScreen))
                await pilot.press("escape")
                await self._wait_for(pilot, lambda: self._login_screens(app) == [])
                self.assertNotIsInstance(app.screen, PickerScreen)
                self.assertEqual(app.config.model, "test-model")

    async def _log_in_with_key(self, app: PaimonApp, pilot, provider: str, key: str) -> None:
        """Drive the login up to the model picker that follows it."""
        app.action_login()
        await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
        app.screen.dismiss(provider)
        for answer in ("", key):
            await self._wait_for(pilot, lambda: isinstance(app.screen, PromptScreen)
                                 and app.screen.is_current)
            screen = app.screen
            screen.dismiss(answer)
            await self._wait_for(pilot, lambda: app.screen is not screen)
        await self._wait_for(pilot, lambda: "Select model" in getattr(app.screen, "_title", ""))

    async def test_a_key_login_stores_the_credential_and_leaves_the_model_to_the_picker(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._log_in_with_key(app, pilot, "zai", "sk-zai")
            self.assertEqual(Config.load().providers, {"zai": {"api_key": "sk-zai"}})
            self.assertTrue(all(option.startswith("zai:") for option in app.screen._options))
            app.screen.dismiss(None)
            await self._wait_for(pilot, lambda: not isinstance(app.screen, PickerScreen))
            self.assertEqual(app.config.model, "test-model", "escaping the picker keeps the model")
            self.assertIsNone(Config.load().model)

    async def test_a_model_typed_after_login_belongs_to_that_provider(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._log_in_with_key(app, pilot, "zai", "sk-zai")
            app.screen.dismiss("glm-9")
            await self._wait_for(pilot, lambda: Config.load().model == "zai:glm-9")

    async def test_a_blank_key_keeps_the_stored_one(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._log_in_with_key(app, pilot, "zai", "sk-zai")
            app.screen.dismiss(None)
            await self._wait_for(pilot, lambda: not isinstance(app.screen, PickerScreen))
            await self._log_in_with_key(app, pilot, "zai", "")
            self.assertEqual(Config.load().providers, {"zai": {"api_key": "sk-zai"}})

    async def test_first_launch_with_a_credential_only_asks_for_the_model(self) -> None:
        app = self.make_app(config=Config(providers={"zai": {"api_key": "k"}}))
        async with app.run_test() as pilot:
            await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            self.assertEqual(app.screen._title, "Switch model")
            self.assertEqual(self._login_screens(app), [])

    @staticmethod
    def _login_screens(app: PaimonApp) -> list[LoginScreen]:
        return [screen for screen in app.screen_stack if isinstance(screen, LoginScreen)]


class ModelSwitchTest(AppTestCase):
    CATALOG = ["zai:glm-5.3", "zai:glm-5.3-flash", "openai:gpt-6.1-sol"]

    def _app(self) -> PaimonApp:
        return self.make_app(config=Config(model="zai:glm-5.3", providers={"zai": {"api_key": "k"}}))

    async def test_switching_saves_the_model_for_every_pane(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            first = app.pane
            await pilot.press("ctrl+t")
            await pilot.pause()
            with patch("paimon.llm.known_models", return_value=self.CATALOG):
                app.action_switch_model()
                await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            self.assertEqual(app.screen._options, ["zai:glm-5.3", "zai:glm-5.3-flash"],
                             "only providers there is a credential for")
            app.screen.dismiss("zai:glm-5.3-flash")
            await self._wait_for(pilot, lambda: Config.load().model == "zai:glm-5.3-flash")
            self.assertEqual(app.pane.agent.model_name, "zai:glm-5.3-flash")
            self.assertEqual(first.agent.model_name, "zai:glm-5.3-flash")
            self.assertEqual(Config.load().providers, {}, "credentials are not touched")

    async def test_a_model_without_a_provider_is_asked_for_again(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            app.action_switch_model()
            await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen))
            first = app.screen
            first.dismiss("glm-5.3-flash")
            await self._wait_for(pilot, lambda: isinstance(app.screen, PickerScreen)
                                 and app.screen is not first)
            self.assertEqual(app.config.model, "zai:glm-5.3")
            self.assertIsNone(Config.load().model)

    async def test_switching_is_refused_while_a_turn_is_running(self) -> None:
        app = self._app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            app.action_switch_model()
            await pilot.pause()
            self.assertNotIsInstance(app.screen, PickerScreen)


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
