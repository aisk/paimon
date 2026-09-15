import asyncio
import contextlib
import tempfile
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from textual.widgets import Input, Static
from textual.worker import WorkerCancelled

from paimon.agent import replay_events
from paimon.app import PaimonApp
from paimon.jobs import Outcome
from paimon.pane import _EventRenderer
from paimon.session import (
    is_shell_message,
    shell_message,
    shell_text,
)
from paimon.ui import (
    AssistantMessage,
    ConfirmPanel,
    PromptInput,
    QuestionPanel,
    ToolCall,
    ToolResult,
    UserMessage,
)
from tests.support.agent import stub_model
from tests.support.app import AppTestCase, end_turn, hold_turn


class ConfirmPanelTest(AppTestCase):

    async def test_enter_allows_and_restores_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            task = await self._open_confirm(app, pilot)
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
            task = await self._open_confirm(app, pilot, args={"command": "rm -rf x"})
            await pilot.press("escape")
            self.assertFalse(await task)

    async def test_number_shortcut_denies(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open_confirm(app, pilot, "write_file", {"path": "c.py", "content": "x"})
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
            task = await self._open_confirm(app, pilot, args={"command": "new"})
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
            task = await self._open_confirm(app, pilot, args={"command": command})
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
                task = await self._open_confirm(app, pilot, "write_file",
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
            task = await self._open_confirm(app, pilot, "start_new_session", {"prompt": "carry on"})
            detail = str(app.query_one("#confirm-detail Static", Static).render())
            self.assertIn("carry on", detail)
            self.assertIn("fresh one", detail)
            await pilot.press("escape")
            self.assertFalse(await task)


class QuestionPanelTest(AppTestCase):
    """ask_user in the TUI: the panel replaces the prompt until answered."""

    @staticmethod
    async def _open(app: PaimonApp, pilot, question: str = "Which db?",
                    options: list[str] | None = None) -> asyncio.Future:
        task = asyncio.ensure_future(app.pane._ask(question, options or []))
        await pilot.pause()
        return task

    async def test_digit_picks_an_option_and_restores_prompt(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            task = await self._open(app, pilot, options=["Postgres", "SQLite"])
            panel = app.query_one(QuestionPanel)
            self.assertFalse(prompt.display, "prompt hidden while asking")
            self.assertIs(app.focused, panel)
            self.assertTrue(app.pane.needs_confirm, "the tab shows the pane is blocked")
            await pilot.press("2")
            self.assertEqual(await task, "SQLite")
            await pilot.pause()
            self.assertFalse(app.query(QuestionPanel))
            self.assertTrue(prompt.display)
            self.assertFalse(app.pane.needs_confirm)

    async def test_enter_picks_the_highlighted_option(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, options=["Postgres", "SQLite"])
            await pilot.press("down", "enter")
            self.assertEqual(await task, "SQLite")

    async def test_last_entry_takes_a_typed_answer(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, options=["Postgres", "SQLite"])
            await pilot.press("3")
            self.assertIsInstance(app.focused, Input, "the answer box has the keyboard")
            await pilot.press("m", "y", "s", "q", "l", "enter")
            self.assertEqual(await task, "mysql")

    async def test_without_options_typing_starts_at_once(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot)
            self.assertIsInstance(app.focused, Input)
            await pilot.press("4", "2", "enter")
            self.assertEqual(await task, "42", "digits are typed, not treated as choices")

    async def test_empty_typed_answer_is_ignored(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot)
            await pilot.press("enter")
            await pilot.pause()
            self.assertFalse(task.done(), "still waiting for an answer")
            await pilot.press("escape")
            self.assertIsNone(await task)

    async def test_escape_dismisses(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            task = await self._open(app, pilot, options=["Postgres"])
            await pilot.press("escape")
            self.assertIsNone(await task)
            await pilot.pause()
            self.assertTrue(app.query_one(PromptInput).display)

    async def test_a_turn_asking_gets_the_answer_back(self) -> None:
        app = self.make_app(mode="yolo")
        arguments = '{"question": "Which db?", "options": ["Postgres", "SQLite"]}'
        with patch("paimon.agent.build_model", return_value=stub_model("ask_user", arguments)):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.query(QuestionPanel))
                await pilot.press("1")
                await self._wait_for(pilot, lambda: not app.pane.is_busy)

                results = " ".join(str(w.render()) for w in app.query(ToolResult))
                self.assertIn("User answered: Postgres", results)
                self.assertTrue(app.query(AssistantMessage), "turn continued after the answer")


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
                await self._wait_for(pilot, lambda: not app.pane.is_busy)

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

    async def test_escape_stops_the_turn_and_the_pane_still_works(self) -> None:
        app = self.make_app()
        with patch("paimon.agent.build_model", return_value=self._endless_model()):
            async with app.run_test() as pilot:
                app.pane.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(pilot, lambda: app.pane.is_busy)

                await pilot.press("escape")
                await self._wait_for(pilot, lambda: not app.pane.is_busy)
                self.assertIs(app.pane.job.result.outcome, Outcome.INTERRUPTED)
                self.assertIn("Paimon stopped", self._log_text(app.pane))

                # The driver survived, so the pane takes the next prompt.
                app.pane.handle_submit(PromptInput.Submitted("again"))
                await self._wait_for(pilot, lambda: app.pane.is_busy)
                self.assertIn("again", self._log_text(app.pane),
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
            self.assertIn("provider failed", self._log_text(app.pane),
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


class UserCommandTest(AppTestCase):
    """The "!" prefix: a command the user runs themselves, beside the turn."""

    @staticmethod
    async def _run(app, pilot, text: str) -> None:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        prompt.load_text(text)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

    async def test_command_is_logged_and_recorded(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._run(app, pilot, "!echo hi")

            self.assertEqual(app.query_one(PromptInput).text, "")
            call = app.query(ToolCall).first()
            self.assertIn("echo hi", str(call.render()))
            self.assertIn("hi", str(app.query(ToolResult).first().render()))

            message = app.pane.agent.history[-1]
            self.assertTrue(is_shell_message(message))
            command, output = shell_text(message)
            self.assertEqual(command, "echo hi")
            self.assertIn("hi", output)
            # Not the session's title: paimon wrote it, not the user.
            self.assertIsNone(app.pane.agent.session.first_user_text())

    async def test_double_bang_runs_without_telling_the_model(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._run(app, pilot, "!!echo hi")
            self.assertIn("hi", str(app.query(ToolResult).first().render()))
            self.assertEqual(app.pane.agent.history, [])

    async def test_a_bare_bang_runs_nothing(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            await self._run(app, pilot, "!   ")
            self.assertEqual(len(app.query(ToolCall)), 0)
            self.assertEqual(app.pane.agent.history, [])

    async def test_running_during_a_turn_waits_for_the_gap_between_steps(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            await self._run(app, pilot, "!echo hi")

            # The turn owns the history until it comes back for the queue.
            self.assertEqual(app.pane.agent.history, [])
            self.assertIn("hi", str(app.query(ToolResult).first().render()))

            queued = app.pane._take_queued()
            self.assertTrue(is_shell_message(queued[0]))

    async def test_a_run_the_turn_never_came_back_for_is_flushed(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            await self._run(app, pilot, "!echo hi")
            await end_turn(app.pane)
            self.assertTrue(is_shell_message(app.pane.agent.history[-1]))

    async def test_queued_runs_come_before_queued_prompts(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            await self._run(app, pilot, "!echo hi")
            await self._run(app, pilot, "and now this")

            queued = app.pane._take_queued()
            self.assertTrue(is_shell_message(queued[0]))
            self.assertEqual(queued[1], "and now this")

    async def test_a_second_command_is_refused_while_one_runs(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.pane._shell_worker = object()
            app.pane.run_user_command("echo hi")
            await pilot.pause()
            self.assertEqual(len(app.query(ToolCall)), 0)

    async def test_escape_stops_the_command_before_the_turn(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            hold_turn(app.pane)
            interrupted = []
            app.pane.job.interrupt = lambda: interrupted.append(True)

            app.pane.run_user_command("sleep 30")
            await pilot.pause()
            app.pane.interrupt()
            with contextlib.suppress(WorkerCancelled):
                await app.workers.wait_for_complete()
            await pilot.pause()

            self.assertEqual(interrupted, [])  # the turn is still running
            self.assertIn("interrupted", str(app.query(ToolResult).first().render()))
            self.assertEqual(app.pane._pending_shell, [])
            # A second Esc, with no command left to stop, reaches the turn.
            app.pane.interrupt()
            self.assertEqual(interrupted, [True])

    async def test_replayed_history_shows_the_run_again(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            renderer = _EventRenderer(app.pane)
            for event in replay_events([shell_message("echo hi", "hi")]):
                await renderer.handle(event)
            await pilot.pause()
            self.assertIn("echo hi", str(app.query(ToolCall).first().render()))
            self.assertIn("hi", str(app.query(ToolResult).first().render()))

    async def test_the_border_marks_a_line_that_will_run(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            await pilot.press("!")
            self.assertIn("bash", prompt.classes)
            await pilot.press("backspace")
            self.assertNotIn("bash", prompt.classes)
