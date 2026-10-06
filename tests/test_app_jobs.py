import asyncio
import json
import re
from unittest.mock import AsyncMock, patch

from pydantic_ai.models.function import FunctionModel
from textual.widgets import RichLog, Static

from paimon import lockfile
from paimon.app import MAX_PANES, PaimonApp
from paimon.commandpane import CommandPane
from paimon.session import Session, is_job_message
from paimon.ui import (
    ConfirmPanel,
    PromptInput,
)
from tests.support.agent import spawning_model, stub_model
from paimon.turns import Outcome
from tests.support.app import AppTestCase, end_turn, hold_turn
from tests.support.shell import alive, pid_from, printer, spawner
from tests.support.turns import FakeCommand


class SpawnAgentTest(AppTestCase):
    """spawn_agent in the UI: work in the background, reported in this pane."""

    async def _spawn(self, app: PaimonApp, pilot) -> str:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: bool(app.pane.agent.jobs) or bool(
            [m for m in app.pane.agent.history if is_job_message(m)]))
        return next(iter(app.pane.agent.jobs), "")

    async def test_an_agent_opens_no_pane_and_shows_in_the_status_bar(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                self.assertEqual(len(app.panes), 1, "an agent is not a tab")
                self.assertIs(app.focused, parent.query_one(PromptInput))
                self.assertIn(job_id, self._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertIn("1 agent running",
                              str(app.query_one("#statusbar", Static).render()))
                gate.set()
                await self._wait_for(pilot, lambda: not parent.agent.jobs)
                await self._wait_for(pilot, lambda: "agent running" not in str(
                    app.query_one("#statusbar", Static).render()))

    async def test_a_finished_child_wakes_the_idle_parent(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate,
                                               answer="first line\nthe parser is fine")):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                # Nobody types: the answer alone starts the turn that reports it.
                gate.set()
                await self._wait_for(
                    pilot, lambda: f"agent {job_id} finished" in self._log_text(parent))
                await self._wait_for(pilot, lambda: not parent.is_busy)

                self.assertTrue(any(is_job_message(message)
                                    for message in parent.agent.history))
                self.assertEqual(parent.agent.notices, [])
                self.assertEqual(parent.agent.session.first_user_text(), "go",
                                 "the wake-up never becomes the session title")
                self.assertEqual(parent.tab_title, "go")

    async def test_a_notice_left_over_when_a_turn_ends_starts_the_next_one(self) -> None:
        # The loop only looks between steps, so a job that ended during the
        # last one is still waiting when the turn is over.
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=stub_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                hold_turn(parent)
                parent.agent.notices.append("agent a1f2 finished:\nfound it")
                parent._jobs_changed()
                await pilot.pause()
                self.assertEqual(len(parent.agent.notices), 1, "busy: nothing starts yet")

                await end_turn(parent)
                await self._wait_for(pilot, lambda: any(
                    is_job_message(message) for message in parent.agent.history))
                await self._wait_for(pilot, lambda: not parent.is_busy)
                self.assertEqual(parent.agent.notices, [])

    async def test_a_turn_the_user_stopped_does_not_restart_itself(self) -> None:
        app = self.make_app(mode="yolo")
        async with app.run_test() as pilot:
            parent = app.pane
            hold_turn(parent)
            parent.agent.notices.append("agent a1f2 finished:\nfound it")
            await end_turn(parent, Outcome.INTERRUPTED)
            await pilot.pause()
            self.assertFalse(parent.is_busy)
            self.assertEqual(len(parent.agent.notices), 1, "it waits for whatever runs next")

    async def test_the_parent_stays_conversational_while_a_child_runs(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                parent.handle_submit(PromptInput.Submitted("and what about the lexer?"))
                await self._wait_for(pilot, lambda: not parent.is_busy)
                self.assertTrue(parent.agent.jobs[job_id].running,
                                "a whole turn ran with the child still going")
                gate.set()
                await self._wait_for(pilot, lambda: not parent.agent.jobs)

    async def test_the_palette_stops_one_agent_and_the_model_hears_of_it(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["one", "two"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)
                first, second = parent.agent.jobs
                titles = [command.title for command in app.get_system_commands(app.screen)]
                self.assertIn(f"Stop agent {first}", titles)
                self.assertIn(f"Stop agent {second}", titles)

                app.action_stop_agent(first)
                await self._wait_for(
                    pilot, lambda: f"agent {first} was stopped by the user"
                    in self._log_text(parent))
                self.assertEqual(list(parent.agent.jobs), [second], "the other one carries on")
                gate.set()
                await self._wait_for(pilot, lambda: not parent.agent.jobs)

    async def test_changing_the_session_stops_its_agents(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)
                old = parent.agent
                (child,) = [session for session in Session.list(old.cwd, include_children=True)
                            if session.parent_id == old.session.id]
                self.assertTrue(lockfile.held(child.path))

                parent.new_session()
                await self._wait_for(pilot, lambda: not lockfile.held(child.path))

                log = self._log_text(parent)
                self.assertIn("Stopped 1 job", log)
                self.assertIn(job_id, log)
                self.assertEqual(parent.agent.jobs, {})
                self.assertEqual(old.notices, [], "the conversation left behind hears nothing")


class ChildConfirmationTest(AppTestCase):
    """A child's confirmation shows in its parent's pane, beside the prompt."""

    async def _ask(self, app: PaimonApp, pilot, job_id: str = "a1f2") -> asyncio.Future:
        task = asyncio.ensure_future(
            app.pane._confirm_child(job_id, "shell", {"command": "rm -rf build"}))
        await self._wait_for(pilot, lambda: bool(app.pane.query(ConfirmPanel)))
        return task

    async def test_it_does_not_take_the_keyboard_from_a_draft(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.pane.query_one(PromptInput)
            await pilot.press("h", "i")
            task = await self._ask(app, pilot)

            self.assertTrue(prompt.display, "the prompt stays where it was")
            self.assertIs(app.focused, prompt)
            await pilot.press("y")
            await pilot.pause()
            self.assertEqual(prompt.text, "hiy", "typing goes on into the draft")
            self.assertFalse(task.done(), "and answers nothing")
            self.assertTrue(app.pane.needs_confirm)
            self.assertIn("agent a1f2 needs permission",
                          str(app.pane.query_one(ConfirmPanel).children[0].render()))

            await pilot.press("ctrl+g")
            await pilot.pause()
            self.assertIs(app.focused, app.pane.query_one(ConfirmPanel))
            await pilot.press("y")
            self.assertTrue(await task)
            await pilot.pause()
            self.assertIs(app.focused, prompt, "the keyboard goes back to the draft")
            self.assertEqual(prompt.text, "hiy")
            self.assertFalse(app.pane.needs_confirm)

    async def test_swapping_the_session_under_a_panel_keeps_the_count_right(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            pane = app.pane
            task = await self._ask(app, pilot)
            old = pane.driver
            self.assertEqual(old.blocked, 1)

            pane.new_session()
            task.cancel()  # what stopping the child does to its confirmation
            await self._wait_for(pilot, lambda: not pane.query(ConfirmPanel))

            self.assertIsNot(pane.driver, old)
            self.assertEqual(pane.driver.blocked, 0, "the new session owes nothing")
            self.assertFalse(pane.needs_confirm)
            self.assertTrue(pane.query_one(PromptInput).display)

    async def test_two_children_asking_at_once_are_asked_in_turn(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            first = await self._ask(app, pilot, "aaaa")
            second = asyncio.ensure_future(
                app.pane._confirm_child("bbbb", "shell", {"command": "ls"}))
            await pilot.pause()
            self.assertEqual(len(app.pane.query(ConfirmPanel)), 1,
                             "a second panel would strand the first one's answer")

            await pilot.press("ctrl+g")
            await pilot.press("n")
            self.assertFalse(await first)
            await self._wait_for(pilot, lambda: "agent bbbb" in str(
                app.pane.query_one(ConfirmPanel).children[0].render()))
            self.assertFalse(second.done())
            await pilot.press("ctrl+g")
            await pilot.press("y")
            self.assertTrue(await second)

    async def test_the_parents_own_confirmation_waits_behind_a_childs(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            child = await self._ask(app, pilot)
            own = asyncio.ensure_future(app.pane._confirm("shell", {"command": "echo hi"}))
            await pilot.pause()
            self.assertEqual(len(app.pane.query(ConfirmPanel)), 1)
            self.assertFalse(child.done(), "the child's panel was not swept away")

            await pilot.press("ctrl+g")
            await pilot.press("y")
            self.assertTrue(await child)
            await self._wait_for(pilot, lambda: not app.pane.query_one(PromptInput).display)
            await self._wait_for(
                pilot, lambda: app.focused is app.pane.query_one(ConfirmPanel))
            await pilot.press("y")
            self.assertTrue(await own)


# What the command prints, as opposed to the command line itself, which also
# holds the word and is shown at the top of its tab.
PID = r"pid \d+"


class BackgroundTaskTest(AppTestCase):
    """run_background in the UI: a process with a tab and no keyboard."""

    # Prints its own pid and starts a child, so "the command is gone" is
    # checked on a process only a kill of the whole tree brings down.
    COMMAND = spawner(then_sleep=True, own_pid=True)

    def _model(self, command: str | None = None) -> FunctionModel:
        return stub_model("run_background", json.dumps(
            {"command": command or self.COMMAND, "description": "dev server"}))

    @staticmethod
    def _pid(pane: CommandPane) -> int:
        return pid_from(pane.command.output.since(0)[0].decode())

    _alive = staticmethod(alive)

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
                self.assertIn(f"{task.job_id} dev server",
                              self._tab_text(app, task))
                self.assertIn(task.job_id, self._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertTrue(task.is_running)

    async def test_its_output_reaches_the_agent_and_the_tab_it_opens(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)

                answer = await parent.agent._job_tool("read_job", {"job_id": task.job_id})
                self.assertRegex(answer, PID)
                self.assertIn("running", answer)
                self.assertNotRegex(self._command_log_text(task), PID,
                                    "a hidden tab writes nothing; RichLog would defer it all")

                app._switch_to(task)
                await self._wait_for(pilot, lambda: re.search(PID, self._command_log_text(task)))
                # Whitespace aside: the log wraps a command line this long.
                self.assertIn("".join(self.COMMAND.split()),
                              "".join(self._command_log_text(task).split()),
                              "the tab opens with the command it is running")

    @staticmethod
    def _command_log_text(pane: CommandPane) -> str:
        return "\n".join(strip.text for strip in pane.query_one("#log", RichLog).lines)

    async def test_the_status_bar_follows_the_task(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                app._switch_to(task)
                await pilot.pause()

                bar = str(app.query_one("#statusbar", Static).render())
                self.assertIn(f"command {task.job_id}", bar)
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
                   return_value=self._model(printer("built", exit_code=1))):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                await self._wait_for(pilot, lambda: not task.is_running)

                self.assertEqual(len(app.panes), 2, "the tab stays, so the output can be read")
                self.assertEqual(task.status_text, "exited (code 1)")
                app._switch_to(task)
                await self._wait_for(pilot, lambda: "built" in self._command_log_text(task))
                self.assertIn("exited (code 1)",
                              str(app.query_one("#statusbar", Static).render()))

    async def test_a_command_belongs_to_the_pane_that_started_it(self) -> None:
        # The user is looking at another conversation when the first one's
        # agent starts a command: ownership follows the caller, not the screen.
        app = self.make_app(mode="yolo")
        # A fake process: only ownership is under test, and its exit has to be
        # the test's to time.
        running = FakeCommand("npm run dev")
        with patch("paimon.agent.build_model", return_value=stub_model()), \
                patch("paimon.tools.start_background",
                      new=AsyncMock(return_value=running)):
            async with app.run_test() as pilot:
                first = app.pane
                await pilot.press("ctrl+t")
                await self._wait_for(pilot, lambda: len(app.panes) == 2)
                second = app.pane
                self.assertIsNot(second, first)

                answer = await first.agent._job_tool(
                    "run_background", {"command": "npm run dev", "description": "dev server"})
                self.assertIn("Started background command", answer)
                await self._wait_for(pilot, lambda: len(app.panes) == 3)
                task = app.panes[2]
                job_id = task.job_id
                self.assertIs(app.pane, second, "the user's pane stays on screen")
                self.assertIn(job_id, first.agent.jobs)
                self.assertNotIn(job_id, second.agent.jobs)

                running.output.append(b"listening on 3000\n")
                mine = await first.agent._job_tool("read_job", {"job_id": job_id})
                self.assertIn("listening on 3000", mine)
                theirs = await second.agent._job_tool("read_job", {"job_id": job_id})
                self.assertIn("no agent or background command", theirs)

                # The exit is news for the pane that started it, and only for
                # that one: the other conversation never heard of this command.
                running.exit(1)
                await self._wait_for(
                    pilot, lambda: any(is_job_message(message)
                                       for message in first.agent.history))
                await self._wait_for(pilot, lambda: not first.is_busy)
                self.assertIn(f"command {job_id} exited (code 1)", self._log_text(first))
                self.assertFalse(second.is_busy, "nothing woke the pane on screen")
                self.assertFalse(any(is_job_message(message)
                                     for message in second.agent.history))

    async def test_crlf_lines_are_shown_and_redrawn_lines_keep_their_last_state(self) -> None:
        app = self.make_app(mode="yolo")
        running = FakeCommand("build")
        with patch("paimon.agent.build_model", return_value=stub_model()), \
                patch("paimon.tools.start_background",
                      new=AsyncMock(return_value=running)):
            async with app.run_test() as pilot:
                await app.pane.agent._job_tool(
                    "run_background", {"command": "build", "description": "build"})
                await self._wait_for(pilot, lambda: len(app.panes) == 2)
                task = app.panes[1]
                # The way Windows programs end a line, then a progress bar
                # redrawn in place and ended the same way.
                running.output.append(b"compiling\r\n10%\r90%\r\ndone\n")

                app._switch_to(task)
                await self._wait_for(pilot, lambda: "done" in self._command_log_text(task))
                shown = self._command_log_text(task)
                self.assertIn("compiling", shown, "a CRLF line is a line, not a blank")
                self.assertIn("90%", shown)
                self.assertNotIn("10%", shown, "only the last redraw is on screen")

    async def test_a_denied_confirmation_starts_nothing(self) -> None:
        app = self.make_app(mode="auto")
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

    async def test_no_free_pane_refuses_and_kills_the_command(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                for _ in range(MAX_PANES - 1):
                    await app.action_new_pane()
                parent = app.panes[0]
                app._switch_to(parent)
                parent.handle_submit(PromptInput.Submitted("run the dev server"))
                await self._wait_for(
                    pilot, lambda: "panes are in use" in self._log_text(parent))
                self.assertEqual(len(app.panes), MAX_PANES)
                self.assertEqual(parent.agent.jobs, {})

    async def test_stopping_it_from_the_model_closes_the_tab(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)
                pid = self._pid(task)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                answer = await parent.agent._job_tool("stop_job", {"job_id": task.job_id})
                self.assertIn("Stopped", answer)
                await self._wait_for(pilot, lambda: len(app.panes) == 1)
                await self._wait_for(pilot, lambda: not self._alive(pid))
                self.assertEqual(parent.agent.notices, [])

    async def test_an_exit_wakes_the_parent_with_the_code(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model",
                   return_value=self._model(printer("built", exit_code=3))):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)
                await self._wait_for(
                    pilot, lambda: f"command {task.job_id} exited (code 3)"
                    in self._log_text(parent))
                await self._wait_for(pilot, lambda: not parent.is_busy)
                self.assertTrue(any(is_job_message(m) for m in parent.agent.history))
