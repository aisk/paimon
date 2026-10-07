import asyncio
import json
import re
from unittest.mock import AsyncMock, patch

from pydantic_ai.models.function import FunctionModel
from textual.widgets import RichLog, Static

from paimon import lockfile
from paimon.agentpane import AgentPane
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
    """spawn_agent in the UI: work in a tab of its own, reported to its parent."""

    async def _spawn(self, app: PaimonApp, pilot) -> str:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: bool(app.pane.agent.jobs) or bool(
            [m for m in app.pane.agent.history if is_job_message(m)]))
        return next(iter(app.pane.agent.jobs), "")

    async def test_an_agent_runs_in_a_background_tab_that_goes_when_it_does(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                self.assertEqual(len(app.panes), 2)
                child = app.panes[1]
                self.assertIsInstance(child, AgentPane)
                self.assertIs(app.pane, parent, "starting an agent does not switch panes")
                self.assertFalse(child.display)
                self.assertIs(app.focused, parent.query_one(PromptInput),
                              "a pane the user did not open takes no keys")
                self.assertIn(f"{job_id} check the parser", self._tab_text(app, child))
                self.assertTrue(child.is_running)
                self.assertIn(job_id, self._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertIn("1 agent running",
                              str(app.query_one("#statusbar", Static).render()))
                await self._wait_for(
                    pilot, lambda: "check the parser" in self._log_text(child))

                app._switch_to(child)
                await pilot.pause()
                bar = str(app.query_one("#statusbar", Static).render())
                self.assertIn(f"agent {job_id}", bar)
                self.assertIn(f"session {child.job.agent.session.id[:8]}", bar)

                gate.set()
                await self._wait_for(pilot, lambda: len(app.panes) == 1)
                self.assertIs(app.pane, parent, "the screen falls back to a pane still there")
                await self._wait_for(
                    pilot, lambda: f"agent {job_id} finished" in self._log_text(parent))
                await self._wait_for(pilot, lambda: "agent running" not in str(
                    app.query_one("#statusbar", Static).render()))

    async def test_no_free_pane_refuses_the_agent(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"])):
            async with app.run_test() as pilot:
                for _ in range(MAX_PANES - 1):
                    await app.action_new_pane()
                parent = app.panes[0]
                app._switch_to(parent)
                parent.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(
                    pilot, lambda: "panes are in use" in self._log_text(parent))
                self.assertEqual(len(app.panes), MAX_PANES)
                self.assertEqual(parent.agent.jobs, {})
                for session in Session.list(parent.cwd, include_children=True):
                    if session.parent_id == parent.agent.session.id:
                        self.assertFalse(lockfile.held(session.path),
                                         "the child that never ran holds nothing")

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

    async def test_closing_its_tab_stops_one_agent_and_the_model_hears_of_it(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["one", "two"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)
                first, second = parent.agent.jobs
                self.assertEqual([pane.job_id for pane in app.panes[1:]], [first, second])

                app._switch_to(app.panes[1])
                await pilot.press("ctrl+w")
                await self._wait_for(
                    pilot, lambda: f"agent {first} was stopped by the user"
                    in self._log_text(parent))
                self.assertEqual(list(parent.agent.jobs), [second], "the other one carries on")
                self.assertEqual([pane.job_id for pane in app.panes[1:]], [second])
                gate.set()
                await self._wait_for(pilot, lambda: not parent.agent.jobs)

    async def test_stopping_it_from_the_model_closes_the_tab(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                job_id = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                answer = await parent.agent._job_tool("stop_job", {"job_id": job_id})
                self.assertIn("Stopped agent", answer)
                await self._wait_for(pilot, lambda: len(app.panes) == 1)
                self.assertEqual(parent.agent.notices, [])

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
                await self._wait_for(pilot, lambda: len(app.panes) == 1)


class ChildConfirmationTest(AppTestCase):
    """A child's confirmation shows in the child's own pane."""

    async def _child(self, app: PaimonApp, pilot) -> AgentPane:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: len(app.panes) == 2)
        await self._wait_for(pilot, lambda: not app.panes[0].is_busy)
        return app.panes[1]

    async def _ask(self, pilot, child: AgentPane, command: str = "rm -rf build") -> asyncio.Future:
        # Through the hook the child's agent was given, as its loop would.
        task = asyncio.ensure_future(child.job.agent.confirm("shell", {"command": command}))
        await self._wait_for(pilot, lambda: child.needs_confirm)
        return task

    async def test_it_waits_in_the_childs_tab_without_taking_the_keyboard(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                parent = app.pane
                prompt = parent.query_one(PromptInput)
                child = await self._child(app, pilot)
                await pilot.press("h", "i")
                task = await self._ask(pilot, child)

                self.assertFalse(parent.query(ConfirmPanel), "nothing lands in the parent")
                self.assertIs(app.focused, prompt)
                await pilot.press("y")
                await pilot.pause()
                self.assertEqual(prompt.text, "hiy", "typing goes on into the draft")
                self.assertFalse(task.done(), "and answers nothing")
                self.assertFalse(child.is_running, "it is waiting, not working")
                self.assertTrue(app.query_one(f"#tab-{child.id}").has_class("-attention"))
                self.assertIn("1 waiting on you",
                              str(app.query_one("#statusbar", Static).render()))

                await pilot.press("ctrl+g")
                await pilot.pause()
                self.assertIs(app.pane, child)
                self.assertIs(app.focused, child.query_one(ConfirmPanel))
                await pilot.press("y")
                self.assertTrue(await task)
                await pilot.pause()
                self.assertFalse(child.needs_confirm)
                self.assertTrue(child.is_running)
                gate.set()
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

    async def test_two_confirmations_at_once_are_asked_in_turn(self) -> None:
        app = self.make_app(mode="yolo")
        gate = asyncio.Event()
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=gate)):
            async with app.run_test() as pilot:
                child = await self._child(app, pilot)
                app._switch_to(child)
                first = await self._ask(pilot, child)
                second = asyncio.ensure_future(child._confirm("shell", {"command": "ls"}))
                await pilot.pause()
                self.assertEqual(len(child.query(ConfirmPanel)), 1,
                                 "a second panel would strand the first one's answer")

                await self._wait_for(
                    pilot, lambda: app.focused is child.query_one(ConfirmPanel))
                await pilot.press("n")
                self.assertFalse(await first)
                await self._wait_for(pilot, lambda: child.needs_confirm and len(
                    child.query(ConfirmPanel)) == 1 and app.focused in child.query(ConfirmPanel))
                self.assertFalse(second.done())
                await pilot.press("y")
                self.assertTrue(await second)
                gate.set()
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

    async def test_closing_the_tab_with_a_panel_up_stops_the_agent(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model",
                   return_value=spawning_model(["check the parser"], gate=asyncio.Event())):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._child(app, pilot)
                task = await self._ask(pilot, child)
                app._switch_to(child)

                await pilot.press("ctrl+w")
                await self._wait_for(pilot, lambda: len(app.panes) == 1)
                task.cancel()  # what stopping the child does to its confirmation
                await self._wait_for(pilot, lambda: not parent.agent.jobs)
                self.assertFalse(parent.needs_confirm)
                await self._wait_for(pilot, lambda: "waiting on you" not in str(
                    app.query_one("#statusbar", Static).render()))


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
