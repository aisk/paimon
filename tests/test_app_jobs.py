import asyncio
import json
import os
import re
from unittest.mock import patch

from pydantic_ai.models.function import FunctionModel
from textual.widgets import RichLog, Static

from paimon import agents, lockfile, tools
from paimon.app import PaimonApp
from paimon.commandpane import CommandPane
from paimon.pane import SessionPane
from paimon.session import (
    Session,
    is_agents_message,
)
from paimon.ui import (
    ConfirmPanel,
    PromptInput,
)
from tests.support.agent import stub_model
from tests.support.app import AppTestCase


class SpawnAgentTest(AppTestCase):
    """spawn_agent in the UI: a second pane nobody asked to look at."""

    @staticmethod
    def _spawning_model() -> FunctionModel:
        return stub_model("spawn_agent", '{"prompt": "check the parser"}')

    async def _spawn(self, app: PaimonApp, pilot) -> SessionPane:
        app.pane.handle_submit(PromptInput.Submitted("go"))
        await self._wait_for(pilot, lambda: len(app.panes) == 2)
        return app.panes[1]

    async def test_the_new_pane_stays_in_the_background(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                prompt = parent.query_one(PromptInput)
                child = await self._spawn(app, pilot)

                self.assertIs(app.pane, parent, "spawning does not switch panes")
                self.assertFalse(child.display)
                self.assertIs(app.focused, prompt, "a pane the user did not open takes no keys")
                self.assertEqual(child.agent.cwd, parent.agent.cwd)
                self.assertEqual(child.mode, parent.mode)
                self.assertIn(child.job.job_id, self._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertIn(f"{child.job.job_id} check the parser",
                              self._tab_text(app, child))

    async def test_the_new_agent_cannot_spawn_or_hand_off(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                child = await self._spawn(app, pilot)
                self.assertNotIn("spawn_agent", child.agent.toolset, "depth stays 1")
                self.assertNotIn("start_new_session", child.agent.toolset,
                                 "a handoff would swap the session out from under its id")
                self.assertNotIn("ask_user", child.agent.toolset,
                                 "a subagent reports to its parent, not to the user")
                self.assertNotIn("run_background", child.agent.toolset,
                                 "only the conversation the user is in leaves processes behind")
                for name in ("read_job", "wait_for_job", "stop_job", "send_to_agent"):
                    self.assertNotIn(name, child.agent.toolset,
                                     "it can start nothing, so it has nothing to look at")

    async def test_the_new_session_is_a_child_and_stays_out_of_the_listings(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)

                self.assertEqual(child.agent.session.parent_id, parent.agent.session.id)
                listed = [session.id for session in Session.list(parent.agent.cwd)]
                self.assertNotIn(child.agent.session.id, listed)
                self.assertIn(child.agent.session.id,
                              [s.id for s in Session.list(parent.agent.cwd, include_children=True)])

    async def test_a_typed_spawn_narrows_tools_and_appends_the_prompt(self) -> None:
        app = self.make_app(mode="yolo")
        model = stub_model("spawn_agent", '{"prompt": "map the modules", "agent": "explore"}')
        with patch("paimon.agent.build_model", return_value=model):
            async with app.run_test() as pilot:
                app.pane.agent.model_override = "test:override"
                child = await self._spawn(app, pilot)

                expected = {name for name, tool in tools.REGISTRY.items()
                            if tool.access in ("read", "none")
                            and name not in tools.SUBAGENT_DENIED}
                self.assertEqual(set(child.agent.toolset), expected)
                self.assertTrue(child.agent.system_prompt.rstrip().endswith(
                    agents.builtin_types()[0].body),
                    "the type's body ends the child's system prompt")
                self.assertEqual(child.agent.model_override, "test:override",
                                 "with no explicit model the caller's override carries over")

    async def test_a_finished_child_wakes_the_parent(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)

                # The child's stub turn ends on its own; the parent is then
                # woken without anybody typing, reports the news and reacts.
                await self._wait_for(
                    pilot, lambda: f"Agents: {child.job.job_id} finished"
                    in self._log_text(parent))
                await self._wait_for(pilot, lambda: not parent.is_busy)
                self.assertTrue(any(is_agents_message(message)
                                    for message in parent.agent.history))
                self.assertNotIn(f"{child.job.job_id} finished",
                                 parent.agent.session.first_user_text() or "",
                                 "the wake-up never becomes the session title")

    async def test_a_stopped_agent_can_be_resumed_by_session_id(self) -> None:
        app = self.make_app(mode="yolo")
        model = stub_model("spawn_agent", '{"prompt": "map the modules", "agent": "explore"}')
        with patch("paimon.agent.build_model", return_value=model):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                session_id = child.agent.session.id
                await self._wait_for(pilot, lambda: not child.is_busy)
                kept = len(child.agent.history)
                self.assertGreater(kept, 0)

                answer = await app._supervisor.handle(
                    "stop_job", {"job_id": child.job.job_id}, caller=parent.agent)
                self.assertIn(session_id[:8], answer,
                              "the durable name rides the stop result")
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

                refused = await app._supervisor.handle(
                    "spawn_agent", {"prompt": "go", "session": parent.agent.session.id},
                    caller=parent.agent)
                self.assertIn("Error", refused, "only this conversation's children qualify")

                answer = await app._supervisor.handle(
                    "spawn_agent", {"prompt": "look further", "session": session_id[:8]},
                    caller=parent.agent)
                self.assertIn("Started agent", answer)
                await self._wait_for(pilot, lambda: len(app.panes) == 2)
                revived = app.panes[1]
                self.assertEqual(revived.agent.session.id, session_id)
                self.assertGreaterEqual(len(revived.agent.history), kept,
                                        "the conversation picks up where it ended")
                self.assertNotIn("write_file", revived.agent.toolset,
                                 "the recorded explore type narrows the tools again")
                self.assertIn("grep", revived.agent.toolset)

    async def test_an_unknown_type_reports_and_opens_no_pane(self) -> None:
        app = self.make_app(mode="yolo")
        model = stub_model("spawn_agent", '{"prompt": "go", "agent": "nope"}')
        with patch("paimon.agent.build_model", return_value=model):
            async with app.run_test() as pilot:
                parent = app.pane
                parent.handle_submit(PromptInput.Submitted("go"))
                await self._wait_for(
                    pilot, lambda: "unknown agent type" in self._log_text(parent))
                self.assertIn("'nope'", self._log_text(parent))
                self.assertEqual(len(app.panes), 1)

    async def test_changing_the_parents_session_stops_its_agents(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                agent_id, path = child.job.job_id, child.agent.session.path
                await self._wait_for(pilot, lambda: not parent.is_busy)

                parent.new_session()
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

                self.assertFalse(lockfile.held(path), "the stopped agent released its session")
                log = self._log_text(parent)
                self.assertIn("Stopped 1 agent", log)
                self.assertIn(agent_id, log)
                self.assertIs(app.pane, parent)

    async def test_closing_an_agents_pane_leaves_its_output_readable(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not child.is_busy)
                app._switch_to(child)

                await pilot.press("ctrl+w")
                await pilot.pause()

                answer = await app._supervisor.handle(
                    "read_job", {"job_id": child.job.job_id, "mode": "all"},
                    caller=parent.agent)
                self.assertIn("killed", answer)
                self.assertIn("done", answer, "what it managed to say survives its pane")

    async def test_closing_the_parent_of_the_only_other_pane_leaves_one_open(self) -> None:
        # Closing a pane stops the agents it started, so both panes can go at
        # once; the app has to be left with a conversation either way.
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy)

                await pilot.press("ctrl+w")
                await self._wait_for(pilot, lambda: len(app.panes) == 1)

                self.assertNotIn(app.pane, (parent, child))
                self.assertTrue(app.pane.display)
                self.assertIs(app.focused, app.pane.query_one(PromptInput))
                self.assertFalse(lockfile.held(child.agent.session.path))

    async def test_the_next_turn_opens_with_what_the_agents_did(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._spawning_model()):
            async with app.run_test() as pilot:
                parent = app.pane
                child = await self._spawn(app, pilot)
                await self._wait_for(pilot, lambda: not parent.is_busy and not child.is_busy)

                parent.handle_submit(PromptInput.Submitted("anything new?"))
                await self._wait_for(pilot, lambda: not parent.is_busy)

                self.assertIn(f"Agents: {child.job.job_id} finished",
                              self._log_text(parent))
                self.assertTrue(any(is_agents_message(message)
                                    for message in parent.agent.history),
                                "the model only learns of it through the history")


class BackgroundTaskTest(AppTestCase):
    """run_background in the UI: a process with a tab and no keyboard."""

    COMMAND = "printf 'pid %s\\n' $$; sleep 30"

    def _model(self, command: str | None = None) -> FunctionModel:
        return stub_model("run_background", json.dumps(
            {"command": command or self.COMMAND, "description": "dev server"}))

    @staticmethod
    def _pid(pane: CommandPane) -> int:
        match = re.search(r"pid (\d+)", pane.command.output.since(0)[0].decode())
        assert match, "the command never printed its pid"
        return int(match.group(1))

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

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
                self.assertIn(f"{task.job.job_id} dev server",
                              self._tab_text(app, task))
                self.assertIn(task.job.job_id, self._log_text(parent),
                              "the parent is told the id it has to use")
                self.assertTrue(task.is_running)

    async def test_its_output_reaches_the_agent_and_the_tab_it_opens(self) -> None:
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                task = await self._start(app, pilot)

                answer = await app._supervisor.handle(
                    "read_job", {"job_id": task.job.job_id}, caller=parent.agent)
                self.assertIn("pid", answer)
                self.assertIn("running", answer)
                self.assertNotIn("pid", self._command_log_text(task),
                                 "a hidden tab writes nothing; RichLog would defer it all")

                app._switch_to(task)
                await self._wait_for(pilot, lambda: "pid" in self._command_log_text(task))
                self.assertIn(self.COMMAND.split(";")[0].strip(), self._command_log_text(task),
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
                self.assertIn(f"command {task.job.job_id}", bar)
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
                   return_value=self._model("printf 'built\\n'; exit 1")):
            async with app.run_test() as pilot:
                task = await self._start(app, pilot)
                await self._wait_for(pilot, lambda: not task.is_running)

                self.assertEqual(len(app.panes), 2, "the tab stays, so the output can be read")
                self.assertEqual(task.status_text, "exited (code 1)")
                app._switch_to(task)
                await self._wait_for(pilot, lambda: "built" in self._command_log_text(task))
                self.assertIn("exited (code 1)",
                              str(app.query_one("#statusbar", Static).render()))

    async def test_an_agent_can_still_be_started_alongside_a_task(self) -> None:
        # Only a conversation has an agent, and finding the one that asked
        # walks the pane list, task panes included.
        app = self.make_app(mode="yolo")
        with patch("paimon.agent.build_model", return_value=self._model()):
            async with app.run_test() as pilot:
                parent = app.pane
                await self._start(app, pilot)

                answer = await app._supervisor.handle(
                    "spawn_agent", {"prompt": "check the parser"}, caller=parent.agent)
                await self._wait_for(pilot, lambda: len(app.panes) == 3)
                self.assertIn("Started agent", answer)
                self.assertEqual(app.panes[2].agent.cwd, parent.agent.cwd)

    async def test_a_denied_confirmation_starts_nothing(self) -> None:
        app = self.make_app(mode="read")
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
