import argparse
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paimon import cli, herdr
from paimon.agent import Agent
from paimon.app import PaimonApp
from paimon.config import Config
from paimon.herdr import Report, Reporter

from .support.app import AppTestCase

_ENV = {"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1", "HERDR_BIN_PATH": "/usr/bin/herdr",
        "HERDR_SOCKET_PATH": "/tmp/herdr.sock"}


class FromEnvTest(unittest.TestCase):
    def test_inside_herdr(self) -> None:
        self.assertIsNotNone(Reporter.from_env(_ENV))

    def test_outside_herdr(self) -> None:
        self.assertIsNone(Reporter.from_env({}))

    def test_every_variable_is_required(self) -> None:
        for name in _ENV:
            with self.subTest(missing=name):
                self.assertIsNone(Reporter.from_env({**_ENV, name: ""}))


def _on_path(found: bool = True):
    return patch("paimon.herdr.shutil.which", return_value="/usr/bin/paimon" if found else None)


class ResumableTest(unittest.TestCase):
    def test_plain_command(self) -> None:
        with _on_path():
            self.assertTrue(herdr.resumable(("paimon", "--resume", "abc")))

    def test_rejected_commands(self) -> None:
        for argv in ((), ("paimon", "it's"), ("paimon", "a\nb"), ("paimon", "a\x85b"),
                     ("paimon",) * 65, ("paimon", "x" * 9000)):
            with self.subTest(argv=argv[:2]), _on_path():
                self.assertFalse(herdr.resumable(argv))

    def test_command_herdr_could_not_find(self) -> None:
        with _on_path(False):
            self.assertFalse(herdr.resumable(("paimon", "--resume", "abc")))


class ReporterTest(unittest.TestCase):
    """Against a stand-in binary that logs one line per call."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "calls"
        self.script = Path(tmp.name) / "herdr.py"
        self._write_script()
        # A launcher rather than the script itself, which Windows cannot run.
        if sys.platform == "win32":
            binary = Path(tmp.name) / "herdr.cmd"
            binary.write_text(f'@"{sys.executable}" "{self.script}" %*\n', encoding="utf-8")
        else:
            binary = Path(tmp.name) / "herdr"
            binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{self.script}" "$@"\n',
                              encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        self.reporter = Reporter(str(binary), "w1:p1")
        on_path = _on_path()
        on_path.start()
        self.addCleanup(on_path.stop)

    def _write_script(self, slow: str = "") -> None:
        """Log the arguments of each call, after a long pause on the ``slow`` command."""
        self.script.write_text(
            "import sys, time\n"
            f"if sys.argv[2] == {slow!r}:\n"
            "    time.sleep(5)\n"
            f"with open({str(self.log)!r}, 'a', encoding='utf-8') as log:\n"
            "    log.write(' '.join(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8")

    def _calls(self, count: int) -> list[list[str]]:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
            if len(lines) >= count:
                return [line.split() for line in lines]
            time.sleep(0.01)
        self.fail(f"expected {count} calls to herdr")

    def test_state_report(self) -> None:
        self.reporter.report(Report(herdr.WORKING))
        (call,) = self._calls(1)
        self.assertEqual(call[:8], ["pane", "report-agent", "w1:p1", "--source", "paimon",
                                    "--agent", "paimon", "--seq"])
        self.assertEqual(call[9:], ["--state", "working"])

    def test_resume_command_follows_a_separator(self) -> None:
        self.reporter.report(Report(herdr.IDLE, "abc", ("paimon", "--resume", "abc")))
        (call,) = self._calls(1)
        self.assertEqual(call[9:], ["--state", "idle", "--agent-session-id", "abc",
                                    "--", "paimon", "--resume", "abc"])

    def test_unacceptable_resume_command_is_left_out(self) -> None:
        self.reporter.report(Report(herdr.IDLE, "abc", ("paimon", "--skill", "it's")))
        (call,) = self._calls(1)
        self.assertEqual(call[9:], ["--state", "idle", "--agent-session-id", "abc"])

    def test_repeated_report_is_sent_once(self) -> None:
        self.reporter.report(Report(herdr.IDLE))
        self._calls(1)
        self.reporter.report(Report(herdr.IDLE))
        self.reporter.report(Report(herdr.WORKING))
        self.assertEqual([call[-1] for call in self._calls(2)], ["idle", "working"])

    def test_reports_piling_up_collapse_into_the_latest(self) -> None:
        with self.reporter._cond:  # hold the sender off while reports pile up
            for state in (herdr.WORKING, herdr.BLOCKED, herdr.IDLE):
                self.reporter.report(Report(state))
            self.assertEqual(self.reporter._pending, Report(herdr.IDLE))
        self.assertEqual(self._calls(1)[0][-1], "idle")

    def test_release_does_not_wait_for_a_report_in_flight(self) -> None:
        self._write_script(slow="report-agent")
        self.reporter.report(Report(herdr.WORKING))
        time.sleep(0.2)
        started = time.monotonic()
        self.reporter.release()
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(self._calls(1)[0][1], "release-agent")

    def test_sequence_numbers_increase(self) -> None:
        self.reporter.report(Report(herdr.WORKING))
        self._calls(1)
        self.reporter.report(Report(herdr.IDLE))
        self._calls(2)
        self.reporter.release()
        seqs = [int(call[8]) for call in self._calls(3)]
        self.assertEqual(seqs, sorted(set(seqs)))

    def test_release_gives_the_pane_back_and_ends_reporting(self) -> None:
        self.reporter.report(Report(herdr.WORKING))
        self._calls(1)
        self.reporter.release()
        self.reporter.report(Report(herdr.IDLE))
        self.reporter.release()
        calls = self._calls(2)
        self.assertEqual([call[1] for call in calls], ["report-agent", "release-agent"])
        self.assertNotIn("--state", calls[1])

    def test_missing_binary_is_ignored(self) -> None:
        reporter = Reporter("/nonexistent/herdr", "w1:p1")
        reporter.report(Report(herdr.WORKING))
        reporter.release()


class _Recorder:
    def __init__(self) -> None:
        self.reports: list[Report] = []
        self.released = 0

    def report(self, report: Report) -> None:
        if not self.reports or self.reports[-1] != report:
            self.reports.append(report)

    def release(self) -> None:
        self.released += 1


class ResumeFlagsTest(unittest.TestCase):
    @staticmethod
    def _flags(**given) -> tuple[str, ...]:
        args = {"strict": False, "no_web_search": False, "no_skills": False, "skills": [],
                "model": "zai:glm-4.7", "mode": "auto"}
        return cli._resume_flags(argparse.Namespace(**{**args, **given}))

    def test_nothing_by_default(self) -> None:
        # Model and mode are the app's to add: they change under it.
        self.assertEqual(self._flags(), ())

    def test_options_fixed_at_launch(self) -> None:
        self.assertEqual(
            self._flags(strict=True, no_web_search=True, no_skills=True, skills=["a", "b"]),
            ("--strict", "--no-web-search", "--no-skills", "--skill", "a", "--skill", "b"))


class AppReportingTest(AppTestCase):
    def _app(self, recorder: _Recorder, **kwargs) -> PaimonApp:
        session = kwargs.pop("session", None)
        agent = Agent.open(session=session, mode="read", config=Config(model="test-model"))
        return PaimonApp(agent, resumed=session is not None, reporter=recorder, **kwargs)

    async def test_untouched_session_is_idle_with_nothing_to_resume(self) -> None:
        recorder = _Recorder()
        async with self._app(recorder).run_test() as pilot:
            await pilot.pause()
            self.assertEqual(recorder.reports, [Report(herdr.IDLE)])
        self.assertEqual(recorder.released, 1)

    async def test_session_with_history_reports_its_resume_command(self) -> None:
        recorder = _Recorder()
        session = self._old_session()
        app = self._app(recorder, session=session, resume_flags=("--strict",))
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertEqual(recorder.reports[-1], Report(
                herdr.IDLE, session.id,
                ("paimon", "--resume", session.id, "--mode", "read",
                 "--model", "test-model", "--strict")))

    async def test_resume_command_follows_mode_and_model(self) -> None:
        recorder = _Recorder()
        app = self._app(recorder, session=self._old_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_cycle_mode()  # read -> auto
            await pilot.pause()
            self.assertIn("auto", recorder.reports[-1].resume)
            app.config.model = "other-model"
            app._report_herdr()
            resume = recorder.reports[-1].resume
            self.assertEqual(resume[resume.index("--model") + 1], "other-model")

    async def test_new_session_has_no_resume_command_until_it_has_a_turn(self) -> None:
        recorder = _Recorder()
        app = self._app(recorder, session=self._old_session())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_new_session()
            await pilot.pause()
            self.assertEqual(recorder.reports[-1], Report(herdr.IDLE))

    async def test_agent_running_in_the_background_is_work(self) -> None:
        recorder = _Recorder()
        app = self._app(recorder)
        async with app.run_test() as pilot:
            await pilot.pause()
            job = SimpleNamespace(kind="agent", agent=object(), running=True)
            app.panes[0].agent.jobs["a1"] = job
            app._sync_panes()
            self.assertEqual(recorder.reports[-1].state, herdr.WORKING)
            job.running = False
            app._sync_panes()
            self.assertEqual(recorder.reports[-1].state, herdr.IDLE)
            del app.panes[0].agent.jobs["a1"]

    async def test_background_command_is_not_work(self) -> None:
        recorder = _Recorder()
        app = self._app(recorder)
        async with app.run_test() as pilot:
            await pilot.pause()
            job = SimpleNamespace(kind="command", agent=None, running=True)
            app.panes[0].agent.jobs["c1"] = job
            app._sync_panes()
            self.assertEqual(recorder.reports[-1].state, herdr.IDLE)
            del app.panes[0].agent.jobs["c1"]

    async def test_confirmation_in_a_hidden_pane_blocks(self) -> None:
        recorder = _Recorder()
        app = self._app(recorder)
        async with app.run_test() as pilot:
            first = app.panes[0]
            await pilot.press("ctrl+t")
            await pilot.pause()
            task = await self._open_confirm(app, pilot, pane=first)
            self.assertEqual(recorder.reports[-1].state, herdr.BLOCKED)
            await pilot.press("ctrl+g")
            await self._wait_for_panel(app, pilot)
            await pilot.press("enter")
            await task
            await pilot.pause()
            self.assertEqual(recorder.reports[-1].state, herdr.IDLE)
