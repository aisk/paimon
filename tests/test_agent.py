import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)

from paimon import compaction, lockfile, tools
from paimon.agent import (
    Agent,
)
from paimon.config import Config
from paimon.session import (
    Session,
    SessionIncompleteError,
)
from paimon.skills import default_skill_dirs as real_default_skill_dirs
from tests.support.agent import make_session, stub_model


class AgentSystemPromptTest(unittest.TestCase):
    def test_resume_rebuilds_the_prompt_and_records_changed_snapshots(self) -> None:
        """PROMPT-1: the dynamic prompt (date, environment, AGENTS.md) is
        rebuilt on resume; the log gains a snapshot only when it changed."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="snapshot") as generate,
            ):
                first = Agent.open(cwd=cwd, config=Config(model="test:stub"))

            self.assertEqual(first.system_prompt, "snapshot")
            self.assertEqual(session.system_prompt(), "snapshot")
            generate.assert_called_once_with(cwd, [])

            def snapshots() -> list[str]:
                return [json.loads(line)["content"]
                        for line in session.path.read_text(encoding="utf-8").splitlines()
                        if json.loads(line).get("type") == "system_prompt"]

            first.session.unlock()  # a session is only ever open once at a time
            with patch("paimon.agent.build_system_prompt", return_value="snapshot"):
                resumed = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            self.assertEqual(resumed.system_prompt, "snapshot")
            self.assertEqual(snapshots(), ["snapshot"], "an unchanged prompt appends nothing")

            resumed.session.unlock()
            with patch("paimon.agent.build_system_prompt", return_value="fresh snapshot"):
                fresh = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            self.assertEqual(fresh.system_prompt, "fresh snapshot")
            self.assertEqual(snapshots(), ["snapshot", "fresh snapshot"],
                             "the audit trail keeps every snapshot, latest wins")
            self.assertEqual(session.system_prompt(), "fresh snapshot")

    def test_resume_keeps_the_appended_role_on_a_rebuilt_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="base"),
            ):
                first = Agent.open(cwd=cwd, config=Config(model="test:stub"),
                                   append_system_prompt="You are a reviewer.")
            first.session.unlock()

            with patch("paimon.agent.build_system_prompt", return_value="new base"):
                resumed = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

            self.assertEqual(resumed.system_prompt, "new base\n\nYou are a reviewer.")

    def test_append_system_prompt_extends_a_new_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", return_value="base"),
            ):
                agent = Agent.open(cwd=cwd, config=Config(model="test:stub"),
                                   append_system_prompt="  You are a reviewer.  ")

            self.assertEqual(agent.system_prompt, "base\n\nYou are a reviewer.")
            self.assertEqual(session.system_prompt(), "base\n\nYou are a reviewer.")

    def test_append_system_prompt_on_resume_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with self.assertRaisesRegex(ValueError, "new session"):
                Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"),
                           append_system_prompt="role")
            self.assertEqual(session.system_prompt(), "snapshot")

    def test_prompt_does_not_enumerate_tool_names(self) -> None:
        # Schemas travel with every request; a prose list would drift from a
        # narrowed toolset.
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent.Session.create", return_value=session):
                agent = Agent.open(cwd=cwd, config=Config(model="test:stub"),
                                   toolset={"read_file": tools.REGISTRY["read_file"]})

            self.assertNotIn("You have these tools", agent.system_prompt)

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)

            with patch("paimon.agent.build_system_prompt") as generate:
                with self.assertRaisesRegex(SessionIncompleteError, "persisted system prompt"):
                    Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

            generate.assert_not_called()


class MentionAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_content_is_persisted_in_session_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello")
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
                _events = [event async for event in agent.run("review @hello.txt")]

            user_texts = [part.content
                          for message in session.messages() if isinstance(message, ModelRequest)
                          for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(len(user_texts), 1)
            self.assertIn('<mentioned_file path=', user_texts[0])
            self.assertIn("hello", user_texts[0])


class SessionLockReleaseTest(unittest.TestCase):
    """Agent.open takes the session lock early; no failure after that may keep it."""

    def test_unparsable_history_releases_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            session.append_system_prompt("snapshot")

            with patch.object(Session, "messages", side_effect=ValueError("corrupt")):
                with self.assertRaises(ValueError):
                    Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

            self.assertFalse(lockfile.held(session.path))

    def test_prompt_build_failure_releases_a_new_sessions_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent.build_system_prompt", side_effect=OSError("disk full")),
            ):
                with self.assertRaises(OSError):
                    Agent.open(cwd=cwd, config=Config(model="test:stub"))

            self.assertFalse(lockfile.held(session.path))

    def test_a_missing_snapshot_still_releases_the_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = make_session(cwd)
            with self.assertRaises(SessionIncompleteError):
                Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))

            self.assertFalse(lockfile.held(session.path))


class AgentCloseTest(unittest.TestCase):
    """An agent holds its session until it is closed, one way or another."""

    def _session(self, cwd: Path) -> Session:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        return session

    def test_with_block_releases_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with Agent.open(cwd=cwd, session=session, config=Config(model="test:stub")) as agent:
                self.assertIs(agent.session, session)
                self.assertTrue(lockfile.held(session.path))

            self.assertFalse(lockfile.held(session.path))

    def test_closing_twice_leaves_a_later_holder_alone(self) -> None:
        """Claims are refcounted per process: a stale close must not drop one."""
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)
            first = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            first.close()

            resumed = Session(session.path, session.id, cwd)
            second = Agent.open(cwd=cwd, session=resumed, config=Config(model="test:stub"))
            self.addCleanup(second.close)

            first.close()
            self.assertTrue(lockfile.held(resumed.path), "the second agent still holds it")

    def test_a_dropped_agent_releases_its_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)
            agent = Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"))
            self.assertTrue(lockfile.held(session.path))

            del agent
            gc.collect()
            self.assertFalse(lockfile.held(session.path))


class ModelOverrideTest(unittest.TestCase):
    """Agents share one Config, so a per-agent model cannot live in it."""

    def test_the_override_wins_and_the_shared_config_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            config = Config(model="test:stub",
                            providers={"test": {"api_base": "https://example/v1",
                                                "api_key": "k"}})
            plain = Agent(make_session(cwd), "snapshot", config=config)
            overridden = Agent(make_session(cwd), "snapshot", config=config,
                               model_override="test:other")

            self.assertEqual(plain.model_name, "test:stub")
            self.assertEqual(overridden.model_name, "test:other")
            self.assertEqual(config.model, "test:stub")

            with patch("paimon.agent.build_model", side_effect=lambda *key: key) as build:
                self.assertEqual(plain._model(), ("test:stub", "https://example/v1", "k"))
                self.assertEqual(overridden._model(), ("test:other", "https://example/v1", "k"))
            self.assertEqual(build.call_count, 2)

    def test_the_override_picks_its_own_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(make_session(Path(directory)), "snapshot",
                          config=Config(model="test:stub"), model_override="claude-x")
            self.assertEqual(compaction.context_window(agent.model_name), 200_000)


class SkillAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _write_skill(directory: Path, name: str) -> Path:
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        path.write_text(f"---\nname: {name}\ndescription: Use {name}.\n---\n# {name}\n\nDo the thing.")
        return path

    def test_project_skills_are_discovered_and_listed_in_the_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / ".git").mkdir()
            path = self._write_skill(cwd / ".agents" / "skills" / "demo", "demo")
            with patch("paimon.skills.default_skill_dirs", wraps=real_default_skill_dirs) as dirs, \
                    patch("paimon.skills.config_root", return_value=cwd / "no-config"), \
                    patch("paimon.skills.Path.home", return_value=cwd / "no-home"):
                agent = Agent.open(cwd=cwd, config=Config(model="test:stub"))
            self.addCleanup(agent.session.unlock)
            dirs.assert_called_once()
            self.assertEqual([s.path for s in agent.skills], [path])
            self.assertIn(f"<location>{path}</location>", agent.system_prompt)
            self.assertEqual(agent.skill_diagnostics, [])

    def test_no_defaults_loads_only_the_configured_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            self._write_skill(cwd / ".agents" / "skills" / "ambient", "ambient")
            explicit = self._write_skill(cwd / "elsewhere" / "explicit", "explicit")
            config = Config(model="test:stub")
            config.skills = [str(explicit.parent)]
            config.include_default_skills = False
            with patch("paimon.skills.default_skill_dirs") as dirs:
                agent = Agent.open(cwd=cwd, config=config)
            self.addCleanup(agent.session.unlock)
            dirs.assert_not_called()
            self.assertEqual([s.name for s in agent.skills], ["explicit"])

    async def test_skill_command_is_expanded_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            path = self._write_skill(cwd / "sk" / "demo", "demo")
            (cwd / "hello.txt").write_text("hello")
            config = Config(model="test:stub")
            config.skills = [str(cwd / "sk")]
            path.write_text(path.read_text() + "\nSee @hello.txt for context.")
            with patch("paimon.agent.build_model", return_value=stub_model()):
                agent = Agent.open(cwd=cwd, config=config)
                self.addCleanup(agent.session.unlock)
                _events = [event async for event in agent.run("/skill:demo also @hello.txt")]
            user_texts = [part.content
                          for message in agent.session.messages() if isinstance(message, ModelRequest)
                          for part in message.parts if isinstance(part, UserPromptPart)]
            self.assertEqual(len(user_texts), 1)
            self.assertTrue(user_texts[0].startswith(f'<skill name="demo" location="{path}">'))
            self.assertIn("Do the thing.", user_texts[0])
            self.assertEqual(user_texts[0].count("<mentioned_file path="), 1,
                             "the user's mention expands, the one in the skill body does not")
            self.assertIn("See @hello.txt for context.", user_texts[0])
