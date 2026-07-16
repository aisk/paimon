import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon.agent import Agent
from paimon.session import Session


class AgentSystemPromptTest(unittest.TestCase):
    @staticmethod
    def _session(cwd: Path) -> Session:
        session = Session(cwd / "session.jsonl", "session-id", cwd)
        session.append({
            "type": "session",
            "version": 1,
            "id": "session-id",
            "cwd": str(cwd),
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        return session

    def test_system_prompt_is_generated_once_then_loaded_from_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with (
                patch("paimon.agent.Session.create", return_value=session),
                patch("paimon.agent._system_prompt", return_value="snapshot") as generate,
            ):
                first = Agent(cwd=cwd)

            self.assertEqual(first.messages[0], {"role": "system", "content": "snapshot"})
            self.assertEqual(session.system_prompt(), "snapshot")
            generate.assert_called_once_with(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                resumed = Agent(cwd=cwd, session=session)

            self.assertEqual(resumed.messages[0], {"role": "system", "content": "snapshot"})
            generate.assert_not_called()

    def test_session_without_snapshot_does_not_regenerate_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            session = self._session(cwd)

            with patch("paimon.agent._system_prompt") as generate:
                with self.assertRaisesRegex(RuntimeError, "persisted system prompt"):
                    Agent(cwd=cwd, session=session)

            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
