import tempfile
import unittest
from pathlib import Path

from paimon.mentions import MentionExpander


class MentionExpanderTest(unittest.TestCase):
    def test_expands_first_version_and_references_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("hello\n")
            expander = MentionExpander(cwd)

            first = expander.expand("review @hello.txt")
            second = expander.expand("review @hello.txt")

            self.assertIn('exposure="full"', first)
            self.assertIn("hello", first)
            self.assertIn('status="previously_mentioned"', second)
            self.assertNotIn("\nhello\n", second)

    def test_changed_file_is_a_new_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("first\n")
            expander = MentionExpander(cwd)
            first = expander.expand("@hello.txt")
            target.write_text("second\n")
            second = expander.expand("@hello.txt")

            self.assertIn("first", first)
            self.assertIn("second", second)
            self.assertNotIn('status="previously_mentioned"', second)

    def test_escaped_space_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "with space.txt").write_text("contents")
            expander = MentionExpander(cwd)

            expanded = expander.expand("@with\\ space.txt @missing.txt")

            self.assertIn("contents", expanded)
            self.assertIn('requested="missing.txt" status="not_found"', expanded)

    def test_restores_versions_from_persisted_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("hello")
            original = MentionExpander(cwd).expand("@hello.txt")

            resumed = MentionExpander(cwd, [{"role": "user", "content": original}])
            repeated = resumed.expand("@hello.txt")

            self.assertIn('status="previously_mentioned"', repeated)


if __name__ == "__main__":
    unittest.main()
