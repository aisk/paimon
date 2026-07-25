import tempfile
import unittest
from pathlib import Path

from paimon.mentions import MAX_INLINE_BYTES, expand_mentions


class ExpandMentionsTest(unittest.TestCase):
    def test_small_file_is_inlined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("hello\n")

            expanded = expand_mentions("review @hello.txt", cwd)

            self.assertIn(f'<mentioned_file path="{(cwd / "hello.txt").resolve()}">', expanded)
            self.assertIn("hello", expanded)
            self.assertTrue(expanded.endswith("</mentioned_file>"))

    def test_large_file_is_referenced_by_path_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "big.txt").write_text("x" * (MAX_INLINE_BYTES + 1))

            expanded = expand_mentions("@big.txt", cwd)

            self.assertEqual(expanded, f'<mentioned_file path="{(cwd / "big.txt").resolve()}" />')

    def test_binary_file_is_referenced_by_path_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")

            expanded = expand_mentions("@blob.bin", cwd)

            self.assertEqual(expanded, f'<mentioned_file path="{(cwd / "blob.bin").resolve()}" />')

    def test_escaped_space_in_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "with space.txt").write_text("contents")

            expanded = expand_mentions("@with\\ space.txt", cwd)

            self.assertIn("contents", expanded)

    def test_absolute_path_mention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            target = cwd / "hello.txt"
            target.write_text("hello")

            expanded = expand_mentions(f"@{target}", Path.cwd())

            self.assertIn("hello", expanded)

    def test_trailing_punctuation_is_not_part_of_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "hello.txt").write_text("contents")

            expanded = expand_mentions("看看 @hello.txt，还有 @hello.txt).", cwd)

            self.assertEqual(expanded.count("contents"), 2)
            self.assertTrue(expanded.endswith("</mentioned_file>)."))
            self.assertIn("，还有", expanded)

    def test_punctuation_in_a_real_filename_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "weird.").write_text("contents")

            expanded = expand_mentions("@weird.", cwd)

            self.assertIn(f'path="{(cwd / "weird.").resolve()}"', expanded)

    def test_unresolvable_mentions_are_left_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)

            expanded = expand_mentions("see @missing.txt and\n@property below", cwd)

            self.assertEqual(expanded, "see @missing.txt and\n@property below")

    def test_directory_mention_is_left_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "pkg").mkdir()

            self.assertEqual(expand_mentions("@pkg", cwd), "@pkg")


if __name__ == "__main__":
    unittest.main()
