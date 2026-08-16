import tempfile
import unittest
from pathlib import Path

from rich.text import Text

from paimon.diff import _highlight, _refine, locate_line, render_diff


class RenderDiffTest(unittest.TestCase):
    def test_hunks_stay_row_aligned(self) -> None:
        table = render_diff("a\nb\nc", "a\nx\ny\nc")
        # 1 equal + max(1, 2) changed + 1 equal rows
        self.assertEqual(table.row_count, 4)

    def test_line_numbers_start_at_start_line(self) -> None:
        table = render_diff("a\nb", "a\nx", start_line=10)
        left_numbers = [str(cell) for cell in table.columns[0]._cells]
        right_numbers = [str(cell) for cell in table.columns[3]._cells]
        self.assertEqual(left_numbers, ["10", "11"])
        self.assertEqual(right_numbers, ["10", "11"])

    def test_no_line_numbers_when_start_unknown(self) -> None:
        table = render_diff("a\nb", "a\nx")
        self.assertEqual(len(table.columns), 3)

    def test_python_code_gets_syntax_colors(self) -> None:
        lines = _highlight("def f():\n    return 1", "m.py", "monokai")
        self.assertEqual([line.plain for line in lines], ["def f():", "    return 1"])
        self.assertTrue(lines[0].spans, "keywords should carry color spans")
        self.assertTrue(
            all(s.style.bgcolor is None for s in lines[0].spans
                if not isinstance(s.style, str)),
            "token backgrounds must be stripped",
        )

    def test_unknown_file_type_stays_plain(self) -> None:
        table = render_diff("hello", "world", path="notes.xyzzy")
        self.assertEqual(table.row_count, 1)

    def test_refine_marks_only_the_changed_characters(self) -> None:
        a_text, b_text = Text("x = 1"), Text("x = 2")
        _refine("x = 1", "x = 2", a_text, b_text, "#6f2b30", "#2b6f42")
        self.assertEqual([(s.start, s.end) for s in a_text.spans], [(4, 5)])
        self.assertEqual([(s.start, s.end) for s in b_text.spans], [(4, 5)])

    def test_refine_skips_full_rewrites(self) -> None:
        a_text, b_text = Text("aaaa"), Text("zzzz")
        _refine("aaaa", "zzzz", a_text, b_text, "#6f2b30", "#2b6f42")
        self.assertFalse(a_text.spans)
        self.assertFalse(b_text.spans)


class LocateLineTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "m.py"
        self.path.write_text("one\ntwo\nthree\n")

    def test_finds_old_string(self) -> None:
        self.assertEqual(locate_line(str(self.path), "two\nthree", "x"), 2)

    def test_falls_back_to_new_string_after_the_edit_ran(self) -> None:
        self.assertEqual(locate_line(str(self.path), "gone", "three"), 3)

    def test_relative_path_resolves_against_cwd(self) -> None:
        self.assertEqual(
            locate_line("m.py", "two", "x", cwd=self.path.parent), 2)

    def test_unlocatable_returns_none(self) -> None:
        self.assertIsNone(locate_line(str(self.path), "gone", "also gone"))
        self.assertIsNone(locate_line("no/such/file", "a", "b"))


if __name__ == "__main__":
    unittest.main()
