import unittest
from unittest.mock import patch

from rich.table import Table
from rich.text import Text

from paimon.diff import _table_diff, render_diff


class TableDiffTest(unittest.TestCase):
    def test_hunks_stay_row_aligned(self) -> None:
        table = _table_diff("a\nb\nc", "a\nx\ny\nc")
        # 1 equal + max(1, 2) changed + 1 equal rows
        self.assertEqual(table.row_count, 4)

    def test_render_diff_falls_back_without_delta(self) -> None:
        with patch("paimon.diff.shutil.which", return_value=None):
            self.assertIsInstance(render_diff("a", "b", width=80), Table)

    def test_render_diff_uses_delta_output(self) -> None:
        with patch("paimon.diff.shutil.which", return_value="/usr/bin/delta"), patch(
            "paimon.diff.subprocess.run"
        ) as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "colored diff"
            self.assertIsInstance(render_diff("a", "b", width=80), Text)

    def test_render_diff_falls_back_on_delta_failure(self) -> None:
        with patch("paimon.diff.shutil.which", return_value="/usr/bin/delta"), patch(
            "paimon.diff.subprocess.run"
        ) as run:
            run.return_value.returncode = 1
            run.return_value.stdout = ""
            self.assertIsInstance(render_diff("a", "b", width=80), Table)


if __name__ == "__main__":
    unittest.main()
