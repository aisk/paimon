import contextlib
import io
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon import cli, commands
from paimon.config import Config
from paimon.tools.registry import REGISTRY

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _help(run, argv: list[str]) -> str:
    """What ``run`` prints for --help."""
    out = io.StringIO()
    with patch("sys.argv", argv), contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        run()
    return out.getvalue()


def _switches(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[a-z][a-z-]+", text)) - {"--help"}


class ReferenceTest(unittest.TestCase):
    """The reference page is written by hand, so what it must list is checked
    against the code: a switch, command, tool or setting added without a line
    there fails here."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (DOCS / "reference.html").read_text(encoding="utf-8")

    def assertListed(self, names: set[str], before: str = "<code>") -> None:
        """Each name opens a code span, and is not just the start of a longer one."""
        missing = sorted(name for name in names
                         if not re.search(re.escape(before + name) + r"(?![\w.-])", self.page))
        self.assertEqual(missing, [], "not in docs/reference.html")

    def test_every_launch_switch_is_listed(self) -> None:
        switches = _switches(_help(cli.main, ["paimon", "--help"]))
        self.assertIn("--mode", switches)
        self.assertListed(switches)

    def test_every_command_and_its_switches_are_listed(self) -> None:
        for name, run in commands.REGISTRY.items():
            with self.subTest(name):
                self.assertListed({f"paimon {name}"})
                self.assertListed(_switches(_help(lambda: run(["--help"]), ["paimon", name])))

    def test_every_tool_is_listed(self) -> None:
        self.assertListed(set(REGISTRY), before="<td><code>")

    def test_every_setting_is_listed(self) -> None:
        settings = {name.replace("compaction_", "compaction.", 1) for name in vars(Config())}
        self.assertListed(settings, before="<td><code>")


class SiteTest(unittest.TestCase):
    def test_pages_link_only_to_pages_that_exist(self) -> None:
        for page in DOCS.glob("*.html"):
            for href in re.findall(r'href="\./([^"#]*)', page.read_text(encoding="utf-8")):
                with self.subTest(page=page.name, href=href):
                    self.assertTrue(href == "" or (DOCS / href).is_file())

    def test_links_into_a_page_name_a_heading_there(self) -> None:
        for page in DOCS.glob("*.html"):
            text = page.read_text(encoding="utf-8")
            for target, anchor in re.findall(r'href="(\./[^"#]*)?#([^"]+)"', text):
                source = (DOCS / target[2:]).read_text(encoding="utf-8") if target else text
                with self.subTest(page=page.name, anchor=anchor):
                    self.assertIn(f'id="{anchor}"', source)
