import tempfile
import unittest
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static

from paimon import tools
from paimon.textual_debug import TextualDebugBridge


class DebugApp(App):
    CSS = "#target { width: 10; color: red; }"

    def compose(self) -> ComposeResult:
        yield Static("hello", id="target", classes="sample")


class TextualDebugBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_tools_inspect_and_modify_the_live_app(self) -> None:
        bridge = TextualDebugBridge()
        registry = bridge.toolset()
        app = DebugApp()
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            async with app.run_test(size=(40, 10)) as pilot:
                bridge.attach(app)

                inspected = await tools.execute_tool(
                    "textual_inspect", {"selector": "#target", "include_styles": True},
                    cwd, registry=registry,
                )
                self.assertIn("Static#target.sample", inspected)
                self.assertIn("width: 10", inspected)

                evaluated = await tools.execute_tool(
                    "textual_eval", {"expression": "query_one('#target').region.width"},
                    cwd, registry=registry,
                )
                self.assertEqual(evaluated, "10")

                executed = await tools.execute_tool(
                    "textual_exec",
                    {"code": "widget = query_one('#target')\nwidget.styles.width = 17\nresult = widget.id"},
                    cwd, registry=registry,
                )
                await pilot.pause()
                self.assertEqual(executed, "'target'")
                self.assertEqual(app.query_one("#target").region.width, 17)

                applied = await tools.execute_tool(
                    "textual_apply_css", {"css": "#target { height: 3; }"},
                    cwd, registry=registry,
                )
                await pilot.pause()
                self.assertIn("applied", applied)
                self.assertEqual(app.query_one("#target").region.height, 3)

                result = await tools.execute_tool(
                    "textual_screenshot", {"path": "shots/ui.svg", "simplify": True},
                    cwd, registry=registry,
                )
                screenshot = cwd / "shots/ui.svg"
                self.assertIn(str(screenshot), result)
                self.assertIn("<svg", screenshot.read_text(encoding="utf-8"))

    async def test_exec_namespace_persists_and_supports_top_level_await(self) -> None:
        bridge = TextualDebugBridge()
        registry = bridge.toolset()
        app = DebugApp()
        async with app.run_test():
            bridge.attach(app)
            await tools.execute_tool(
                "textual_exec", {"code": "import asyncio\nvalue = 40\nawait asyncio.sleep(0)"},
                Path.cwd(), registry=registry,
            )
            result = await tools.execute_tool(
                "textual_eval", {"expression": "value + 2"}, Path.cwd(), registry=registry,
            )
            self.assertEqual(result, "42")
