import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual import events

from paimon import clipboard
from paimon.mentions import expand_mentions
from paimon.tools.files import MAX_IMAGE_BYTES
from paimon.tools.result import Image
from paimon.ui import PromptInput
from tests.support.app import AppTestCase

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class ClipboardImageTest(unittest.TestCase):
    def test_what_the_clipboard_tool_prints_is_read_as_an_image(self) -> None:
        with patch("paimon.clipboard._read", return_value=PNG):
            self.assertEqual(clipboard.read_image(), Image(PNG, "image/png"))

    def test_a_clipboard_without_an_image_gives_none(self) -> None:
        for data in (None, b"", b"just text"):
            with patch("paimon.clipboard._read", return_value=data):
                self.assertIsNone(clipboard.read_image())

    def test_an_image_too_large_to_send_is_refused(self) -> None:
        with patch("paimon.clipboard._read", return_value=PNG + b"\x00" * MAX_IMAGE_BYTES):
            with self.assertRaises(clipboard.ClipboardError):
                clipboard.read_image()

    def test_with_no_desktop_there_is_nothing_to_read(self) -> None:
        with patch("paimon.clipboard.sys.platform", "linux"), \
                patch.dict("os.environ", clear=True):
            self.assertIsNone(clipboard._read())

    def test_the_best_format_on_offer_is_asked_for(self) -> None:
        asked: list[list[str]] = []

        def output(command: list[str]) -> bytes:
            asked.append(command)
            return b"TARGETS\ntext/html\nimage/bmp\nimage/jpeg\n" if "TARGETS" in command else b"jpeg"

        with patch("paimon.clipboard.sys.platform", "linux"), \
                patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True), \
                patch("paimon.clipboard._output", output):
            self.assertEqual(clipboard._read(), b"jpeg")
        self.assertEqual(asked[-1][asked[-1].index("-t") + 1], "image/jpeg")

    def test_inside_wsl_the_windows_clipboard_is_the_fallback(self) -> None:
        def output(command: list[str]) -> bytes | None:
            if command[0] == "wslpath":
                return b"\\\\wsl.localhost\\Ubuntu\\tmp\\x\\clipboard.png\n"
            if command[0] == "powershell.exe":
                Path(written[0]).write_bytes(PNG)
                return b""
            return None

        written: list[str] = []
        real = clipboard.tempfile.TemporaryDirectory

        def directory():
            made = real()
            written.append(str(Path(made.name) / "clipboard.png"))
            return made

        with patch("paimon.clipboard.sys.platform", "linux"), \
                patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True), \
                patch("paimon.clipboard.tempfile.TemporaryDirectory", directory), \
                patch("paimon.clipboard._output", output):
            self.assertEqual(clipboard.read_image(), Image(PNG, "image/png"))

    def test_a_path_dragged_from_explorer_is_found_under_mnt(self) -> None:
        with patch("paimon.clipboard.is_wsl", return_value=True), \
                patch("paimon.clipboard.Path.is_file", return_value=False) as is_file:
            clipboard.pasted_image_path('"C:\\Users\\me\\my shot.png"')
        self.assertEqual(is_file.call_count, 1)
        with patch("paimon.clipboard.is_wsl", return_value=True), \
                patch("paimon.clipboard.Path.is_absolute", lambda path: seen.append(path) or False):
            seen: list[Path] = []
            clipboard.pasted_image_path("C:\\Users\\me\\shot.png")
        self.assertEqual(seen, [Path("/mnt/c/Users/me/shot.png")])

    def test_saved_images_never_overwrite_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("paimon.clipboard.time.strftime", return_value="20260101-000000"):
                first = clipboard.save(Image(PNG, "image/png"), Path(directory) / "pasted")
                second = clipboard.save(Image(PNG, "image/png"), Path(directory) / "pasted")

            self.assertNotEqual(first, second)
            self.assertEqual(first.suffix, ".png")
            self.assertEqual(second.read_bytes(), PNG)


class PastedPathTest(unittest.TestCase):
    def test_a_dropped_image_is_recognized_however_the_terminal_writes_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "my shot.png"
            path.write_bytes(PNG)
            for text in (str(path).replace(" ", "\\ "), f"'{path}'", f'"{path}"',
                         path.as_uri(), f"  {path.as_uri()}\n"):
                with self.subTest(text=text):
                    self.assertEqual(clipboard.pasted_image_path(text), path)

    def test_other_pastes_are_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory).resolve()
            (cwd / "notes.txt").write_text("hello")
            (cwd / "shot.png").write_bytes(PNG)
            for text in ("", "hello", str(cwd / "notes.txt"), str(cwd / "missing.png"),
                         "shot.png", f"look at {cwd / 'shot.png'}",
                         f"{cwd / 'shot.png'}\n{cwd / 'shot.png'}"):
                with self.subTest(text=text):
                    self.assertIsNone(clipboard.pasted_image_path(text))

    def test_the_mention_written_attaches_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "my shot.png"
            path.write_bytes(PNG)
            images: list = []

            expanded = expand_mentions(f"see {clipboard.mention(path)} please", Path(directory), images)

            self.assertIn("<mentioned_image", expanded)
            self.assertTrue(expanded.endswith(" please"))
            self.assertEqual(len(images), 1)


class PromptPasteTest(AppTestCase):
    async def _submitted(self, app, pilot) -> str:
        sent: list[str] = []
        with patch.object(app.pane.driver, "submit", sent.append):
            await pilot.press("enter")
            await self._wait_for(pilot, lambda: bool(sent))
        return sent[0]

    async def test_ctrl_v_attaches_a_clipboard_image_behind_a_placeholder(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            with patch("paimon.clipboard.read_image", return_value=Image(PNG, "image/png")):
                await pilot.press("ctrl+v")
                await self._wait_for(pilot, lambda: bool(prompt.text))
            self.assertEqual(prompt.text, "[Image #1] ")
            await pilot.press("w", "h", "a", "t")

            sent = await self._submitted(app, pilot)

            self.assertTrue(sent.startswith("@"))
            self.assertTrue(sent.endswith(".png what"))
            saved = Path(sent[1:].removesuffix(" what"))
            self.assertEqual(saved.read_bytes(), PNG)
            self.assertEqual(saved.parent, Path(os.environ["PAIMON_DATA_HOME"]) / "pasted")

    async def test_ctrl_v_without_an_image_pastes_text_as_before(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            app.copy_to_clipboard("plain words")
            with patch("paimon.clipboard.read_image", return_value=None):
                await pilot.press("ctrl+v")
                await self._wait_for(pilot, lambda: bool(prompt.text))

            self.assertEqual(prompt.text, "plain words")

    async def test_a_dropped_image_path_is_attached_and_other_text_stays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "shot.png"
            path.write_bytes(PNG)
            app = self.make_app()
            async with app.run_test() as pilot:
                prompt = app.query_one(PromptInput)
                app.post_message(events.Paste(f"'{path}'"))
                await self._wait_for(pilot, lambda: bool(prompt.text))
                self.assertEqual(prompt.text, "[Image #1] ")

                app.post_message(events.Paste("and words"))
                await self._wait_for(pilot, lambda: prompt.text.endswith("words"))
                self.assertEqual(prompt.text, "[Image #1] and words")
                self.assertEqual(await self._submitted(app, pilot), f"@{path} and words")

    async def test_backspace_takes_a_whole_placeholder_and_typed_ones_stay_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "shot.png"
            path.write_bytes(PNG)
            app = self.make_app()
            async with app.run_test() as pilot:
                prompt = app.query_one(PromptInput)
                prompt.insert("see ")
                app.post_message(events.Paste(str(path)))
                await self._wait_for(pilot, lambda: prompt.text.endswith("] "))
                await pilot.press("backspace")
                self.assertEqual(prompt.text, "see ")
                await pilot.press("backspace")
                self.assertEqual(prompt.text, "see")

                # A number nothing was pasted under is the user's own text.
                prompt.insert(" [Image #7]")
                await pilot.press("backspace")
                self.assertEqual(prompt.text, "see [Image #7")
                prompt.insert("]")
                self.assertEqual(await self._submitted(app, pilot), "see [Image #7]")
