"""Pictures drawn in the terminal: the protocol, and where the log puts them."""

import base64
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import PIL.Image
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart, UserPromptPart

from paimon import termimage
from paimon.events import TextDelta, ToolEnd, ToolStart, UserInput, replay_events
from paimon.mentions import linked_images, mentioned_images
from paimon.tools.result import Image
from paimon.transcript import EventRenderer
from paimon.ui import AssistantMessage, Picture
from tests.support.app import AppTestCase


def encoded(kind: str, width: int, height: int, mode: str = "RGB", colour="red") -> bytes:
    file = io.BytesIO()
    PIL.Image.new(mode, (width, height), colour).save(file, kind)
    return file.getvalue()


def png(width: int, height: int) -> bytes:
    return encoded("PNG", width, height)


def size(data: bytes) -> tuple[int, int]:
    with PIL.Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"
        return image.size


BROKEN = b"\xff\xd8\xff\xe0" + b"\x00" * 64


class SupportTest(unittest.TestCase):
    def test_kitty_and_ghostty_are_told_from_the_environment(self) -> None:
        for environ in ({"KITTY_WINDOW_ID": "1"}, {"TERM": "xterm-kitty"},
                        {"TERM": "xterm-ghostty"}, {"TERM_PROGRAM": "ghostty"}):
            self.assertTrue(termimage.supported(environ), environ)

    def test_other_terminals_and_multiplexers_get_none(self) -> None:
        for environ in ({}, {"TERM": "xterm-256color", "TERM_PROGRAM": "WezTerm"},
                        {"TERM_PROGRAM": "iTerm.app"},
                        {"KITTY_WINDOW_ID": "1", "TMUX": "/tmp/tmux-1000/default,1,0"},
                        {"KITTY_WINDOW_ID": "1", "TERM": "tmux-256color"},
                        {"TERM": "xterm-kitty", "ZELLIJ": "0"}):
            self.assertFalse(termimage.supported(environ), environ)


class MeasureTest(unittest.TestCase):
    def test_a_small_picture_keeps_its_size(self) -> None:
        self.assertEqual(termimage.fit((80, 32), (8, 16)), (10, 2))
        self.assertEqual(termimage.fit((1, 1), (8, 16)), (1, 1))

    def test_a_large_one_is_scaled_into_the_box_in_its_own_shape(self) -> None:
        # Wide: the columns run out first. Tall: the rows do.
        self.assertEqual(termimage.fit((3200, 1600), (8, 16)), (40, 10))
        self.assertEqual(termimage.fit((1000, 4000), (8, 16)), (6, 12))
        self.assertEqual(termimage.fit((100_000, 10), (8, 16)), (40, 1))


class PrepareTest(unittest.TestCase):
    def test_a_small_png_is_sent_as_it_is(self) -> None:
        shot = png(80, 32)
        self.assertEqual(termimage.prepare(shot, (8, 16)), (shot, 10, 2))

    def test_every_format_becomes_a_png(self) -> None:
        for kind in ("JPEG", "GIF", "WEBP"):
            sent, columns, rows = termimage.prepare(encoded(kind, 80, 32), (8, 16))
            self.assertEqual((size(sent), columns, rows), ((80, 32), 10, 2), kind)

    def test_a_large_one_is_drawn_down_to_the_cells_it_gets(self) -> None:
        for kind in ("PNG", "JPEG", "WEBP"):
            sent, columns, rows = termimage.prepare(encoded(kind, 3200, 1600), (8, 16))
            self.assertEqual((size(sent), columns, rows), ((320, 160), 40, 10), kind)

    def test_transparency_and_a_palette_survive(self) -> None:
        sent, _, _ = termimage.prepare(encoded("WEBP", 16, 16, "RGBA", (255, 0, 0, 128)), (8, 16))
        with PIL.Image.open(io.BytesIO(sent)) as image:
            self.assertEqual(image.mode, "RGBA")
        self.assertIsNotNone(termimage.prepare(encoded("GIF", 16, 16, "P"), (8, 16)))

    def test_a_photograph_turned_by_the_camera_is_stood_up(self) -> None:
        file = io.BytesIO()
        exif = PIL.Image.Exif()
        exif[0x0112] = 6  # Orientation: the top of the scene is on the right.
        PIL.Image.new("RGB", (160, 80), "red").save(file, "JPEG", exif=exif)
        sent, columns, rows = termimage.prepare(file.getvalue(), (8, 16))
        self.assertEqual((size(sent), columns, rows), ((80, 160), 10, 10))

    def test_what_will_not_open_is_passed_over(self) -> None:
        for data in (BROKEN, b"", b"plain text", png(80, 32)[:40]):
            self.assertIsNone(termimage.prepare(data, (8, 16)))


class SequenceTest(unittest.TestCase):
    def test_a_picture_is_sent_whole_in_chunks_the_protocol_takes(self) -> None:
        data = bytes(range(256)) * 40
        sent = termimage.transmit(7, data, 5, 3)
        parts = re.findall(r"\x1b_G([^;]*);([^\x1b]*)\x1b\\", sent)
        self.assertEqual("".join(f"\x1b_G{keys};{chunk}\x1b\\" for keys, chunk in parts), sent)
        self.assertGreater(len(parts), 1)
        self.assertEqual(parts[0][0], "a=T,U=1,f=100,i=7,c=5,r=3,q=2,m=1")
        self.assertEqual([keys for keys, _ in parts[1:]], ["m=1"] * (len(parts) - 2) + ["m=0"])
        self.assertTrue(all(len(chunk) <= 4096 for _, chunk in parts))
        self.assertEqual(base64.standard_b64decode("".join(chunk for _, chunk in parts)), data)

    def test_one_that_fits_a_chunk_says_it_is_the_last(self) -> None:
        self.assertTrue(termimage.transmit(7, png(4, 4), 1, 1).startswith(
            "\x1b_Ga=T,U=1,f=100,i=7,c=1,r=1,q=2,m=0;"))

    def test_a_row_names_itself_once_and_is_as_wide_as_the_picture(self) -> None:
        self.assertEqual(termimage.placeholder(0, 3), "\U0010eeee̅\U0010eeee\U0010eeee")
        self.assertEqual(termimage.placeholder(2, 1), "\U0010eeee̎")

    def test_a_number_fits_the_colour_that_carries_it(self) -> None:
        self.assertEqual(termimage.colour(0x0A0B0C), (10, 11, 12))
        first, second = termimage.new_id(), termimage.new_id()
        self.assertNotEqual(first, second)
        for number in (first, second):
            self.assertTrue(0 < number <= 0xFFFFFF)


class EventImagesTest(unittest.TestCase):
    def test_a_replay_carries_the_images_beside_the_words(self) -> None:
        shot = png(8, 8)
        events = replay_events([ModelRequest(parts=[
            UserPromptPart(content=["look", BinaryContent(data=shot, media_type="image/png")]),
            ToolReturnPart(tool_name="read_file", tool_call_id="c1",
                           content=["an image", BinaryContent(data=shot, media_type="image/png")]),
        ])])
        self.assertEqual([type(event) for event in events], [UserInput, ToolEnd])
        self.assertEqual(events[0].text, "look")
        for event in events:
            self.assertEqual(event.images, (Image(shot, "image/png"),))

    def test_the_images_a_prompt_mentions_are_found(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(png(8, 8))
            (cwd / "notes.txt").write_text("words")
            self.assertEqual(mentioned_images("see @shot.png and @notes.txt", cwd),
                             (Image(png(8, 8), "image/png"),))
            self.assertEqual(mentioned_images("nothing here", cwd), ())


class PictureTest(AppTestCase):
    async def test_a_terminal_without_the_protocol_gets_the_words_alone(self) -> None:
        app = self.make_app()
        async with app.run_test() as pilot:
            app.pane.transcript.add_user("look", [Image(png(80, 32), "image/png")])
            await pilot.pause()
            self.assertEqual(len(app.query(Picture)), 0)

    async def test_an_attached_image_is_sent_and_drawn_under_the_message(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            written: list[str] = []
            with patch.object(Picture, "drawn_by", return_value=True), \
                    patch.object(termimage, "cell_size", return_value=(8, 16)), \
                    patch.object(app._driver, "write", written.append):
                renderer = EventRenderer(app.pane.transcript, app.pane.agent)
                await renderer.handle(UserInput("look", (Image(png(80, 32), "image/png"),
                                                         Image(BROKEN, "image/jpeg"))))
                await pilot.pause()
                # Only the one that opens.
                picture = app.query_one(Picture)
                self.assertEqual((picture.size.width, picture.size.height), (10, 2))
                sent = [text for text in written if text.startswith("\x1b_Ga=T")]
                self.assertEqual(len(sent), 1)
                self.assertIn(f"i={picture._id},c=10,r=2,", sent[0])

                lines = [picture.render_line(y) for y in range(2)]
                self.assertEqual([line.text for line in lines],
                                 [termimage.placeholder(0, 10), termimage.placeholder(1, 10)])
                self.assertEqual([line.cell_length for line in lines], [10, 10])
                colour = next(iter(lines[0])).style.color.triplet
                self.assertEqual(tuple(colour), termimage.colour(picture._id))

                await picture.remove()
                self.assertIn(termimage.delete(picture._id), written)

    async def test_an_image_a_tool_read_is_drawn_with_its_result(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            with patch.object(Picture, "drawn_by", return_value=True):
                renderer = EventRenderer(app.pane.transcript, app.pane.agent)
                await renderer.handle(ToolStart("c1", "read_file", {"path": "shot.png"}))
                await renderer.handle(ToolEnd("c1", "read_file", "shot.png is an image",
                                              images=(Image(png(80, 32), "image/png"),)))
                await pilot.pause()
                picture = app.query_one(".tool-entry-detail Picture", Picture)
                self.assertIsNotNone(picture)


class LinkedImagesTest(unittest.TestCase):
    def test_an_answer_shows_the_local_images_it_links_to(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "chart.png").write_bytes(png(8, 8))
            (cwd / "my shot.png").write_bytes(png(9, 9))
            (cwd / "notes.txt").write_text("words")
            text = ("Here: ![the chart](chart.png) and ![](<my shot.png>) with "
                    f'![abs]({cwd / "chart.png"} "title").\n'
                    "Not these: ![remote](https://example.com/a.png) ![gone](gone.png) "
                    "![text](notes.txt) [a link](chart.png)\n"
                    "```\n![in code](chart.png)\n```\n")
            self.assertEqual([size(image.data) for image in linked_images(text, cwd)],
                             [(8, 8), (9, 9), (8, 8)])


class AnswerPictureTest(AppTestCase):
    async def test_an_image_the_answer_links_to_is_drawn_under_it(self) -> None:
        app = self.make_app()
        async with app.run_test(size=(100, 40)) as pilot:
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(Picture, "drawn_by", return_value=True):
                shot = Path(directory) / "chart.png"
                shot.write_bytes(png(80, 32))
                renderer = EventRenderer(app.pane.transcript, app.pane.agent)
                await renderer.handle(TextDelta(f"Done: ![the chart]({shot}"))
                await renderer.handle(TextDelta(")"))
                self.assertEqual(len(app.query(Picture)), 0, "not before the block is whole")
                await renderer.close()
                await pilot.pause()
                log = app.query_one("#log")
                self.assertIsInstance(log.children[-1], Picture)
                self.assertIsInstance(log.children[-2], AssistantMessage)
