import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from paimon import compaction, headless, retry
from paimon.agent import Agent
from paimon.config import Config
from paimon.events import ImagesRejected, ToolEnd, UserInput, replay_events
from paimon.images import UNSEEN, content_text
from paimon.mentions import expand_mentions
from paimon.session import Session
from paimon.tools.files import MAX_IMAGE_BYTES, image_type
from paimon.tools.registry import execute_tool
from tests.support.agent import make_session

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _images(messages) -> list[BinaryContent]:
    return [item for message in messages if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, (UserPromptPart, ToolReturnPart)) and isinstance(part.content, list)
            for item in part.content if isinstance(item, BinaryContent)]


class ImageTypeTest(unittest.TestCase):
    def test_the_type_is_told_from_the_content(self) -> None:
        self.assertEqual(image_type(PNG), "image/png")
        self.assertEqual(image_type(JPEG), "image/jpeg")
        self.assertEqual(image_type(b"GIF89a..."), "image/gif")
        self.assertEqual(image_type(b"RIFF\x10\x00\x00\x00WEBPVP8 "), "image/webp")
        self.assertIsNone(image_type(b"RIFF\x10\x00\x00\x00WAVEfmt "))
        self.assertIsNone(image_type(b"plain text"))
        self.assertIsNone(image_type(b""))


class MentionedImageTest(unittest.TestCase):
    def test_a_mentioned_image_is_collected_and_tagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            # Named for another format: the content decides.
            (cwd / "shot.png").write_bytes(JPEG)
            (cwd / "a.txt").write_text("hello\n")
            images: list = []

            expanded = expand_mentions("compare @shot.png with @a.txt", cwd, images)

            self.assertIn(f'<mentioned_image path="{(cwd / "shot.png").resolve()}" />', expanded)
            self.assertIn("hello", expanded)
            self.assertEqual([(image.data, image.media_type) for image in images],
                             [(JPEG, "image/jpeg")])

    def test_without_a_list_an_image_is_only_referenced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)

            self.assertEqual(expand_mentions("@shot.png", cwd),
                             f'<mentioned_file path="{(cwd / "shot.png").resolve()}" />')

    def test_an_image_too_large_to_send_is_not_attached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "huge.png").write_bytes(PNG + b"\x00" * MAX_IMAGE_BYTES)
            images: list = []

            expanded = expand_mentions("@huge.png", cwd, images)

            self.assertIn('error="too large to attach"', expanded)
            self.assertEqual(images, [])

    def test_a_headless_prompt_attaches_its_images_but_not_piped_ones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)

            content = headless.build_prompt("what is in @shot.png", "see @shot.png", cwd)

            self.assertIsInstance(content, list)
            self.assertIn("see @shot.png", content[0])
            self.assertIn("<mentioned_image", content[0])
            self.assertEqual([item.data for item in content[1:]], [PNG])


class ReadImageTest(unittest.IsolatedAsyncioTestCase):
    async def test_read_file_returns_the_picture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)

            result = await execute_tool("read_file", {"path": "shot.png"}, cwd)

            self.assertTrue(result.ok)
            self.assertIn("image/png", result.text)
            self.assertEqual([(image.data, image.media_type) for image in result.images],
                             [(PNG, "image/png")])

    async def test_read_file_refuses_an_image_too_large_to_send(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "huge.png").write_bytes(PNG + b"\x00" * MAX_IMAGE_BYTES)

            result = await execute_tool("read_file", {"path": "huge.png"}, cwd)

            self.assertEqual(result.status, "failed")
            self.assertIn("5 MB", result.text)
            self.assertEqual(result.images, ())


class ImageTurnTest(unittest.IsolatedAsyncioTestCase):
    def _open(self, cwd: Path, model: FunctionModel) -> Agent:
        session = make_session(cwd)
        session.append_system_prompt("snapshot")
        patcher = patch("paimon.agent.build_model", return_value=model)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Agent.open(cwd=cwd, session=session, config=Config(model="test:stub"), mode="yolo")

    async def test_a_mentioned_image_reaches_the_model_and_survives_a_resume(self) -> None:
        seen: list = []

        async def stream(messages, info):
            seen.append(_images(messages))
            yield "a cat"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)
            agent = self._open(cwd, FunctionModel(stream_function=stream))
            [event async for event in agent.run("what is @shot.png")]
            agent.session.unlock()

            self.assertEqual([(image.data, image.media_type) for image in seen[0]],
                             [(PNG, "image/png")])
            restored = Session(agent.session.path, "session-id", cwd).messages()
            self.assertEqual([image.data for image in _images(restored)], [PNG])
            # A replay shows the words, never the bytes.
            shown = [event.text for event in replay_events(restored) if isinstance(event, UserInput)]
            self.assertEqual(len(shown), 1)
            self.assertIn("<mentioned_image", shown[0])
            self.assertNotIn("BinaryContent", shown[0])
            self.assertIn("<mentioned_image", agent.session.first_user_text())

    async def test_an_image_the_model_reads_comes_back_in_the_tool_result(self) -> None:
        seen: list = []

        async def stream(messages, info):
            seen.append(_images(messages))
            if len(seen) == 1:
                yield {0: DeltaToolCall(name="read_file", json_args='{"path": "shot.png"}',
                                        tool_call_id="call-1")}
            else:
                yield "a cat"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)
            agent = self._open(cwd, FunctionModel(stream_function=stream))
            events = [event async for event in agent.run("look at shot.png")]

            self.assertEqual([image.data for image in seen[1]], [PNG])
            ended = [event for event in events if isinstance(event, ToolEnd)]
            self.assertIn("image/png", ended[0].result)
            replayed = [event for event in replay_events(agent.history) if isinstance(event, ToolEnd)]
            self.assertEqual(replayed[0].result, ended[0].result)

    async def test_a_model_whose_profile_takes_no_images_gets_words_instead(self) -> None:
        seen: list = []

        async def stream(messages, info):
            seen.append(messages)
            yield "ok"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)
            agent = self._open(cwd, FunctionModel(stream_function=stream,
                                                  profile={"supports_image_input": False}))
            [event async for event in agent.run("what is @shot.png")]

            self.assertEqual(_images(seen[0]), [])
            self.assertIn(UNSEEN, seen[0][-1].parts[0].content)
            # The history keeps the image for a model that can see it.
            self.assertEqual(len(_images(agent.history)), 1)

    async def test_a_model_that_refuses_images_is_asked_again_without_them(self) -> None:
        seen: list = []

        async def stream(messages, info):
            seen.append(len(_images(messages)))
            if _images(messages):
                raise ModelHTTPError(400, "stub", {"message": "image input is not supported"})
            yield "ok"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "shot.png").write_bytes(PNG)
            agent = self._open(cwd, FunctionModel(stream_function=stream))
            first = [event async for event in agent.run("what is @shot.png")]
            second = [event async for event in agent.run("and now?")]

            self.assertEqual(seen, [1, 0, 0])
            self.assertEqual(len([e for e in first if isinstance(e, ImagesRejected)]), 1)
            self.assertFalse(any(isinstance(e, ImagesRejected) for e in second))

    async def test_a_refusal_with_no_images_in_play_still_ends_the_turn(self) -> None:
        async def stream(messages, info):
            raise ModelHTTPError(400, "stub", {"message": "bad request"})
            yield "never"

        with tempfile.TemporaryDirectory() as directory:
            agent = self._open(Path(directory), FunctionModel(stream_function=stream))
            with self.assertRaises(ModelHTTPError):
                [event async for event in agent.run("hi")]


class RejectionTest(unittest.TestCase):
    def test_only_a_refusal_of_the_content_counts(self) -> None:
        for status in (400, 404, 415, 422):
            self.assertTrue(retry.is_rejection(ModelHTTPError(status, "m")), status)
        for status in (401, 403, 429, 500):
            self.assertFalse(retry.is_rejection(ModelHTTPError(status, "m")), status)
        self.assertFalse(retry.is_rejection(
            ModelHTTPError(400, "m", {"message": "prompt is too long"})))
        self.assertFalse(retry.is_rejection(asyncio.TimeoutError()))


class ImageAccountingTest(unittest.TestCase):
    def _messages(self, data: bytes) -> list:
        image = BinaryContent(data=data, media_type="image/png")
        return [
            ModelRequest(parts=[UserPromptPart(content=["look", image])]),
            ModelResponse(parts=[ToolCallPart(tool_name="read_file", args={"path": "a.png"},
                                              tool_call_id="c1")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="read_file", content=["an image", image],
                                               tool_call_id="c1")]),
        ]

    def test_an_image_is_counted_by_a_flat_rate_not_its_bytes(self) -> None:
        small = compaction.count_tokens(self._messages(PNG))
        large = compaction.count_tokens(self._messages(PNG + b"\x00" * 2_000_000))

        self.assertEqual(small, large)
        self.assertLess(large, 5_000)
        self.assertGreater(large, 2_000)

    def test_a_summary_request_carries_no_image_bytes(self) -> None:
        serialized = compaction._serialize_messages(self._messages(PNG + b"\x00" * 30_000))

        self.assertLess(len(serialized), 2_000)
        self.assertEqual(serialized.count("[image]"), 2)
        for line in serialized.splitlines():
            json.loads(line)

    def test_the_words_of_dumped_content_leave_the_image_out(self) -> None:
        raw = compaction.ModelMessagesTypeAdapter.dump_python(self._messages(PNG), mode="json")

        self.assertEqual(content_text(raw[0]["parts"][0]["content"]), "look")
        self.assertEqual(content_text("plain"), "plain")
        self.assertEqual(content_text(None), "")
