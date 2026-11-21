"""Images inside messages: attaching them, and reading around them.

A message that carries an image has a list for its content, the text first
and the images after it. Everything that wants only the words (a replay, the
reviewer, a summary, the token estimate) goes through here, so none of it
meets the bytes.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model

from .tools.result import Image

# What one image is counted as in the context estimate. Providers charge by
# pixels, from a few hundred tokens to a few thousand; this is the middle.
IMAGE_TOKENS = 1_500

# Sent in an image's place to a model that cannot see it.
UNSEEN = "(an image was attached here, which this model cannot read)"


def with_images(text: str, images: Sequence[Image]) -> str | list[str | BinaryContent]:
    """The content of a message saying ``text`` and showing ``images``."""
    if not images:
        return text
    return [text, *(BinaryContent(data=image.data, media_type=image.media_type) for image in images)]


def _is_image(item: object) -> bool:
    """Whether ``item`` is an image, as an object or as a log record has it."""
    if isinstance(item, BinaryContent):
        return item.is_image
    return (isinstance(item, dict) and item.get("kind") == "binary"
            and str(item.get("media_type")).startswith("image/"))


def content_text(content: object) -> str:
    """The words of a message's content, live or as dumped to a log.

    The images are left out without a mark: the text beside them already
    names each one, by its mention tag or by the tool result's first line.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(item if isinstance(item, str) else str(item)
                       for item in content if not _is_image(item))
    return "" if content is None else str(content)


def replace_images(content: object, stand_in: object) -> object:
    """Dumped content with ``stand_in`` where each image was."""
    if isinstance(content, list):
        return [stand_in if _is_image(item) else item for item in content]
    return content


def has_images(messages: Sequence[ModelMessage]) -> bool:
    return any(
        isinstance(part, (UserPromptPart, ToolReturnPart)) and isinstance(part.content, list)
        and any(_is_image(item) for item in part.content)
        for message in messages if isinstance(message, ModelRequest) for part in message.parts)


def without_images(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """The messages with a line of text where each image was, on copies."""
    sent: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest) and has_images([message]):
            message = dataclasses.replace(message, parts=[
                dataclasses.replace(part, content=replace_images(part.content, UNSEEN))
                if isinstance(part, (UserPromptPart, ToolReturnPart)) else part
                for part in message.parts])
        sent.append(message)
    return sent


def takes_images(model: Model) -> bool:
    """Whether the model's profile lets images through. A profile that does
    is no promise: nothing says otherwise is all it means."""
    return bool(model.profile.get("supports_image_input", True))
