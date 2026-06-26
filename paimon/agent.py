"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable, Optional

import litellm

from . import config, tools

litellm.telemetry = False
litellm.suppress_debug_info = True


# ---- Events yielded by Agent.run -------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ToolStart:
    id: str
    name: str
    args: dict


@dataclass
class ToolEnd:
    id: str
    name: str
    result: str
    denied: bool = False


@dataclass
class TurnEnd:
    pass


# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]


def _system_prompt(cwd: Path) -> str:
    return f"""You are Paimon, a concise coding assistant operating in a terminal.

You help with software engineering tasks by reading and editing files and running
shell commands. You have these tools: read_file, write_file, edit_file, bash.

Guidelines:
- Working directory: {cwd}
- Today's date: {date.today().isoformat()}
- Prefer reading a file before editing it. For edits, use edit_file with a unique
  old_string; only use write_file for new files or full rewrites.
- Use the bash tool for listing, searching (grep/find/ls), git, and running tests.
- Be direct. When the task is done, briefly state what you did. Don't narrate every step.
"""


class Agent:
    def __init__(self, cwd: Optional[Path] = None, confirm: Optional[ConfirmFn] = None):
        self.cwd = Path(cwd or Path.cwd())
        self.confirm = confirm
        self.messages: list[dict] = [
            {"role": "system", "content": _system_prompt(self.cwd)}
        ]

    async def run(self, user_input: str) -> AsyncIterator[object]:
        """Run one user turn to completion, yielding events along the way."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = await litellm.acompletion(
                model=config.MODEL,
                api_base=config.API_BASE,
                api_key=config.API_KEY,
                messages=self.messages,
                tools=tools.TOOLS,
                stream=True,
            )

            content = ""
            # index -> {"id", "name", "args"} accumulated across stream deltas
            calls: dict[int, dict] = {}

            async for chunk in response:
                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield ReasoningDelta(reasoning)

                if delta.content:
                    content += delta.content
                    yield TextDelta(delta.content)

                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments

            ordered = [calls[i] for i in sorted(calls)]

            assistant_msg: dict = {"role": "assistant", "content": content or None}
            if ordered:
                assistant_msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["args"]},
                    }
                    for c in ordered
                ]
            self.messages.append(assistant_msg)

            if not ordered:
                yield TurnEnd()
                return

            for c in ordered:
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = c["name"]
                yield ToolStart(c["id"], name, args)

                denied = False
                if self.confirm and name in tools.DANGEROUS:
                    allowed = await self.confirm(name, args)
                    if not allowed:
                        denied = True
                        result = "User denied this operation."
                if not denied:
                    result = await tools.execute_tool(name, args, self.cwd)

                yield ToolEnd(c["id"], name, result, denied=denied)
                self.messages.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": result}
                )
            # loop again so the model can react to tool results
