"""Code mode: the run_code tool, a sandbox whose only way out is the agent's tools.

The model writes a Python script; it runs in Monty, a Python subset with no
filesystem, network, environment or clock, and calls the agent's other tools
as async functions. Only what the script prints and the value of its last
expression go back to the model, so the results of the calls it makes never
enter the conversation.

This module is the sandbox and what the model is told about it. The calls
themselves are the agent's: it passes in the function that gates and runs one.
"""

import asyncio
import dataclasses
import json
import time
import types
from typing import Annotated, Awaitable, Callable, Literal, Union, get_args, get_origin, get_type_hints

from typing_extensions import NotRequired, Required, is_typeddict

from . import tools

NAME = "run_code"

# Runs one nested call: (tool name, arguments) to the tool's result. Raises
# ToolCallError for a call that was not run.
CallFn = Callable[[str, dict], Awaitable[str]]

# Both clocks behind the duration limit stop while the script waits on a tool,
# so this bounds a runaway loop without cutting a long shell command short.
_LIMITS = {"max_memory": 256 * 1024 * 1024, "max_feed_duration_secs": 30.0}
_MAX_ERROR = 4_000  # chars of a traceback kept; the output gets the rest
_MAX_DEPTH = 50  # containers a rendered value is followed into
_SCALARS = (str, int, float, bool, type(None))

_INTRO = """\
Run a Python script that calls your other tools, and get back only what it \
prints plus the value of its last expression: what the calls return never \
enters the conversation unless the script outputs it.

The sandbox runs Monty, a subset of Python:
- No filesystem, network, environment or clock, and no third-party packages. \
Importable: asyncio, json, re, math, datetime, typing, unicodedata, random.
- Its only way out is the tools named below, each an async function taking the \
tool's own arguments as keywords and returning its result as a str: \
`text = await read_file(path="a.py")`. Run independent calls together with \
`await asyncio.gather(f(...), g(...))`; there is no create_task or wait.
- A call that is denied, invalid or over the tool budget raises an Exception \
carrying the reason.
- The script is type-checked against the tools' arguments before it runs. \
Nothing is kept between scripts, and calls already made are not undone when \
one fails.

Callable: """


class ToolCallError(Exception):
    """A nested tool call that was not run: denied, invalid, or over budget."""


def callable_tools(toolset: dict[str, tools.Tool]) -> dict[str, tools.Tool]:
    """What a script may call: every tool with an executor of its own.

    The ones the agent loop runs itself (the todo list, the question, the job
    tools) stay direct-only, and run_code is one of those.
    """
    return {name: tool for name, tool in toolset.items() if tool.run is not None}


def _annotation(tp: object) -> str:
    origin = get_origin(tp)
    if origin in (Annotated, NotRequired, Required):
        return _annotation(get_args(tp)[0])
    if origin is Literal:
        return f"Literal[{', '.join(repr(arg) for arg in get_args(tp))}]"
    if origin in (Union, types.UnionType):
        return " | ".join(_annotation(arg) for arg in get_args(tp))
    if tp is type(None):
        return "None"
    if origin is None:
        return "dict" if is_typeddict(tp) else getattr(tp, "__name__", "Any")
    name = getattr(origin, "__name__", None)
    return f"{name}[{', '.join(_annotation(arg) for arg in get_args(tp))}]" if name else "Any"


def _signature(name: str, tool: tools.Tool) -> str:
    """One stub line for a tool, generated from the TypedDict of its arguments."""
    optional = tool.params.__optional_keys__
    params = [f"{key}: {_annotation(tp)}{' = ...' if key in optional else ''}"
              for key, tp in get_type_hints(tool.params, include_extras=True).items()]
    return f"async def {name}({'*, ' + ', '.join(params) if params else ''}) -> str: ..."


def stubs(toolset: dict[str, tools.Tool]) -> str:
    """The signatures a script is type-checked against."""
    lines = [_signature(name, tool) for name, tool in callable_tools(toolset).items()]
    return "\n".join(["from typing import Any, Literal", *lines]) + "\n"


def tool(toolset: dict[str, tools.Tool]) -> tools.Tool:
    """run_code as an agent holding ``toolset`` is shown it.

    Built per agent, like the schemas: the description names exactly the
    tools that agent's scripts can call. Their arguments are not repeated,
    since every one of them is declared to the model beside run_code.
    """
    return dataclasses.replace(toolset[NAME],
                               description=_INTRO + ", ".join(callable_tools(toolset)))


class _Full(Exception):
    """The rendering has all the characters it may have."""


def _render(value: object, limit: int = tools.MAX_OUTPUT) -> tuple[str, bool]:
    """A script's last value as text, and whether it was cut short.

    A str is returned as it is, anything else as JSON.

    Written piece by piece and abandoned once past ``limit``, never built
    whole and cut afterwards. The value is the script's, and a few lines can
    make one that is small in the sandbox, where a list repeated a million
    times is one list, and enormous written out.
    """
    if isinstance(value, str):
        return value, False
    pieces: list[str] = []
    size = 0

    def emit(text: str) -> None:
        nonlocal size
        pieces.append(text)
        size += len(text)
        if size > limit:
            raise _Full

    def leaf(item: object) -> str:
        if isinstance(item, (str, bytes)):
            item = item[:limit]
        return json.dumps(item, ensure_ascii=False, default=str)

    def walk(item: object, depth: int) -> None:
        if isinstance(item, dict):
            # A tuple key can be as deceptively small as any other value.
            entries = [(leaf(str(key) if isinstance(key, _SCALARS) else f"<{type(key).__name__}>") + ": ",
                        entry) for key, entry in item.items()]
            brackets = "{}"
        elif isinstance(item, (list, tuple, set, frozenset)):
            entries = [("", entry) for entry in item]
            brackets = "[]"
        else:
            emit(leaf(item))
            return
        if not entries:
            emit(brackets)
        elif depth >= _MAX_DEPTH:
            emit(brackets[0] + "..." + brackets[1])
        else:
            for index, (key, entry) in enumerate(entries):
                emit((brackets[0] if index == 0 else ",") + "\n" + " " * (depth + 1) + key)
                walk(entry, depth + 1)
            emit("\n" + " " * depth + brackets[1])

    try:
        walk(value, 0)
    except _Full:
        return "".join(pieces), True
    return "".join(pieces), False


def _describe(exc: Exception) -> str:
    """A script failure as the model reads it, with line numbers where Monty has them."""
    display = getattr(exc, "display", None)
    text = (display() if callable(display) else f"{type(exc).__name__}: {exc}").rstrip()
    return text if len(text) <= _MAX_ERROR else text[:_MAX_ERROR] + "\n... (error truncated)"


async def run(code: str, toolset: dict[str, tools.Tool], call: CallFn) -> tuple[str, bool]:
    """Run one script. Returns ``(result for the model, completed)``.

    Nested calls run as tasks of their own, concurrently when the script
    gathers them, and none outlives the script: whatever is still running
    when it ends, fails or is interrupted is cancelled.
    """
    # Imported here: the sandbox is a native extension nobody pays for at
    # startup unless code mode is on and a script actually runs.
    import pydantic_monty as monty

    tasks: set[asyncio.Task] = set()

    def bind(name: str):
        async def function(**args):
            # Monty leaves a call the script never awaited running, and does
            # not cancel the others when it is cancelled itself, so each one
            # is a task this function can find again.
            task = asyncio.ensure_future(call(name, args))
            tasks.add(task)
            return await task
        return function

    printed = monty.CollectString()
    value, error = None, None
    started = time.perf_counter()
    try:
        async with monty.AsyncMonty() as pool, pool.checkout(
            script_name="script.py", limits=_LIMITS,
            type_check=True, type_check_stubs=stubs(toolset), type_check_format="concise",
            # Without a handler to call, this is what takes the clock away.
            os_policy={"datetime": "call_host"},
        ) as session:
            value = await session.feed_run(
                code, print_callback=printed,
                external_lookup={name: bind(name) for name in callable_tools(toolset)})
    except Exception as exc:  # noqa: BLE001 — a script failure is a result, not the turn's
        error = _describe(exc)
    except BaseException as exc:
        # A panic in the sandbox's own Rust code arrives as a BaseException
        # from a module that cannot be imported; anything else (cancellation
        # above all) is not this function's to swallow.
        if type(exc).__name__ != "PanicException":
            raise
        error = f"The sandbox crashed: {exc}"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - started
    output, clipped = printed.output, False
    if error is None and value is not None:
        rendered, clipped = _render(value)
        output += ("\n" if output and not output.endswith("\n") else "") + rendered
    output = output.rstrip("\n")
    header = f"Script failed after {elapsed:.2f}s" if error else f"Script completed in {elapsed:.2f}s"
    # The output gives way to the error, never the other way round: the
    # traceback is what the model needs to fix the script.
    room = max(0, tools.MAX_OUTPUT - len(header) - len(error or "") - 2)
    if len(output) > room:
        # How much more there was is unknown for a value abandoned midway.
        more = "" if clipped else f", {len(output) - room} more chars"
        output = output[:room] + f"\n... (truncated{more})"
    parts = [header, output or ("" if error else "(no output)"), error or ""]
    return "\n".join(part for part in parts if part), error is None
