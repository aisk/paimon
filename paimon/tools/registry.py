"""Tool definitions and execution.

Each tool is one ``Tool`` entry in ``REGISTRY``: its description and the
TypedDict of its arguments, which the schema sent to the model is generated
from, the function that runs it, and the access class that drives permission
gating. Adding a tool means adding one registry entry.
"""

import asyncio
import dataclasses
import inspect
from functools import cache
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Literal

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, with_config
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer
from pydantic_ai.tools import GenerateToolJsonSchema, ToolDefinition
from typing_extensions import NotRequired, TypedDict

from .. import history, websearch
from .base import MAX_OUTPUT, Tool, ToolContext
from .files import _edit_file, _glob, _read_file, _write_file
from .grep import _GREP_DEFAULT_RESULTS, _GREP_MAX_RESULTS, _grep_async
from .result import ToolError, ToolResult
from .shell import _shell


# An ask callback puts a question (and optional choices) to the user and
# returns their answer, or None when they dismissed it without answering.
AskFn = Callable[[str, list[str]], Awaitable[str | None]]


_TODO_MARKERS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def normalize_todos(value: object) -> list[dict] | None:
    """The ``todos`` argument as a task list, or None when it is malformed.

    The model writes this argument, so an unexpected shape has to become a
    tool error rather than an exception: every consumer (the result text, the
    TUI panel, the JSON stream) assumes a list of objects.
    """
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    return value


def render_todos(todos: list[dict]) -> str:
    """Plain-text rendering of the todo list, used as the tool result the model sees."""
    if not todos:
        return "(todo list cleared)"
    return "\n".join(f"{_TODO_MARKERS.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos)


@cache
def _adapter(params: type) -> TypeAdapter:
    return TypeAdapter(params)


def _untitled(schema: object) -> object:
    """``schema`` without the class names pydantic records as titles."""
    if isinstance(schema, list):
        return [_untitled(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    # A property that happens to be called "title" maps to a schema, never to
    # a string, so it is kept.
    return {key: _untitled(value) for key, value in schema.items()
            if not (key == "title" and isinstance(value, str))}


@cache
def _parameters(params: type) -> dict:
    """The JSON schema of a tool's arguments, as the model is shown it."""
    schema = _adapter(params).json_schema(schema_generator=GenerateToolJsonSchema)
    # Inlined, as providers differ in whether they follow a $ref.
    return _untitled(InlineDefsJsonSchemaTransformer(schema).walk())


def _argument_path(loc: tuple) -> str:
    path = ""
    for step in loc:
        path += f"[{step}]" if isinstance(step, int) else f".{step}" if path else str(step)
    return path


def validate_args(name: str, args: dict, toolset: dict) -> str | None:
    """Why ``args`` do not fit the tool's declared arguments, or None when they do.

    The one validation layer every ToolCallPart passes before gating or
    dispatch, agent-handled tools included, so a malformed argument becomes a
    tool error result instead of an exception ending the turn. Strict, so a
    value of the wrong type is reported rather than coerced: the executors
    read the arguments as the model wrote them. Unknown extra arguments pass:
    models add stray keys, and rejecting them costs more than ignoring them.
    """
    tool = toolset.get(name)
    if tool is None:
        return f"unknown tool {name!r}"
    try:
        _adapter(tool.params).validate_python(args, strict=True)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        *parent, last = error["loc"]
        if error["type"] == "missing":
            where = f" in '{_argument_path(tuple(parent))}'" if parent else ""
            return f"missing required argument '{last}'{where}"
        problem = error["msg"].replace("Input should be", "must be", 1)
        return f"'{_argument_path(error['loc'])}' {problem}"
    return None


# Self-truncation headroom: read_history cuts at record boundaries with a note
# naming the seq it stopped at, which execute_tool's generic chop cannot do.
_READ_BUDGET = MAX_OUTPUT - 1_000


def _search_history(args: dict, ctx: ToolContext) -> str:
    if ctx.session is None:
        raise ToolError("Error: no session log available")
    return history.search(ctx.session, args)


def _read_history(args: dict, ctx: ToolContext) -> str:
    if ctx.session is None:
        raise ToolError("Error: no session log available")
    return history.read(ctx.session, args, _READ_BUDGET)


async def execute_tool(name: str, args: dict, cwd: Path, mode: str = "yolo",
                       registry: dict[str, Tool] | None = None,
                       ctx: ToolContext | None = None) -> ToolResult:
    """Run a registered tool. Never raises for the tool's own failure: that
    comes back as a failed result, with text the model can act on."""
    tool = (REGISTRY if registry is None else registry).get(name)
    if tool is None or tool.run is None:
        return ToolResult(f"Error: unknown tool {name!r}", "failed")
    ctx = ToolContext() if ctx is None else ctx
    try:
        result = tool.run(args, cwd, mode, ctx)
        if inspect.isawaitable(result):
            result = await result
    except asyncio.CancelledError:
        raise
    except ToolError as exc:
        result = exc.result()
    except Exception as exc:  # noqa: BLE001 — surface any tool error to the model
        return ToolResult(f"Error executing {name}: {exc}", "failed")
    if not isinstance(result, ToolResult):
        result = ToolResult(result)
    text = result.text
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n... (truncated, {len(text) - MAX_OUTPUT} more chars)"
        result = dataclasses.replace(result, text=text)
    return result


# What each tool takes, one TypedDict per tool: the model-facing parameter
# schema and the argument validation both come from it. A field's docstring is
# its description; NotRequired marks an optional one.
_spec = with_config(ConfigDict(use_attribute_docstrings=True))


@_spec
class Todo(TypedDict):
    content: str
    """Short description of the task."""
    status: Literal["pending", "in_progress", "completed"]


@_spec
class ReadFileArgs(TypedDict):
    path: str
    """File path, relative to the working directory or absolute."""
    offset: NotRequired[int]
    """1-indexed line to start from (optional)."""
    limit: NotRequired[int]
    """Maximum number of lines to read (optional)."""


@_spec
class WriteFileArgs(TypedDict):
    path: str
    content: str


@_spec
class EditFileArgs(TypedDict):
    path: str
    old_string: str
    """Exact text to replace (must be unique in the file)."""
    new_string: str
    """Replacement text."""


@_spec
class GlobArgs(TypedDict):
    pattern: str
    """Glob pattern. Use '**' to match any number of directories."""
    path: NotRequired[str]
    """Base directory to search in (optional, defaults to the working directory)."""
    include_ignored: NotRequired[bool]
    """Search inside noise dirs like node_modules/.venv/.git too (optional, default false)."""


@_spec
class GrepArgs(TypedDict):
    pattern: str
    """Python regular expression, matched against each LF/CRLF-delimited line."""
    path: NotRequired[str]
    """File or directory to search (optional, defaults to the working directory)."""
    glob: NotRequired[str]
    """Only search files whose name matches this pattern, e.g. '*.py' (optional)."""
    max_results: NotRequired[Annotated[int, Field(description=(
        f"Maximum matching lines to return (optional, default {_GREP_DEFAULT_RESULTS}, "
        f"maximum {_GREP_MAX_RESULTS})."))]]


@_spec
class WebSearchArgs(TypedDict):
    query: str
    """The search query."""
    max_results: NotRequired[Annotated[int, Field(description=(
        f"Maximum results to return (optional, default {websearch.DEFAULT_RESULTS}, "
        f"maximum {websearch.MAX_RESULTS})."))]]


@_spec
class ShellArgs(TypedDict):
    command: str


@_spec
class SearchHistoryArgs(TypedDict):
    query: str
    """Regular expression, matched case-insensitively. An invalid regex is searched as literal text."""
    max_results: NotRequired[int]
    """Maximum matches to return (optional, default 20, max 100)."""


@_spec
class ReadHistoryArgs(TypedDict):
    seq: int
    """Seq of the first record to read (from search_history results)."""
    count: NotRequired[int]
    """Number of consecutive records to read (optional, default 1, max 20)."""


@_spec
class WriteTodosArgs(TypedDict):
    todos: list[Todo]
    """The complete task list, in order."""


@_spec
class AskUserArgs(TypedDict):
    question: str
    """The question, self-contained and specific."""
    options: NotRequired[list[str]]
    """Choices to pick from, most likely first (optional, at most 9)."""


@_spec
class SpawnAgentArgs(TypedDict):
    prompt: str
    """The new agent's only user message; must be self-contained."""
    model: NotRequired[str]
    """Model for this agent only, as 'provider:name' from list_models (optional; defaults to the current one)."""


@_spec
class ListModelsArgs(TypedDict):
    pass


@_spec
class RunBackgroundArgs(TypedDict):
    command: str
    description: str
    """A few words naming the job; it labels the tab."""


@_spec
class ReadJobArgs(TypedDict):
    job_id: str
    """The id of a background command you started."""
    mode: NotRequired[Literal["new", "all"]]
    """'new' (default) since your last read, or 'all'."""


@_spec
class StopJobArgs(TypedDict):
    job_id: str
    """The id of an agent or a background command you started."""


@_spec
class RunCodeArgs(TypedDict):
    code: str
    """The Python script to run."""


REGISTRY: dict[str, Tool] = {
    "read_file": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _read_file(args, cwd),
        description="Read a text file and return its contents with line numbers.",
        params=ReadFileArgs,
    ),
    "write_file": Tool(
        access="write",
        run=lambda args, cwd, mode, ctx: _write_file(args, cwd),
        description="Create or overwrite a file with the given content.",
        params=WriteFileArgs,
    ),
    "edit_file": Tool(
        access="write",
        run=lambda args, cwd, mode, ctx: _edit_file(args, cwd),
        description="Replace an exact substring in a file. old_string must appear exactly once.",
        params=EditFileArgs,
    ),
    "glob": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _glob(args, cwd, sandboxed=mode != "yolo"),
        description="Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts'). Returns matching paths sorted by most recently modified first.",
        params=GlobArgs,
    ),
    "grep": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _grep_async(args, cwd, sandboxed=mode != "yolo"),
        description=(
            "Search file contents for a regular expression and return matching "
            "lines as path:line:text, in file order. Directories are searched "
            "recursively in deterministic directory order, skipping VCS/dependency/build noise "
            "and binary files. Searches time out after 10 seconds. "
            "Prefer this over shell grep: it is never held for approval."
        ),
        params=GrepArgs,
    ),
    # Not gated in any mode: a search changes nothing on this machine. The
    # query does leave it, for whichever public search engines ddgs picks, so
    # --no-web-search takes the tool away for a run where that matters.
    "web_search": Tool(
        access="none",
        run=lambda args, cwd, mode, ctx: websearch.search(args),
        description=(
            "Search the web and return a numbered list of results, each with its "
            "title, URL and a short snippet. Use it for anything that may have "
            "changed since your training data: current library versions, API "
            "documentation, error messages. It returns snippets, not whole pages."
        ),
        params=WebSearchArgs,
    ),
    "shell": Tool(
        access="execute",
        run=lambda args, cwd, mode, ctx: _shell(args, cwd, ctx),
        description=(
            "Run a shell command in the working directory and return its combined "
            "stdout/stderr. Use this for git, running tests and anything the other "
            "tools do not cover. Output is truncated to the last 2000 lines or ~28KB, whichever "
            "comes first; when that happens the full output is written to a file and its "
            "path is included in the result, so you can read the earlier part back."
        ),
        params=ShellArgs,
    ),
    # "read" because these only read the agent's own session log, never a user
    # file: with no path argument the gate resolves to cwd and always allows,
    # which is right — everything in the log already went through this
    # conversation once.
    "search_history": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _search_history(args, ctx),
        description=(
            "Search this session's full history log on disk. The log keeps every "
            "user message, assistant reply, tool call and tool result of this "
            "session, including everything that context compaction has since "
            "summarized away — use it to recover exact details (file paths, "
            "command output, earlier decisions) instead of guessing from a "
            "compaction summary. Returns one line per matching part, prefixed "
            "with a seq number to pass to read_history for the full record."
        ),
        params=SearchHistoryArgs,
    ),
    "read_history": Tool(
        access="read",
        run=lambda args, cwd, mode, ctx: _read_history(args, ctx),
        description=(
            "Read full records from this session's history log by seq number, "
            "as returned by search_history. Use it to see the complete content "
            "behind a match: the whole user message, tool output or assistant "
            "reply, even when it predates a context compaction."
        ),
        params=ReadHistoryArgs,
    ),
    # Stateful: mutates agent-held state, so the agent loop runs it itself.
    "write_todos": Tool(
        run=None,
        description=(
            "Create or update the task list for a multi-step task. Always pass the COMPLETE list; "
            "it overwrites the previous one. Use it to plan work and show progress on tasks with 3+ "
            "steps; skip it for trivial single-step requests. Keep exactly one task in_progress at a time, "
            "and mark a task completed as soon as it is done."
        ),
        params=WriteTodosArgs,
    ),
    # Needs the user at the keyboard, so the agent loop puts it to the UI itself.
    "ask_user": Tool(
        run=None,
        description=(
            "Ask the user one question and wait for the answer. Use it only when "
            "different answers would lead to materially different work and neither "
            "the request nor the code settles it; otherwise make the routine call "
            "yourself and say what you assumed. Pass options when the answer is a "
            "choice; the user can always type something else instead. Ask one "
            "question per call."
        ),
        params=AskUserArgs,
    ),
    # The rest of the registry is stateful in the same way: each of them acts
    # on the jobs the calling agent started, which the agent itself holds, so
    # the loop runs them instead of this module.
    #
    # access is written out rather than left to default so the choice is on the
    # record: all of them but run_background are "none" because they touch
    # neither the filesystem nor a process of their own, and everything a
    # started agent goes on to do is gated under the mode it inherited. Adding
    # a tool here that does reach outside the process needs its own access
    # class, the way run_background has one.
    "spawn_agent": Tool(
        run=None,
        access="none",
        description=(
            "Start another agent working in parallel, in the same working "
            "directory and with the same tools, and return its id at once. Use it "
            "for work that is independent of what you are doing right now, and "
            "keep the number small: only a few may run at once. The prompt must "
            "be self-contained: state the goal, the key file paths, the decisions "
            "already made and what to report back, because the new agent has no "
            "memory of this "
            "conversation, cannot ask you anything and cannot be sent a follow-up. "
            "Do not wait for it: its final answer is delivered to you as a message "
            "when it finishes, clipped if long, so ask for a short report. It "
            "cannot start agents of its own."
        ),
        params=SpawnAgentArgs,
    ),
    "list_models": Tool(
        run=None,
        access="none",
        description=(
            "List the models spawn_agent can start an agent on, across every "
            "provider you are signed in to. Call it before giving an "
            "agent a model other than your own, for a second opinion from a "
            "different model or a cheaper one for simple work. The list is read "
            "fresh on each call."
        ),
        params=ListModelsArgs,
    ),
    # Starting a background command is the one job tool that reaches outside
    # the process, so it is the one with an access class of its own:
    # "background" rather than "execute" so the safe_command allowance can
    # never apply to it, and never left to default to "none", which would hand
    # the model an unconfirmed, untimed, turn-outliving way to run anything.
    "run_background": Tool(
        run=None,
        access="background",
        description=(
            "Start a long-running command in its own tab and return a job id, "
            "instead of waiting for it like the shell tool does. Use it for things "
            "that are meant to keep running \u2014 a dev server, a file watcher, a long "
            "build or test suite \u2014 and use shell for anything that finishes on its "
            "own within a couple of minutes. The command gets no terminal and no "
            "input, so it must be non-interactive; output may arrive in blocks "
            "rather than line by line, because a pipe is not a terminal. You are "
            "told when it exits; call read_job for its output and stop_job when "
            "you are done with it."
        ),
        params=RunBackgroundArgs,
    ),
    "read_job": Tool(
        run=None,
        access="none",
        description=(
            "Read the tail of what a background command you started has printed "
            "since you last read it (mode 'new', the default) or in total (mode "
            "'all'), with its state and, once it has stopped, its exit code."
        ),
        params=ReadJobArgs,
    ),
    "stop_job": Tool(
        run=None,
        access="none",
        description=(
            "Stop an agent or a background command you started. A stopped agent "
            "reports nothing back; a stopped command's output stays readable. "
            "Stop what you no longer need: an agent going the wrong way keeps "
            "spending tokens, and a background command keeps running."
        ),
        params=StopJobArgs,
    ),
    # Code mode. The loop runs it, because every call a script makes is gated
    # and counted as that agent's own. Held only by an agent whose config turns
    # code mode on, and that agent swaps this description for one listing the
    # functions its scripts can call (see paimon.codemode).
    "run_code": Tool(
        run=None,
        access="none",
        description="Run a Python script in a sandbox that can call your other tools.",
        params=RunCodeArgs,
    ),
}

# Tools that need a tab to show a process in. Headless has none, and runs under
# asyncio.run, which cancels everything still alive on the way out: a command
# started there would be a process group nobody ever kills.
BACKGROUND_TOOLS = ("run_background", "read_job")

# Tools that need a user at the keyboard. Headless has nobody to answer, so it
# leaves them out rather than offering the model a question it cannot ask.
INTERACTIVE_TOOLS = ("ask_user",)

# What a spawned agent must not be given: every job tool and the question.
#
# Every job tool, because depth stays 1: only the conversation the user is
# actually in starts agents and leaves processes running behind it, and an
# agent that can start nothing has nothing to read or stop either.
#
# The question, because a subagent works for the agent that started it, not
# for the user: what it cannot settle belongs in its report, where the parent
# can decide or ask on its behalf.
SUBAGENT_DENIED = (*INTERACTIVE_TOOLS, *BACKGROUND_TOOLS, "spawn_agent", "list_models", "stop_job")


def without(registry: dict[str, Tool], names) -> dict[str, Tool]:
    """A copy of ``registry`` with ``names`` removed.

    Narrowing the registry — rather than the agent-handled table, which is
    class-level and shared — is what actually disables a tool for one agent:
    the loop rejects any name outside its own toolset before dispatching.
    """
    return {name: tool for name, tool in registry.items() if name not in set(names)}


def schemas(registry: dict[str, Tool]) -> list[dict]:
    """The OpenAI-style schema list for a registry, in registry order."""
    return [
        {"type": "function",
         "function": {"name": name, "description": tool.description,
                      "parameters": _parameters(tool.params)}}
        for name, tool in registry.items()
    ]


def definitions(registry: dict[str, Tool]) -> list[ToolDefinition]:
    """The same schemas as pydantic-ai tool definitions."""
    return [
        ToolDefinition(name=name, description=tool.description,
                       parameters_json_schema=_parameters(tool.params))
        for name, tool in registry.items()
    ]
