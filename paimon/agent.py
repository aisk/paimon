"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import asyncio
import dataclasses
import json
import locale
import ntpath
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import AsyncIterator, Optional

from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters

from . import compaction, tools
from .config import Config
from .llm import build_model
from .mentions import MentionExpander
from .session import Session


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
class TodosUpdate:
    todos: list[dict]


@dataclass
class TurnEnd:
    pass


@dataclass
class ContextCompacted:
    tokens_before: int
    tokens_after: int


@dataclass
class ContextCompactionFailed:
    error: str


# ---- Replay-only events (history has no deltas, so these mark structure) ----


@dataclass
class UserInput:
    text: str


@dataclass
class CompactionNotice:
    """A compaction checkpoint encountered while replaying history."""


def _parse_args(args: object) -> dict:
    """Tool-call arguments as a dict, tolerating malformed JSON from the model."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def replay_events(messages: list[ModelMessage]) -> list[object]:
    """Persisted messages replayed as the events a live ``Agent.run`` yields.

    Lets a UI render resumed history through the same code path as live turns.
    """
    events: list[object] = []
    for message in messages:
        if compaction.is_summary_message(message):
            events.append(CompactionNotice())
            continue
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content:
                    events.append(UserInput(part.content))
                elif isinstance(part, ToolReturnPart) and part.tool_name != "write_todos":
                    events.append(ToolEnd(part.tool_call_id, part.tool_name,
                                          str(part.content or "(no output)")))
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    events.append(ReasoningDelta(part.content))
                elif isinstance(part, TextPart) and part.content:
                    events.append(TextDelta(part.content))
                elif isinstance(part, ToolCallPart):
                    args = _parse_args(part.args)
                    if part.tool_name == "write_todos":
                        events.append(TodosUpdate(args.get("todos") or []))
                    else:
                        events.append(ToolStart(part.tool_call_id, part.tool_name, args))
    return events


def _strip_foreign_thinking(history: list[ModelMessage], model: Model) -> list[ModelMessage]:
    """Drop thinking parts produced by a different provider/model.

    Preserved thinking is only valid replayed verbatim to the model that
    produced it; other endpoints may reject or misread it (cf. pi's
    transformMessages: same model keeps thinking, a changed model strips it).
    """
    current = (model.system, model.model_name)
    sanitized: list[ModelMessage] = []
    for message in history:
        if (isinstance(message, ModelResponse)
                and (message.provider_name, message.model_name) != current
                and any(isinstance(part, ThinkingPart) for part in message.parts)):
            message = dataclasses.replace(
                message, parts=[part for part in message.parts if not isinstance(part, ThinkingPart)]
            )
        sanitized.append(message)
    return sanitized


def _user_texts(history: list[ModelMessage]) -> list[str]:
    """All plain-text user prompt contents, for mention-version restoration."""
    return [part.content
            for message in history if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)]


# Re-exported so UI code can keep importing it from here.
ConfirmFn = tools.ConfirmFn


CONTEXT_FILE = "AGENTS.md"

_VERSION_COMMANDS = {
    "rg": ("--version",),
    "jq": ("--version",),
    "fd": ("--version",),
    "curl": ("--version",),
    "wget": ("--version",),
    "unzip": ("-v",),
    "tree": ("--version",),
    "uv": ("--version",),
    "npm": ("--version",),
    "python": ("--version",),
    "node": ("--version",),
    "perl": ("-v",),
    "ruby": ("--version",),
    "git": ("--version",),
}

_WINDOWS_VERSION_COMMANDS = {
    "pwsh": ("-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"),
    "powershell": ("-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"),
    "dotnet": ("--version",),
    "py": ("--version",),
}

def _command_version(executable: str, args: tuple[str, ...] = ("--version",)) -> str:
    """Return a compact version line without letting probes delay startup."""
    try:
        result = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "version unknown"
    output = result.stdout or result.stderr
    return next((line.strip()[:160] for line in output.splitlines() if line.strip()), "version unknown")


def _terminal_description() -> str:
    if os.environ.get("WT_SESSION"):
        host = "Windows Terminal"
    elif os.environ.get("TERM_PROGRAM"):
        host = os.environ["TERM_PROGRAM"]
    elif os.environ.get("VSCODE_INJECTION") or os.environ.get("VSCODE_PID"):
        host = "Visual Studio Code"
    elif os.environ.get("ConEmuANSI"):
        host = "ConEmu"
    else:
        host = "unknown"
    details = [f"host={host}", f"TERM={os.environ.get('TERM') or 'unknown'}"]
    if os.environ.get("COLORTERM"):
        details.append(f"COLORTERM={os.environ['COLORTERM']}")
    return ", ".join(details)


def _shell_description(system: str) -> str:
    if system == "Windows":
        # Windows has no SHELL convention. PowerShell-specific environment
        # variables are the best portable hint without adding a dependency.
        powershell_hint = os.environ.get("POWERSHELL_DISTRIBUTION_CHANNEL") or os.environ.get("PSModulePath")
        candidates = ("pwsh", "powershell") if powershell_hint else ()
        executable = next((path for name in candidates if (path := shutil.which(name))), None)
        if executable:
            name = ntpath.splitext(ntpath.basename(executable))[0].lower()
            args = _WINDOWS_VERSION_COMMANDS[name]
            return f"{executable} ({_command_version(executable, args)}; inferred from environment)"
        comspec = os.environ.get("ComSpec") or shutil.which("cmd")
        if comspec:
            return f"{comspec} ({_command_version(comspec, ('/d', '/c', 'ver'))}; inferred from environment)"
        return "unknown"

    shell_path = os.environ.get("SHELL") or "unknown"
    executable = shutil.which(shell_path) if shell_path != "unknown" else None
    version = _command_version(executable) if executable else "unknown"
    return f"{shell_path} ({version})"


def _runtime_flags(system: str) -> str:
    flags = []
    if Path("/.dockerenv").exists() or os.environ.get("container"):
        flags.append("container")
    if os.environ.get("CI"):
        flags.append("CI")
    return ", ".join(flags) or ("native Windows" if system == "Windows" else "native")


def _environment_context() -> str:
    """Describe the host and useful installed CLIs for the model."""
    system = platform.system()
    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME", system)
    except OSError:
        os_name = platform.platform()

    lines = [
        f"Operating system: {os_name}",
        f"Kernel: {system} {platform.release()}",
        f"CPU architecture: {platform.machine()}",
        f"Runtime: {_runtime_flags(system)}",
        f"Locale/encoding: {locale.getlocale()[0] or 'unknown'} / {locale.getpreferredencoding(False)}",
        f"Terminal: {_terminal_description()}",
        f"Shell: {_shell_description(system)}",
        "Available command-line tools:",
    ]
    aliases = {"fd": ("fd", "fdfind"), "python": ("python", "python3")}
    version_commands = dict(_VERSION_COMMANDS)
    if system == "Windows":
        version_commands.update(_WINDOWS_VERSION_COMMANDS)
        aliases["python"] = ("python", "py")
    for name, version_args in version_commands.items():
        candidates = aliases.get(name, (name,))
        executable = next((path for candidate in candidates if (path := shutil.which(candidate))), None)
        if executable:
            executable_name = (
                ntpath.splitext(ntpath.basename(executable))[0] if system == "Windows" else Path(executable).name
            )
            displayed_name = ntpath.basename(executable) if system == "Windows" else Path(executable).name
            alias = "" if executable_name == name else f" (executable: {displayed_name})"
            lines.append(f"- {name}: {_command_version(executable, version_args)}{alias}")
        else:
            lines.append(f"- {name}: not installed")
    return "\n".join(lines)


def _load_context_files(cwd: Path) -> list[tuple[Path, str]]:
    """Find AGENTS.md from cwd up to the filesystem root.

    Returned root-first so the file closest to cwd comes last in the prompt
    (later instructions take precedence), matching pi's behaviour.
    """
    found: list[tuple[Path, str]] = []
    current = cwd.resolve()
    while True:
        candidate = current / CONTEXT_FILE
        if candidate.is_file():
            try:
                found.append((candidate, candidate.read_text(errors="replace")))
            except OSError:
                pass
        if current == current.parent:
            break
        current = current.parent
    found.reverse()
    return found


def _system_prompt(cwd: Path) -> str:
    prompt = """You are Paimon, a concise coding assistant operating in a terminal.

You help with software engineering tasks by reading and editing files and running
shell commands. You have these tools: read_file, write_file, edit_file, glob, bash,
write_todos.

Guidelines:
- Prefer reading a file before editing it. For edits, use edit_file with a unique
  old_string; only use write_file for new files or full rewrites.
- Use glob to find files by name pattern; use the bash tool for content search
  (grep), git, and running tests.
- For tasks with several steps, call write_todos first to lay out a plan, then keep
  it updated as you go (one task in_progress at a time). Skip it for simple tasks.
- User @path mentions are expanded into <mentioned_file> tags. A tag with
  exposure="full" contains the complete file version. exposure="preview" contains
  only its beginning; use read_file for omitted or exact content. A tag with
  status="previously_mentioned" repeats no content but identifies a version already
  present in this active context. A tag with another status is an attachment error.
- Be direct. When the task is done, briefly state what you did. Don't narrate every step."""

    context_files = _load_context_files(cwd)
    if context_files:
        prompt += "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
        for path, content in context_files:
            prompt += f'<project_instructions path="{path}">\n{content}\n</project_instructions>\n\n'
        prompt += "</project_context>"

    prompt += "\n\n<environment>"
    prompt += f"\nCurrent date: {date.today().isoformat()}"
    prompt += f"\nCurrent working directory: {cwd}"
    prompt += f"\n{_environment_context()}"
    prompt += "\n</environment>"
    return prompt


class Agent:
    def __init__(self, cwd: Optional[Path] = None, confirm: Optional[ConfirmFn] = None,
                 session: Optional[Session] = None, mode: str = "read",
                 config: Optional[Config] = None):
        self.cwd = Path(cwd or Path.cwd())
        self.confirm = confirm
        self.mode = mode
        self.config = config or Config.load()
        self.todos: list[dict] = []
        if session is None:
            self.session = Session.create(self.cwd)
            system_prompt = _system_prompt(self.cwd)
            self.session.append_system_prompt(system_prompt)
        else:
            self.session = session
            system_prompt = self.session.system_prompt()
            if system_prompt is None:
                raise RuntimeError("Session does not contain a persisted system prompt")
        self.system_prompt = system_prompt
        self.history: list[ModelMessage] = self.session.messages()
        self.mentions = MentionExpander(self.cwd, _user_texts(self.history))
        self._cached_model: Optional[tuple[tuple, Model]] = None

    def _model(self) -> Model:
        """The configured model, rebuilt when login changes the config."""
        if not self.config.model:
            raise RuntimeError("No model configured; log in first")
        key = (self.config.model, self.config.api_base, self.config.api_key)
        if self._cached_model is None or self._cached_model[0] != key:
            self._cached_model = (key, build_model(*key))
        return self._cached_model[1]

    # The two methods below are the only paths that write conversation state.
    # They keep the invariant that ``self.history`` always equals what
    # replaying the session log would produce.

    def _append_message(self, message: ModelMessage) -> str:
        self.history.append(message)
        return self.session.append_message(message)

    def _replace_message(self, record_id: str, message: ModelMessage) -> None:
        """Persist the final version of a message already present in ``self.history``."""
        self.session.append_message(message, replaces=record_id)

    async def _maybe_compact(self) -> Optional[compaction.CompactionResult]:
        if not self.config.compaction_enabled:
            return None

        window = compaction.context_window(self.config.compaction_context_window)
        tokens_before = compaction.count_tokens(self.history, tools.TOOLS)
        if not compaction.should_compact(tokens_before, window, self.config.compaction_reserve_tokens):
            return None

        result = await compaction.compact(
            self.history,
            model=self._model(),
            keep_recent_tokens=self.config.compaction_keep_recent_tokens,
            tokens_before=tokens_before,
            tool_schemas=tools.TOOLS,
        )
        if result is None:
            return None

        # Mirrors how Session.messages() replays a compaction record, so the
        # append-message invariant (see _append_message) still holds afterwards.
        self.session.append_compaction(result.summary, result.kept_messages, result.tokens_before)
        self.history = [compaction.summary_message(result.summary), *result.kept_messages]
        # Mentions may only reference file bodies that remain in the effective
        # prompt. A compacted prefix is no longer available to the model.
        self.mentions = MentionExpander(self.cwd, _user_texts(self.history))
        result.tokens_after = compaction.count_tokens(self.history, tools.TOOLS)
        return result

    async def run(self, user_input: str) -> AsyncIterator[object]:
        """Run one user turn to completion, yielding events along the way."""
        self._append_message(ModelRequest(parts=[UserPromptPart(content=self.mentions.expand(user_input))]))
        compaction_failed = False

        while True:
            if not compaction_failed:
                try:
                    compacted = await self._maybe_compact()
                except Exception as exc:  # noqa: BLE001 - the normal request may still fit
                    compaction_failed = True
                    yield ContextCompactionFailed(str(exc))
                else:
                    if compacted:
                        yield ContextCompacted(compacted.tokens_before, compacted.tokens_after)

            model = self._model()
            request_messages: list[ModelMessage] = [
                ModelRequest(parts=[SystemPromptPart(content=self.system_prompt)]),
                *_strip_foreign_thinking(self.history, model),
            ]
            parameters = ModelRequestParameters(
                function_tools=tools.TOOL_DEFINITIONS, allow_text_output=True
            )

            content = ""  # accumulated text, kept if the stream is interrupted

            try:
                async with model_request_stream(
                    model, request_messages, model_request_parameters=parameters
                ) as stream:
                    async for event in stream:
                        if isinstance(event, PartStartEvent):
                            part = event.part
                            if isinstance(part, ThinkingPart) and part.content:
                                yield ReasoningDelta(part.content)
                            elif isinstance(part, TextPart) and part.content:
                                content += part.content
                                yield TextDelta(part.content)
                        elif isinstance(event, PartDeltaEvent):
                            delta = event.delta
                            if isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                                yield ReasoningDelta(delta.content_delta)
                            elif isinstance(delta, TextPartDelta) and delta.content_delta:
                                content += delta.content_delta
                                yield TextDelta(delta.content_delta)
                    response = stream.get()
            except asyncio.CancelledError:
                # Interrupted mid-stream: keep partial text but drop incomplete
                # tool calls and thinking (Z.ai-style preserved thinking must be
                # replayed complete or not at all) so the history stays valid.
                self._append_message(ModelResponse(
                    parts=[TextPart(content=content or "(interrupted)")],
                    model_name=model.model_name,
                    provider_name=model.system,
                ))
                raise

            self._append_message(response)

            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            if not calls:
                yield TurnEnd()
                return

            # Pre-seed a tool result for every call up front, so even if we are
            # interrupted mid-execution the history never has a dangling tool call
            # (the API requires each tool_call_id to be answered). Unfinished ones
            # keep this placeholder.
            returns = [
                ToolReturnPart(tool_name=call.tool_name, content="Interrupted by user.",
                               tool_call_id=call.tool_call_id)
                for call in calls
            ]
            tool_request = ModelRequest(parts=returns)
            record_id = self._append_message(tool_request)

            for slot, call in zip(returns, calls):
                args = _parse_args(call.args)
                name = call.tool_name
                yield ToolStart(call.tool_call_id, name, args)

                # write_todos mutates agent-held state rather than the filesystem,
                # so it is handled here instead of in the stateless execute_tool.
                if name == "write_todos":
                    self.todos = args.get("todos") or []
                    result = tools.render_todos(self.todos)
                    slot.content = result
                    self._replace_message(record_id, tool_request)
                    yield TodosUpdate(list(self.todos))
                    yield ToolEnd(call.tool_call_id, name, result)
                    continue

                result, denied = await tools.run_tool(name, args, self.cwd, self.mode, self.confirm)

                slot.content = result
                self._replace_message(record_id, tool_request)
                yield ToolEnd(call.tool_call_id, name, result, denied=denied)
            # loop again so the model can react to tool results
