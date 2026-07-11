"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import asyncio
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
from typing import AsyncIterator, Awaitable, Callable, Optional

import litellm

from . import config, tools
from .session import Session

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
class TodosUpdate:
    todos: list[dict]


@dataclass
class TurnEnd:
    pass


# A confirm callback returns True to allow a dangerous tool, False to deny.
ConfirmFn = Callable[[str, dict], Awaitable[bool]]


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
                 session: Optional[Session] = None):
        self.cwd = Path(cwd or Path.cwd())
        self.confirm = confirm
        self.todos: list[dict] = []
        self.session = session or Session.create(self.cwd)
        self.messages: list[dict] = [{"role": "system", "content": _system_prompt(self.cwd)}]
        self.messages.extend(self.session.messages())

    def _append_message(self, message: dict) -> None:
        self.messages.append(message)
        self.session.append_message(message)

    async def run(self, user_input: str) -> AsyncIterator[object]:
        """Run one user turn to completion, yielding events along the way."""
        self._append_message({"role": "user", "content": user_input})

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

            try:
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
            except asyncio.CancelledError:
                # Interrupted mid-stream: keep partial text (drop incomplete tool
                # calls) so the message history stays valid for the next request.
                self._append_message({"role": "assistant", "content": content or "(interrupted)"})
                raise

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
            self._append_message(assistant_msg)

            if not ordered:
                yield TurnEnd()
                return

            # Pre-seed a tool result for every call up front, so even if we are
            # interrupted mid-execution the history never has a dangling tool_call
            # (the API requires each tool_call_id to be answered). Unfinished ones
            # keep this placeholder.
            tool_msgs = [
                {"role": "tool", "tool_call_id": c["id"], "content": "Interrupted by user."}
                for c in ordered
            ]
            self.messages.extend(tool_msgs)
            persisted_tool_ids = [self.session.append_message(message) for message in tool_msgs]

            for slot, c, persisted_id in zip(tool_msgs, ordered, persisted_tool_ids):
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = c["name"]
                yield ToolStart(c["id"], name, args)

                # write_todos mutates agent-held state rather than the filesystem,
                # so it is handled here instead of in the stateless execute_tool.
                if name == "write_todos":
                    self.todos = args.get("todos") or []
                    result = tools.render_todos(self.todos)
                    slot["content"] = result
                    self.session.append_message(slot, replaces=persisted_id)
                    yield TodosUpdate(list(self.todos))
                    yield ToolEnd(c["id"], name, result)
                    continue

                denied = False
                if self.confirm and name in tools.DANGEROUS:
                    allowed = await self.confirm(name, args)
                    if not allowed:
                        denied = True
                        result = "User denied this operation."
                if not denied:
                    result = await tools.execute_tool(name, args, self.cwd)

                slot["content"] = result
                self.session.append_message(slot, replaces=persisted_id)
                yield ToolEnd(c["id"], name, result, denied=denied)
            # loop again so the model can react to tool results
