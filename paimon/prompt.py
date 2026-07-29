"""The system prompt: instructions, project context files, host environment.

Built once when a session is created and then persisted with it, so a resumed
session keeps the prompt it was started with.
"""

import locale
import os
import platform
from datetime import date
from pathlib import Path

CONTEXT_FILE = "AGENTS.md"

INSTRUCTIONS = """You are Paimon, a concise coding assistant operating in a terminal.

You help with software engineering tasks by reading and editing files and running
shell commands. You have these tools: read_file, write_file, edit_file, glob, shell,
write_todos, start_new_session.

Guidelines:
- Prefer reading a file before editing it. For edits, use edit_file with a unique
  old_string; only use write_file for new files or full rewrites.
- Use glob to find files by name pattern; use the shell tool for content search
  (grep), git, and running tests.
- For tasks with several steps, call write_todos first to lay out a plan, then keep
  it updated as you go (one task in_progress at a time). Skip it for simple tasks.
- When the earlier conversation is mostly irrelevant to the next phase of work,
  call start_new_session with a self-contained handoff prompt instead of
  continuing in a bloated context.
- User @path mentions are expanded into <mentioned_file> tags. A tag with a body
  contains the complete file; a self-closing tag gives only the path, so call
  read_file when you need its contents.
- Be direct. When the task is done, briefly state what you did. Don't narrate every step."""


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
    # Windows has no SHELL convention; ComSpec is the closest equivalent.
    variable = "ComSpec" if system == "Windows" else "SHELL"
    return os.environ.get(variable) or "unknown"


def _runtime_flags(system: str) -> str:
    flags = []
    if Path("/.dockerenv").exists() or os.environ.get("container"):
        flags.append("container")
    if os.environ.get("CI"):
        flags.append("CI")
    return ", ".join(flags) or ("native Windows" if system == "Windows" else "native")


def environment_context() -> str:
    """Describe the host for the model.

    Environment variables and platform metadata only: the model can check for
    a tool with the shell tool when it actually needs one, which is cheaper and
    more accurate than probing a fixed list of executables at every startup.
    """
    system = platform.system()
    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME", system)
    except OSError:
        os_name = platform.platform()

    return "\n".join([
        f"Operating system: {os_name}",
        f"Kernel: {system} {platform.release()}",
        f"CPU architecture: {platform.machine()}",
        f"Runtime: {_runtime_flags(system)}",
        f"Locale/encoding: {locale.getlocale()[0] or 'unknown'} / {locale.getpreferredencoding(False)}",
        f"Terminal: {_terminal_description()}",
        f"Shell: {_shell_description(system)}",
    ])


def load_context_files(cwd: Path) -> list[tuple[Path, str]]:
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


def build_system_prompt(cwd: Path) -> str:
    prompt = INSTRUCTIONS

    context_files = load_context_files(cwd)
    if context_files:
        prompt += "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
        for path, content in context_files:
            prompt += f'<project_instructions path="{path}">\n{content}\n</project_instructions>\n\n'
        prompt += "</project_context>"

    prompt += "\n\n<environment>"
    prompt += f"\nCurrent date: {date.today().isoformat()}"
    prompt += f"\nCurrent working directory: {cwd}"
    prompt += f"\n{environment_context()}"
    prompt += "\n</environment>"
    return prompt
