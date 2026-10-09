"""The system prompt: instructions, project context files, host environment.

Built when a session is created and rebuilt when it is resumed, so the date,
the environment and the project rules are current; each snapshot a turn ran
with is persisted in the session log.
"""

import locale
import os
import platform
from datetime import date
from pathlib import Path
from typing import Collection, Sequence

from .skills import Skill, format_skills_for_prompt

CONTEXT_FILE = "AGENTS.md"

_INTRO = """You are Paimon, a concise coding assistant operating in a terminal.

You help with software engineering tasks by reading and editing files and running
shell commands."""

# The available tools are not enumerated here: their schemas travel with every
# request. A guideline that names a tool is paired with the tools it needs and
# dropped for an agent that does not hold all of them (headless, a subagent, a
# narrowed toolset), so the prompt never points the model at a tool it cannot
# call.
_GUIDELINES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("edit_file", "write_file"),
     "Prefer reading a file before editing it. For edits, use edit_file with a unique "
     "old_string; only use write_file for new files or full rewrites."),
    (("glob", "grep"),
     "Use glob to find files by name pattern and grep to search file contents."),
    (("shell",),
     "Use the shell tool for git, running tests and anything the other tools do not cover."),
    (("run_code",),
     "Call tools directly by default. Use run_code when a script saves steps or context: "
     "three or more independent calls, a loop over many files or results, or output that "
     "should be filtered before it reaches you."),
    (("write_todos",),
     "For tasks with several steps, call write_todos first to lay out a plan, then keep "
     "it updated as you go (one task in_progress at a time). Skip it for simple tasks."),
    (("ask_user",),
     "When different readings of the request would lead to materially different work "
     "and nothing in the code settles it, call ask_user before building; for routine "
     "calls decide yourself and say what you assumed."),
    (("start_new_session",),
     "When the earlier conversation is mostly irrelevant to the next phase of work, "
     "call start_new_session with a self-contained handoff prompt instead of "
     "continuing in a bloated context."),
    (("search_history", "read_history"),
     "Details lost to context compaction are not gone: the session log keeps the "
     "full history. Use search_history to find them and read_history to read the "
     "originals instead of guessing from the summary."),
    ((),
     "User @path mentions are expanded into <mentioned_file> tags. A tag with a body "
     "contains the complete file; a self-closing tag gives only the path, so read "
     "the file when you need its contents."),
    ((),
     "Be direct. When the task is done, briefly state what you did. Don't narrate every step."),
)


def instructions(tool_names: Collection[str] | None = None) -> str:
    """The fixed part of the prompt, for an agent holding ``tool_names``.

    None keeps every guideline, for callers with no toolset to narrow by.
    """
    lines = [text for needs, text in _GUIDELINES
             if tool_names is None or all(name in tool_names for name in needs)]
    return _INTRO + "\n\nGuidelines:\n" + "\n".join(f"- {line}" for line in lines)


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


def _shell_description() -> str:
    # The shell the shell tool actually uses — never $SHELL, which names the
    # user's login shell and used to make the prompt promise bash while the
    # tool ran /bin/sh.
    from .tools import shell_executable

    shell = shell_executable()
    if shell is not None:
        return shell
    # Windows: create_subprocess_shell uses ComSpec, defaulting to cmd.exe.
    return os.environ.get("ComSpec") or "cmd.exe"


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
        f"Shell: {_shell_description()}",
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
                found.append((candidate, candidate.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
        if current == current.parent:
            break
        current = current.parent
    found.reverse()
    return found


def build_system_prompt(cwd: Path, skills: Sequence[Skill] = (),
                        tool_names: Collection[str] | None = None) -> str:
    prompt = instructions(tool_names)

    context_files = load_context_files(cwd)
    if context_files:
        prompt += "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
        for path, content in context_files:
            prompt += f'<project_instructions path="{path}">\n{content}\n</project_instructions>\n\n'
        prompt += "</project_context>"

    skills_block = format_skills_for_prompt(skills)
    if skills_block:
        prompt += f"\n\n{skills_block}"

    prompt += "\n\n<environment>"
    prompt += f"\nCurrent date: {date.today().isoformat()}"
    prompt += f"\nCurrent working directory: {cwd}"
    prompt += f"\n{environment_context()}"
    prompt += "\n</environment>"
    return prompt
