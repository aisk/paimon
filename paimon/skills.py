"""Agent Skills: discovery, prompt listing and explicit ``/skill:name`` invocation.

A skill is a directory with a ``SKILL.md`` (YAML frontmatter plus Markdown
instructions) per https://agentskills.io. Only name, description and location
go into the system prompt; the model reads the file with read_file when a task
matches, and the user can force it with ``/skill:name args``.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Sequence, Union

import yaml

from .config import config_root

SKILL_FILE = "SKILL.md"
COMMAND_PREFIX = "/skill:"
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

_NAME_RE = re.compile(r"^[a-z0-9-]+$")
_BLOCK_START_RE = re.compile(r'^<skill name="([^"]+)" location="([^"]+)">\n')


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    base_dir: Path
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillDiagnostic:
    message: str
    path: Path


@dataclass(frozen=True)
class SkillBlock:
    name: str
    location: str
    body: str
    user_message: Optional[str]


# ---- parsing ---------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``---`` delimited YAML frontmatter from the body.

    Without frontmatter the whole text is the body. A header that is not a
    mapping counts as empty. Raises ``yaml.YAMLError`` on malformed YAML.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines[0] != "---":
        return {}, text
    # The closing fence is a line that is exactly ---, so ---- or ---x in the
    # header do not end it and an empty header (--- right after ---) does.
    for index in range(1, len(lines)):
        if lines[index] == "---":
            header = yaml.safe_load("\n".join(lines[1:index]))
            body = "\n".join(lines[index + 1:]).strip()
            return (header if isinstance(header, dict) else {}), body
    return {}, text


def _validate_name(name: str) -> list[str]:
    errors = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not _NAME_RE.match(name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def load_skill_file(path: Path, *, declared: bool) -> tuple[Optional[Skill], list[SkillDiagnostic]]:
    """Load one skill file.

    ``declared`` means the file is a ``SKILL.md``: problems are reported.
    Any other Markdown file is a skill only when it says so with a
    description, and is otherwise skipped silently.
    """
    diagnostics: list[SkillDiagnostic] = []
    try:
        frontmatter, _ = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        diagnostics.append(SkillDiagnostic(f"cannot read skill: {exc}", path))
        return None, diagnostics
    except yaml.YAMLError as exc:
        if declared:
            diagnostics.append(SkillDiagnostic(f"invalid frontmatter: {exc}", path))
        return None, diagnostics

    description = frontmatter.get("description")
    has_description = isinstance(description, str) and description.strip() != ""
    if not has_description:
        if declared:
            diagnostics.append(SkillDiagnostic("description is required", path))
        return None, diagnostics
    if len(description) > MAX_DESCRIPTION_LENGTH:
        diagnostics.append(SkillDiagnostic(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})", path))

    raw_name = frontmatter.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else path.parent.name
    for error in _validate_name(name):
        diagnostics.append(SkillDiagnostic(error, path))

    skill = Skill(
        name=name,
        description=description.strip(),
        path=path,
        base_dir=path.parent,
        disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
    )
    return skill, diagnostics


# ---- directory scanning ----------------------------------------------------


class _Ignore:
    """The subset of gitignore syntax worth honoring while scanning skills.

    A pattern applies below the directory of the ignore file that declared
    it. Patterns without a slash match any path component there, anchored
    ones (a slash inside, or leading) match the path relative to that
    directory, so ``build/`` and ``*.bak`` work as expected. Matching is case
    sensitive like git's. Negation (``!``) and ``**`` are not supported.
    """

    def __init__(self) -> None:
        # (prefix of the declaring directory, pattern, anchored, directory only)
        self._patterns: list[tuple[str, str, bool, bool]] = []

    def add_file(self, path: Path, prefix: str) -> None:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            directory_only = line.endswith("/")
            line = line.rstrip("/")
            anchored = "/" in line
            self._patterns.append((prefix, line.lstrip("/"), anchored, directory_only))

    def ignores(self, relative: str, is_dir: bool) -> bool:
        for prefix, pattern, anchored, directory_only in self._patterns:
            if directory_only and not is_dir:
                continue
            if not relative.startswith(prefix):
                continue
            below = relative[len(prefix):]
            if anchored:
                if fnmatchcase(below, pattern):
                    return True
            elif any(fnmatchcase(part, pattern) for part in PurePosixPath(below).parts):
                return True
        return False


def load_skills_from_dir(root: Path) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """Scan a directory for skills.

    A directory holding a ``SKILL.md`` is a skill root and is not descended
    into. Otherwise the root's own ``.md`` files are candidates and every
    subdirectory is scanned in turn. Dotfiles, ``node_modules`` and anything
    matched by a ``.gitignore``/``.ignore``/``.fdignore`` along the way are
    skipped.
    """
    root = root.expanduser()
    if not root.is_dir():
        return [], []
    return _scan(root, root, _Ignore(), include_root_files=True)


def _scan(directory: Path, root: Path, ignore: _Ignore, *,
          include_root_files: bool) -> tuple[list[Skill], list[SkillDiagnostic]]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    relative_dir = directory.relative_to(root).as_posix() if directory != root else ""
    prefix = f"{relative_dir}/" if relative_dir else ""
    for name in IGNORE_FILE_NAMES:
        candidate = directory / name
        if candidate.is_file():
            ignore.add_file(candidate, prefix)

    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        diagnostics.append(SkillDiagnostic(f"cannot list directory: {exc}", directory))
        return skills, diagnostics

    declared = directory / SKILL_FILE
    if declared.is_file() and not ignore.ignores(prefix + SKILL_FILE, False):
        skill, found = load_skill_file(declared, declared=True)
        if skill is not None:
            skills.append(skill)
        return skills, diagnostics + found

    for entry in entries:
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        try:
            is_dir = entry.is_dir()
            is_file = entry.is_file()
        except OSError:
            continue
        relative = prefix + entry.name
        if ignore.ignores(relative, is_dir):
            continue
        if is_dir:
            sub_skills, sub_diagnostics = _scan(entry, root, ignore, include_root_files=False)
            skills.extend(sub_skills)
            diagnostics.extend(sub_diagnostics)
        elif is_file and include_root_files and entry.suffix == ".md":
            skill, found = load_skill_file(entry, declared=False)
            if skill is not None:
                skills.append(skill)
            diagnostics.extend(found)
    return skills, diagnostics


# ---- locations -------------------------------------------------------------


def default_skill_dirs(cwd: Path) -> list[Path]:
    """Where skills are looked for without any configuration, most specific
    first: ``.agents/skills`` from cwd up to the git root (or the filesystem
    root outside a repository), then the user's ``~/.agents/skills`` and
    ``~/.config/paimon/skills``.
    """
    user_agents = Path.home() / ".agents" / "skills"
    dirs: list[Path] = []
    current = cwd.resolve()
    while True:
        candidate = current / ".agents" / "skills"
        if candidate != user_agents.resolve() and candidate not in dirs:
            dirs.append(candidate)
        if (current / ".git").exists() or current == current.parent:
            break
        current = current.parent
    return [*dirs, user_agents, config_root() / "skills"]


def discover_skills(cwd: Path, *, extra_paths: Sequence[Union[str, Path]] = (),
                    include_defaults: bool = True) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """Load skills from ``extra_paths`` and then the default locations.

    Extra paths may be files or directories, ``~`` expands, relative paths
    resolve against cwd. On a name collision the first skill found wins, so
    explicitly configured beats project beats global; the loser is reported.
    The same file reached twice (symlinks) is dropped silently.
    """
    by_name: dict[str, Skill] = {}
    seen_files: set[Path] = set()
    diagnostics: list[SkillDiagnostic] = []

    def add(found: list[Skill], found_diagnostics: list[SkillDiagnostic]) -> None:
        diagnostics.extend(found_diagnostics)
        for skill in found:
            try:
                real = skill.path.resolve()
            except OSError:
                real = skill.path
            if real in seen_files:
                continue
            existing = by_name.get(skill.name)
            if existing is not None:
                diagnostics.append(SkillDiagnostic(
                    f'name "{skill.name}" collision: kept {existing.path}', skill.path))
                continue
            by_name[skill.name] = skill
            seen_files.add(real)

    for raw in extra_paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if path.is_dir():
            add(*load_skills_from_dir(path))
        elif path.is_file() and path.suffix == ".md":
            skill, found = load_skill_file(path, declared=True)
            add([skill] if skill is not None else [], found)
        elif path.exists():
            diagnostics.append(SkillDiagnostic("skill path is not a markdown file", path))
        else:
            diagnostics.append(SkillDiagnostic("skill path does not exist", path))

    if include_defaults:
        for directory in default_skill_dirs(cwd):
            add(*load_skills_from_dir(directory))

    return list(by_name.values()), diagnostics


# ---- prompt and invocation -------------------------------------------------


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """The ``<available_skills>`` block, or "" when nothing is model-visible."""
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Use read_file to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill "
        "directory (the parent of SKILL.md) and use that absolute path in tool calls.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines += [
            "  <skill>",
            f"    <name>{html.escape(skill.name, quote=True)}</name>",
            f"    <description>{html.escape(skill.description, quote=True)}</description>",
            f"    <location>{html.escape(str(skill.path), quote=True)}</location>",
            "  </skill>",
        ]
    lines.append("</available_skills>")
    return "\n".join(lines)


def find_skill(name: str, skills: Sequence[Skill]) -> Optional[Skill]:
    return next((s for s in skills if s.name == name), None)


def format_invocation(skill: Skill, body: str, args: str = "") -> str:
    attributes = (f'name="{html.escape(skill.name, quote=True)}" '
                  f'location="{html.escape(str(skill.path), quote=True)}"')
    block = (f"<skill {attributes}>\nReferences are relative to {skill.base_dir}.\n\n"
             f"{body}\n</skill>")
    return f"{block}\n\n{args}" if args else block


def expand_skill_command(text: str, skills: Sequence[Skill],
                         expand_args: Callable[[str], str] = lambda args: args) -> str:
    """Expand a ``/skill:name args`` message into the skill's full content.

    ``expand_args`` is applied to the user's arguments only: the skill body is
    the author's text and must not have, say, an ``@path`` in it rewritten
    against whatever the user's cwd happens to hold. Anything that is not a
    command, including an unknown or unreadable skill, is returned as is: the
    model then sees what the user typed and can say so.
    """
    if not text.startswith(COMMAND_PREFIX):
        return text
    match = re.match(r"(\S*)\s*([\s\S]*)", text[len(COMMAND_PREFIX):])
    name, args = match.group(1), match.group(2)
    skill = find_skill(name, skills)
    if skill is None:
        return text
    try:
        _, body = parse_frontmatter(skill.path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, yaml.YAMLError):
        return text
    return format_invocation(skill, body, expand_args(args.strip()))


def parse_skill_block(text: str) -> Optional[SkillBlock]:
    """The inverse of ``format_invocation``, for rendering a stored message.

    The block ends at the last ``</skill>`` line, so a body that quotes one
    is not cut short.
    """
    match = _BLOCK_START_RE.match(text)
    if match is None:
        return None
    end = text.rfind("\n</skill>")
    if end < match.end():
        return None
    body = text[match.end():end]
    tail = text[end + len("\n</skill>"):]
    if tail == "":
        user_message = None
    elif tail.startswith("\n\n"):
        user_message = tail[2:]
    else:
        return None
    return SkillBlock(html.unescape(match.group(1)), html.unescape(match.group(2)), body, user_message)
