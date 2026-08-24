"""Agent types for spawn_agent: what a named subagent runs as.

An agent type is one markdown file: YAML frontmatter naming it and narrowing
what it may use, plus a body appended to the subagent's system prompt. The
type's name goes into spawn_agent's schema, so the model picks one the way it
picks any other argument; everything else about starting the agent (the pane,
the session, the depth limit) stays with the supervisor and the app.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml

from . import tools
from .config import config_root
from .skills import SkillDiagnostic, _validate_name, parse_frontmatter

MAX_DESCRIPTION_LENGTH = 1024


@dataclass(frozen=True)
class AgentType:
    name: str
    # Shown to the model in spawn_agent's description: when to pick this type.
    description: str
    # Appended to the subagent's system prompt; may be empty.
    body: str
    # Allowed tool names; None means everything a subagent may hold. Names the
    # registry does not know are kept here and filtered at spawn time, so a
    # definition written for a newer paimon still loads.
    tools: Optional[tuple[str, ...]] = None
    model: Optional[str] = None
    # None for a built-in type.
    path: Optional[Path] = None


# ---- loading ----------------------------------------------------------------


def _parse_tools(value: object, path: Path,
                 diagnostics: list[SkillDiagnostic]) -> Optional[tuple[str, ...]]:
    """The ``tools`` field as a tuple of names; None when absent or unusable."""
    if value is None:
        return None
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        names = [item.strip() for item in value if item.strip()]
    else:
        diagnostics.append(SkillDiagnostic(
            "tools must be a list of tool names or a comma-separated string", path))
        return None
    for name in names:
        if name not in tools.REGISTRY:
            diagnostics.append(SkillDiagnostic(f'unknown tool "{name}"', path))
    return tuple(names)


def load_agent_type(path: Path) -> tuple[Optional[AgentType], list[SkillDiagnostic]]:
    """Load one agent type file. Problems are always reported: unlike a stray
    markdown file in a skills tree, everything in an agents directory claims
    to be a definition."""
    diagnostics: list[SkillDiagnostic] = []
    try:
        frontmatter, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return None, [SkillDiagnostic(f"cannot read agent type: {exc}", path)]
    except yaml.YAMLError as exc:
        return None, [SkillDiagnostic(f"invalid frontmatter: {exc}", path)]

    description = frontmatter.get("description")
    if not (isinstance(description, str) and description.strip()):
        diagnostics.append(SkillDiagnostic("description is required", path))
        return None, diagnostics
    if len(description) > MAX_DESCRIPTION_LENGTH:
        diagnostics.append(SkillDiagnostic(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})", path))

    raw_name = frontmatter.get("name")
    name = raw_name if isinstance(raw_name, str) and raw_name else path.stem
    for error in _validate_name(name):
        diagnostics.append(SkillDiagnostic(error, path))

    model = frontmatter.get("model")
    agent_type = AgentType(
        name=name,
        description=description.strip(),
        body=body,
        tools=_parse_tools(frontmatter.get("tools"), path, diagnostics),
        model=model if isinstance(model, str) and model else None,
        path=path,
    )
    return agent_type, diagnostics


def load_agent_types_from_dir(root: Path) -> tuple[list[AgentType], list[SkillDiagnostic]]:
    """Every ``*.md`` directly in ``root``, one type per file.

    Flat on purpose: a type has no companion files, so nesting would only
    invite the SKILL.md directory shape where it buys nothing.
    """
    if not root.is_dir():
        return [], []
    found: list[AgentType] = []
    diagnostics: list[SkillDiagnostic] = []
    for path in sorted(root.glob("*.md")):
        if not path.is_file():
            continue
        agent_type, file_diagnostics = load_agent_type(path)
        diagnostics.extend(file_diagnostics)
        if agent_type is not None:
            found.append(agent_type)
    return found, diagnostics


# ---- built-in types ---------------------------------------------------------


_EXPLORE_DESCRIPTION = (
    "Read-only scout for questions about the code as it is: where something "
    "lives, how a mechanism works, what a change would touch. It cannot edit "
    "files or run commands, so its findings need no review before use."
)

_EXPLORE_PROMPT = """\
You are a read-only explore agent. You investigate; you never change anything.

You have no tools that edit files, run commands, or alter any state, and you
must not try to work around that. Search with grep and glob, read files with
read_file, and keep digging until the question is answered by what the code
actually says, citing paths (and line numbers where useful).

Nothing you see reaches the caller on its own: your final message is the only
thing handed back. End with a self-contained report of what you found, precise
enough to act on without re-reading the files."""


def builtin_types() -> list[AgentType]:
    """Types every installation has. Derived from the registry at call time,
    so a new read-only tool joins explore without a list to maintain."""
    read_only = tuple(
        name for name, tool in tools.REGISTRY.items()
        if tool.access in ("read", "none") and name not in tools.SUBAGENT_DENIED
    )
    return [AgentType(name="explore", description=_EXPLORE_DESCRIPTION,
                      body=_EXPLORE_PROMPT, tools=read_only)]


# ---- locations and discovery ------------------------------------------------


def default_agent_dirs(cwd: Path) -> list[Path]:
    """Where agent types are looked for without any configuration, most
    specific first: ``.agents/agents`` from cwd up to the git root (or the
    filesystem root outside a repository), then the user's
    ``~/.agents/agents`` and ``~/.config/paimon/agents``.
    """
    user_agents = Path.home() / ".agents" / "agents"
    dirs: list[Path] = []
    current = cwd.resolve()
    while True:
        candidate = current / ".agents" / "agents"
        if candidate != user_agents.resolve() and candidate not in dirs:
            dirs.append(candidate)
        if (current / ".git").exists() or current == current.parent:
            break
        current = current.parent
    return [*dirs, user_agents, config_root() / "agents"]


def discover_agent_types(cwd: Path, *, extra_paths: Sequence[Union[str, Path]] = (),
                         include_defaults: bool = True,
                         ) -> tuple[list[AgentType], list[SkillDiagnostic]]:
    """Load agent types from ``extra_paths``, the default locations, and the
    built-ins, in that order.

    First found wins, like skills: explicitly configured beats project beats
    global, and any of them beats a built-in — shadowing ``explore`` is a
    feature, so it is the one collision that is not reported.
    """
    by_name: dict[str, AgentType] = {}
    seen_files: set[Path] = set()
    diagnostics: list[SkillDiagnostic] = []

    def add(found: list[AgentType], found_diagnostics: list[SkillDiagnostic]) -> None:
        diagnostics.extend(found_diagnostics)
        for agent_type in found:
            try:
                real = agent_type.path.resolve() if agent_type.path else None
            except OSError:
                real = agent_type.path
            if real is not None and real in seen_files:
                continue
            existing = by_name.get(agent_type.name)
            if existing is not None:
                diagnostics.append(SkillDiagnostic(
                    f'name "{agent_type.name}" collision: kept {existing.path}',
                    agent_type.path or Path(agent_type.name)))
                continue
            by_name[agent_type.name] = agent_type
            if real is not None:
                seen_files.add(real)

    for raw in extra_paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = cwd / path
        if path.is_dir():
            add(*load_agent_types_from_dir(path))
        elif path.is_file() and path.suffix == ".md":
            agent_type, found = load_agent_type(path)
            add([agent_type] if agent_type is not None else [], found)
        elif path.exists():
            diagnostics.append(SkillDiagnostic("agent path is not a markdown file", path))
        else:
            diagnostics.append(SkillDiagnostic("agent path does not exist", path))

    if include_defaults:
        for directory in default_agent_dirs(cwd):
            add(*load_agent_types_from_dir(directory))

    for builtin in builtin_types():
        by_name.setdefault(builtin.name, builtin)

    return list(by_name.values()), diagnostics


def find_type(name: str, types: Sequence[AgentType]) -> Optional[AgentType]:
    return next((t for t in types if t.name == name), None)


# ---- the dynamic spawn_agent schema -----------------------------------------


def spawn_tool_with_types(tool: tools.Tool, types: Sequence[AgentType]) -> tools.Tool:
    """A copy of the spawn_agent tool that knows these types.

    A copy, never a mutation: the registry's schema dicts are shared by every
    agent in the process, and are what the static tool falls back to where no
    types exist. No enum on the parameter — a stale name should reach the
    launcher and come back as a readable error listing what does exist, not
    die in generic argument validation.
    """
    schema = copy.deepcopy(tool.schema)
    function = schema["function"]
    function["parameters"]["properties"]["agent"] = {
        "type": "string",
        "description": "Named agent type to run as (optional; omit for a "
                       "general-purpose agent with your own tools).",
    }
    listing = "\n".join(f"- {t.name}: {t.description}" for t in types)
    function["description"] += f"\n\nAgent types available for 'agent':\n{listing}"
    return dataclasses.replace(tool, schema=schema)
