"""Opt-in tools that let the agent inspect and modify its own Textual app.

The bridge is deliberately powerful: ``textual_eval`` and ``textual_exec`` run
Python in the UI process.  It is only constructed by the CLI when
``--textual-debug`` is present, so normal agents never receive these schemas.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
from pathlib import Path

import textual
from pydantic import ConfigDict, with_config
from typing_extensions import NotRequired, TypedDict

from . import tools


_spec = with_config(ConfigDict(use_attribute_docstrings=True))


@_spec
class InspectArgs(TypedDict):
    selector: NotRequired[str]
    """CSS selector to inspect. Omit it to show the complete active screen tree."""
    include_styles: NotRequired[bool]
    """Include the full computed CSS for every matching node (default false)."""
    limit: NotRequired[int]
    """Maximum number of nodes to return (default 100, maximum 500)."""


@_spec
class EvalArgs(TypedDict):
    expression: str
    """Python expression evaluated with app, screen, pane, focused, query and textual available."""


@_spec
class ExecArgs(TypedDict):
    code: str
    """Python code to execute. Top-level await and imports are supported; assign result to return it."""


@_spec
class CssArgs(TypedDict):
    css: str
    """Textual CSS to add as a live override. Overrides accumulate until the app exits."""


@_spec
class ScreenshotArgs(TypedDict):
    path: NotRequired[str]
    """SVG output path, relative to the working directory (default .paimon/textual-debug.svg)."""
    title: NotRequired[str]
    """Optional title embedded in the SVG."""
    simplify: NotRequired[bool]
    """Ask Textual to simplify the SVG output (default false)."""


class TextualDebugBridge:
    """Late-bound access to a running app for the opt-in debug tool set."""

    def __init__(self) -> None:
        self.app = None
        # Names created by imports and assignments in textual_exec deliberately
        # survive later calls, making iterative UI experiments practical.
        self.namespace: dict[str, object] = {"textual": textual, "state": {}}

    def attach(self, app) -> None:
        """Attach after PaimonApp exists but before its event loop starts."""
        self.app = app

    def _scope(self) -> dict[str, object]:
        if self.app is None:
            raise RuntimeError("Textual app is not attached yet")
        app = self.app
        screen = app.screen
        self.namespace.update({
            "app": app,
            "screen": screen,
            "pane": getattr(app, "pane", None),
            "focused": getattr(app, "focused", None),
            "query": screen.query,
            "query_one": screen.query_one,
            "bridge": self,
        })
        return self.namespace

    @staticmethod
    def _captured(stdout: io.StringIO, value: object = None, *, show_none: bool = False) -> str:
        output = stdout.getvalue().rstrip()
        if value is not None or show_none:
            rendered = repr(value)
            output = f"{output}\n{rendered}" if output else rendered
        return output or "(no output)"

    async def evaluate(self, args: dict, cwd: Path, mode: str, ctx: tools.ToolContext) -> str:
        scope = self._scope()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            value = eval(compile(args["expression"], "<textual_eval>", "eval"), scope)
            if inspect.isawaitable(value):
                value = await value
        return self._captured(stream, value, show_none=True)

    async def execute(self, args: dict, cwd: Path, mode: str, ctx: tools.ToolContext) -> str:
        scope = self._scope()
        scope.pop("result", None)
        stream = io.StringIO()
        code = compile(args["code"], "<textual_exec>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            pending = eval(code, scope)
            if inspect.isawaitable(pending):
                await pending
        return self._captured(stream, scope.get("result"))

    def inspect(self, args: dict, cwd: Path, mode: str, ctx: tools.ToolContext) -> str:
        scope = self._scope()
        screen = scope["screen"]
        selector = args.get("selector")
        if selector:
            nodes = list(screen.query(selector))
        else:
            nodes = list(screen.walk_children(with_self=True))
        limit = max(1, min(args.get("limit", 100), 500))
        include_styles = args.get("include_styles", False)
        focused = scope["focused"]
        lines: list[str] = []
        for node in nodes[:limit]:
            depth = 0
            parent = getattr(node, "parent", None)
            while parent is not None and parent is not screen:
                depth += 1
                parent = getattr(parent, "parent", None)
            identifier = str(getattr(node, "css_identifier", type(node).__name__))
            classes = sorted(getattr(node, "classes", ()))
            if classes:
                identifier += "".join(f".{name}" for name in classes)
            region = getattr(node, "region", None)
            virtual = getattr(node, "virtual_region", None)
            pseudo = ",".join(sorted(node.get_pseudo_classes()))
            flags = []
            if node is focused:
                flags.append("focused")
            if not getattr(node, "display", True):
                flags.append("hidden")
            suffix = f" [{' '.join(flags)}]" if flags else ""
            lines.append(
                f"{'  ' * depth}{identifier} region={region} virtual={virtual} pseudo={pseudo}{suffix}"
            )
            if include_styles:
                css = getattr(getattr(node, "styles", None), "css", "") or "(no computed rules)"
                lines.extend(f"{'  ' * (depth + 1)}{line}" for line in css.splitlines())
        if not lines:
            return f"No nodes match {selector!r}."
        if len(nodes) > limit:
            lines.append(f"... ({len(nodes) - limit} more nodes; raise limit to show them)")
        return "\n".join(lines)

    def apply_css(self, args: dict, cwd: Path, mode: str, ctx: tools.ToolContext) -> str:
        app = self._scope()["app"]
        app.stylesheet.add_source(args["css"])
        app.stylesheet.reparse()
        app.refresh_css(animate=False)
        return "CSS override applied. It remains active until this app exits."

    def screenshot(self, args: dict, cwd: Path, mode: str, ctx: tools.ToolContext) -> str:
        app = self._scope()["app"]
        path = Path(args.get("path", ".paimon/textual-debug.svg"))
        if not path.is_absolute():
            path = cwd / path
        path.parent.mkdir(parents=True, exist_ok=True)
        svg = app.export_screenshot(title=args.get("title"), simplify=args.get("simplify", False))
        path.write_text(svg, encoding="utf-8")
        return f"Saved Textual screenshot to {path} ({len(svg)} chars)."

    def toolset(self) -> dict[str, tools.Tool]:
        """A fresh mapping suitable for merging into an Agent toolset."""
        return {
            "textual_inspect": tools.Tool(
                description=(
                    "Inspect the live Textual DOM by CSS selector, including geometry, pseudo-classes, "
                    "focus/visibility and optionally every computed CSS rule. With no selector, returns "
                    "the active screen tree. Available only under --textual-debug."
                ),
                params=InspectArgs,
                run=self.inspect,
            ),
            "textual_eval": tools.Tool(
                description=(
                    "Evaluate an arbitrary Python expression inside this running Textual UI and return "
                    "its repr. The persistent namespace contains app, screen, pane, focused, query, "
                    "query_one, textual, state and bridge. Awaitables are awaited. Use this to inspect "
                    "any public or private Textual API. Available only under --textual-debug."
                ),
                params=EvalArgs,
                run=self.evaluate,
            ),
            "textual_exec": tools.Tool(
                description=(
                    "Execute arbitrary Python inside this running Textual UI. Imports, assignments and "
                    "top-level await work; names persist across calls. stdout/stderr are returned, as is "
                    "the variable result when assigned. Use it to mount/remove widgets, invoke actions, "
                    "patch internals or run multi-step experiments. Available only under --textual-debug."
                ),
                params=ExecArgs,
                run=self.execute,
            ),
            "textual_apply_css": tools.Tool(
                description=(
                    "Parse and apply a live Textual CSS override to the running app. Overrides accumulate "
                    "and relayout immediately; restart the app to clear them. Available only under "
                    "--textual-debug."
                ),
                params=CssArgs,
                run=self.apply_css,
            ),
            "textual_screenshot": tools.Tool(
                description=(
                    "Export the currently rendered Textual screen as SVG to a file. Use after DOM/style "
                    "changes to capture the actual render. Available only under --textual-debug."
                ),
                params=ScreenshotArgs,
                run=self.screenshot,
            ),
        }
