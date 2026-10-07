"""A conversation as it is drawn: the scrolling log and what feeds it events.

Shared by every pane that shows an agent at work. ``SessionPane`` puts a
prompt under one; ``AgentPane`` shows one and nothing else.
"""

from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Static
from textual.widgets.markdown import MarkdownStream

from .agent import (
    Agent,
    JobNotice,
    CompactionNotice,
    ContextCompactionFailed,
    ContextCompacted,
    ModelRetry,
    ReasoningDelta,
    ShellRun,
    TextDelta,
    TodosUpdate,
    ToolEnd,
    ToolStart,
    UserInput,
)
from .skills import parse_skill_block
from .ui import (
    AssistantMessage,
    FoldedText,
    ToolCall,
    ToolEntry,
    ToolGroup,
    ToolResult,
    UserMessage,
)

# All three markers are East Asian Width "narrow", so the labels stay aligned on
# terminals that render ambiguous-width glyphs double-wide. Finished work is
# struck through and dimmed to keep the accent on whatever is in progress.
_TODO_STYLE = {
    "completed": ("✓", "$text-disabled strike"),
    "in_progress": ("▸", "$text-accent b"),
    "pending": ("◦", "$text-muted"),
}


class Transcript(VerticalScroll):
    """The log of one conversation.

    Anchored once it is mounted: the compositor keeps an anchored scrollable
    pinned to the bottom as content grows, releases the anchor while the user
    scrolls up, and re-engages it when they return to the bottom. The helpers
    therefore just mount widgets, with no manual scrolling.
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._todo_panel: Static | None = None

    def on_mount(self) -> None:
        self.anchor()

    def add(self, renderable, classes: str = "") -> Static:
        widget = Static(renderable, classes=classes)
        self.mount(widget)
        return widget

    def add_user(self, body: str) -> UserMessage:
        # A replayed /skill:name turn is stored expanded; show it folded
        # behind the skill's name, followed by whatever the user added.
        block = parse_skill_block(body)
        if block is not None:
            self.mount(FoldedText(block.body, classes="skill-invocation", label=f"skill {block.name}"))
            body = block.user_message or f"/skill:{block.name}"
        widget = UserMessage(body)
        self.mount(widget)
        return widget

    def add_tool_result(
        self, result: str, *, label: str = "", denied: bool = False,
        container: Widget | None = None,
    ) -> ToolResult:
        widget = ToolResult(result, label=label, denied=denied)
        (container if container is not None else self).mount(widget)
        return widget

    async def add_shell_call(self, command: str) -> Vertical:
        """The box a "!" run is logged in, with its command line already in it.

        Awaited, since the result mounts into the box the moment the command
        exits and a container has to be mounted before anything goes in it.
        """
        step = Vertical(classes="shell-step")
        await self.mount(step)
        await step.mount(ToolCall("!", command))
        return step

    def show_todos(self, todos: list[dict]) -> None:
        """Update the panel in place while it is still the tail of the log, so a
        burst of revisions collapses into one; once anything is logged under it
        the panel stays put as a snapshot of the plan at that point and the next
        revision starts a new one."""
        if not todos:
            if self._todo_panel is not None:
                self._todo_panel.remove()
                self._todo_panel = None
            return
        body = self._render_todos(todos)
        if self._todo_panel is not None and self.children[-1:] == [self._todo_panel]:
            self._todo_panel.update(body)
        else:
            self._todo_panel = self.add(body, classes="todos")

    def _render_todos(self, todos: list[dict]) -> Content:
        done = sum(1 for t in todos if t.get("status") == "completed")
        lines = [f"[$text-muted b]Plan[/][$text-muted]  {done}/{len(todos)}[/]"]
        kwargs = {}
        for i, t in enumerate(todos):
            marker, style = _TODO_STYLE.get(t.get("status"), _TODO_STYLE["pending"])
            kwargs[f"c{i}"] = t.get("content", "")
            lines.append(f"[{style}]{marker} ${f'c{i}'}[/]")
        return Content.from_markup("\n".join(lines), **kwargs)

    def clear(self) -> None:
        """Empty the log for a different conversation."""
        self.remove_children()
        self._todo_panel = None


class EventRenderer:
    """Renders one agent's events into a transcript.

    The single rendering path: live turns and resumed-history replay both feed
    events through ``handle``, so history always looks like it did live.
    """

    def __init__(self, transcript: Transcript, agent: Agent) -> None:
        self._log = transcript
        # Asked for its config and cwd as they are needed rather than once:
        # the config is process-wide and can be replaced under a live agent.
        self._agent = agent
        self._stream: MarkdownStream | None = None
        self._reasoning: FoldedText | None = None
        self._reasoning_buf = ""
        self._first_text_block = True
        # Consecutive calls between prose blocks share one collapsed activity
        # group. Entries stay addressable by call id until their result arrives.
        self._tool_group: ToolGroup | None = None
        self._tool_entries: dict[str, ToolEntry] = {}

    async def handle(self, ev: object) -> None:
        if isinstance(ev, UserInput):
            await self.close()
            self._first_text_block = True
            self._log.add_user(ev.text)

        elif isinstance(ev, CompactionNotice):
            await self.close()
            self._first_text_block = True
            self._log.add(Content.from_markup("[$text-muted]Earlier context was compacted[/]"))

        elif isinstance(ev, JobNotice):
            await self.close()
            self._first_text_block = True
            # The first line names the job and how it ended; an agent's answer
            # follows it and folds like any other result.
            header, _, body = ev.text.partition("\n")
            header = header.rstrip(":")
            self._log.add(Content.from_markup("[$text-muted]$text[/]", text=header))
            if body:
                self._log.add_tool_result(body, label=header)

        elif isinstance(ev, ShellRun):
            # Replay only: a command run live is logged as it happens, by the
            # worker that runs it, so its output is on screen before the model
            # ever hears about it.
            await self.close()
            self._first_text_block = True
            step = await self._log.add_shell_call(ev.command)
            self._log.add_tool_result(ev.output, label=ev.command, container=step)

        elif isinstance(ev, ReasoningDelta):
            self._reasoning_buf += ev.text
            if self._reasoning is None:
                self._reasoning = FoldedText(
                    "",
                    classes="reasoning",
                    expanded=self._agent.config.show_reasoning,
                    label="reasoning",
                )
                await self._log.mount(self._reasoning)
            self._reasoning.set_text(self._reasoning_buf)

        elif isinstance(ev, TextDelta):
            if self._stream is None:
                await self._close_content()
                self._finish_tool_group()
                widget = AssistantMessage("", heading=self._first_text_block)
                self._first_text_block = False
                # Await the mount so the initial document (the Paimon heading)
                # is rendered before the stream appends to it.
                await self._log.mount(widget)
                self._stream = AssistantMessage.get_stream(widget)
            await self._stream.write(ev.text)

        elif isinstance(ev, ToolStart):
            # Reasoning immediately before a call belongs in that call's hidden
            # detail rather than taking another line in the conversation.
            reasoning = self._reasoning
            await self._close_content()
            if reasoning is not None:
                await reasoning.remove()
            group = await self._enter_tool_group()
            self._tool_entries[ev.id] = await group.add_call(ev.name, ev.args, reasoning)

        elif isinstance(ev, TodosUpdate):
            await self.close()
            self._log.show_todos(ev.todos)

        elif isinstance(ev, ToolEnd):
            entry = self._tool_entries.pop(ev.id, None)
            if entry is None:
                # Tolerate an incomplete/old log with a result but no call.
                self._log.add_tool_result(ev.result, label=ev.name, denied=ev.denied)
            else:
                label = f"{ev.name} {entry.summary}"
                await entry.finish(ev.result, label=label, denied=ev.denied)
                if self._tool_group is not None:
                    self._tool_group.refresh_header()

        elif isinstance(ev, ContextCompacted):
            await self.close()
            self._log.add(
                Content.from_markup(
                    "[$text-muted]Context compacted: $before → ~$after tokens[/]",
                    before=f"{ev.tokens_before:,}",
                    after=f"{ev.tokens_after:,}",
                )
            )

        elif isinstance(ev, ContextCompactionFailed):
            await self.close()
            self._log.add(
                Content.from_markup(
                    "[$text-warning]Context compaction failed; continuing without it: $error[/]",
                    error=ev.error,
                )
            )

        elif isinstance(ev, ModelRetry):
            await self.close()
            self._log.add(
                Content.from_markup(
                    "[$text-warning]$error — retrying in $delay s ($attempt/$total)[/]",
                    error=ev.error,
                    delay=f"{ev.delay:g}",
                    attempt=str(ev.attempt),
                    total=str(ev.max_attempts - 1),
                )
            )

    async def _enter_tool_group(self) -> ToolGroup:
        if self._tool_group is None:
            self._tool_group = ToolGroup(self._agent.cwd)
            await self._log.mount(self._tool_group)
        return self._tool_group

    def _finish_tool_group(self) -> None:
        self._tool_group = None

    async def _close_content(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
        if self._reasoning is not None and self._agent.config.show_reasoning:
            # Fold the live stream now that the block is over; blocks the user
            # clicked open themselves are left alone.
            self._reasoning.collapse()
        self._reasoning = None
        self._reasoning_buf = ""

    async def close(self) -> None:
        """End current content and make the next call start a fresh group."""
        await self._close_content()
        self._finish_tool_group()
