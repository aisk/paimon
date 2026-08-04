"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

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

from . import compaction, retry, tools
from .config import Config
from .llm import build_model
from .mentions import expand_mentions
from .prompt import build_system_prompt
from .session import Session, SessionIncompleteError, is_summary_message


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
class SessionHandoff:
    """The user approved start_new_session: the turn ends here and the UI
    should open a fresh session whose first user message is ``prompt``.
    Never fires without a confirm hook (headless), where the call is denied.
    """

    prompt: str


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


@dataclass
class ModelRetry:
    """A transient model failure is about to be retried after ``delay`` seconds."""

    attempt: int
    max_attempts: int
    delay: float
    error: str


# ---- Replay-only events (history has no deltas, so these mark structure) ----


@dataclass
class UserInput:
    text: str


@dataclass
class CompactionNotice:
    """A compaction checkpoint encountered while replaying history."""


# Everything ``Agent.run`` and ``replay_events`` can yield. Renderers dispatch
# on isinstance; the alias exists so a type checker can flag an unhandled one.
AgentEvent = (
    TextDelta | ReasoningDelta | ToolStart | ToolEnd | TodosUpdate
    | SessionHandoff | TurnEnd | ContextCompacted | ContextCompactionFailed
    | ModelRetry | UserInput | CompactionNotice
)


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


def replay_events(messages: list[ModelMessage]) -> list[AgentEvent]:
    """Persisted messages replayed as the events a live ``Agent.run`` yields.

    Lets a UI render resumed history through the same code path as live turns.
    """
    events: list[AgentEvent] = []
    for message in messages:
        if is_summary_message(message):
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


# Re-exported so UI code can keep importing it from here.
ConfirmFn = tools.ConfirmFn


class Agent:
    """One conversation against one session.

    Built through :meth:`open`, which is where the session file is created or
    resumed and its lock taken; the constructor itself only wires up state a
    caller already holds, so it can neither fail nor leave a lock behind.
    """

    def __init__(self, session: Session, system_prompt: str, *, cwd: Optional[Path] = None,
                 confirm: Optional[ConfirmFn] = None, mode: str = "read",
                 config: Optional[Config] = None,
                 toolset: Optional[dict[str, tools.Tool]] = None):
        self.cwd = Path(cwd or Path.cwd())
        self.confirm = confirm
        self.mode = mode
        self.config = config or Config.load()
        self.todos: list[dict] = []
        self.session = session
        self.system_prompt = system_prompt
        self.history: list[ModelMessage] = session.messages()
        # This agent's tool set; None means everything in tools.REGISTRY.
        self.toolset = dict(tools.REGISTRY if toolset is None else toolset)
        self.tool_schemas = tools.schemas(self.toolset)
        self._tool_definitions = tools.definitions(self.toolset)
        self._cached_model: Optional[tuple[tuple, Model]] = None

    @classmethod
    def open(cls, cwd: Optional[Path] = None, *, session: Optional[Session] = None,
             confirm: Optional[ConfirmFn] = None, mode: str = "read",
             config: Optional[Config] = None,
             append_system_prompt: Optional[str] = None,
             toolset: Optional[dict[str, tools.Tool]] = None) -> "Agent":
        """Start a new session, or resume ``session``, and take its lock.

        ``append_system_prompt`` is added to the end of a new session's system
        prompt and persisted with it, so a resumed session keeps it. Resuming
        with it set raises ``ValueError``: the persisted prompt is immutable.

        Raises ``SessionBusyError`` when another process holds the session and
        ``SessionIncompleteError`` when a resumed log has no system prompt
        snapshot — both ``SessionError``, and neither leaves a lock held.
        """
        cwd = Path(cwd or Path.cwd())
        if session is not None and append_system_prompt:
            raise ValueError("append_system_prompt only applies to a new session")
        if session is None:
            session = Session.create(cwd)
            session.lock()
            system_prompt = build_system_prompt(cwd)
            if append_system_prompt:
                system_prompt += f"\n\n{append_system_prompt.strip()}"
            session.append_system_prompt(system_prompt)
        else:
            session.lock()
            system_prompt = session.system_prompt()
            if system_prompt is None:
                session.unlock()
                raise SessionIncompleteError("Session does not contain a persisted system prompt")
        return cls(session, system_prompt, cwd=cwd, confirm=confirm, mode=mode,
                   config=config, toolset=toolset)

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

    async def _maybe_compact(self, *, force: bool = False) -> Optional[compaction.CompactionResult]:
        """Compact the context if it is close to full.

        ``force`` is the manual path: the user asked for it, so neither the
        auto-compaction toggle nor the token threshold applies.  It still keeps
        the recent window, and so returns None on a history short enough that
        there is nothing to summarize.
        """
        if not force:
            if not self.config.compaction_enabled:
                return None
            window = compaction.context_window(self.config.model,
                                               self.config.compaction_context_window)
            tokens_before = compaction.count_tokens(self.history, self.tool_schemas)
            if not compaction.should_compact(tokens_before, window, self.config.compaction_reserve_tokens):
                return None
        else:
            tokens_before = compaction.count_tokens(self.history, self.tool_schemas)

        result = await compaction.compact(
            self.history,
            model=self._model(),
            keep_recent_tokens=self.config.compaction_keep_recent_tokens,
            tokens_before=tokens_before,
            tool_schemas=self.tool_schemas,
        )
        if result is None:
            return None

        # Mirrors how Session.messages() replays a compaction record, so the
        # append-message invariant (see _append_message) still holds afterwards.
        self.session.append_compaction(result.summary, result.kept_messages, result.tokens_before)
        self.history = result.messages
        result.tokens_after = compaction.count_tokens(self.history, self.tool_schemas)
        return result

    async def compact_now(self) -> Optional[compaction.CompactionResult]:
        """Compact on demand; None when the history is too short to be worth it."""
        return await self._maybe_compact(force=True)

    # ---- Tools the agent loop runs itself ----------------------------------
    # These mutate agent-held state or end the turn, so they cannot go through
    # the stateless tools.run_tool. Each handler fills ``slot`` with the tool
    # result, calls ``persist()`` after every slot update, and yields the
    # events to forward; yielding SessionHandoff ends the whole turn.

    async def _run_write_todos(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                               persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        # Reports itself as TodosUpdate alone — no ToolStart/ToolEnd — which is
        # also the shape replay_events produces for it, so no renderer has to
        # special-case the name.
        self.todos = args.get("todos") or []
        slot.content = tools.render_todos(self.todos)
        persist()
        yield TodosUpdate(list(self.todos))

    async def _run_start_new_session(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                                     persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        # On approval the turn ends without another model request: the whole
        # point is to stop spending tokens on this history. The result is
        # still written into its pre-seeded slot first, so the session stays
        # valid and resumable.
        yield ToolStart(call.tool_call_id, call.tool_name, args)
        prompt_text = str(args.get("prompt") or "").strip()
        if not prompt_text:
            slot.content = "Error: prompt is required."
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)
            return
        needs_confirm = tools.gate(call.tool_name, args, self.mode, self.cwd,
                                   self.toolset) == "confirm"
        allowed = not needs_confirm or (
            await self.confirm(call.tool_name, args) if self.confirm else False
        )
        if not allowed:
            slot.content = "User denied this operation."
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content, denied=True)
            return
        slot.content = ("Handoff accepted: this session ended here and a new "
                        "session continued with the provided prompt.")
        persist()
        yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)
        yield SessionHandoff(prompt_text)

    _AGENT_HANDLED = {
        "write_todos": _run_write_todos,
        "start_new_session": _run_start_new_session,
    }

    async def run(self, user_input: str, *, expand: bool = True) -> AsyncIterator[AgentEvent]:
        """Run one user turn to completion, yielding events along the way.

        ``expand=False`` skips @path expansion, for callers that assembled the
        prompt themselves and must not have unrelated text rewritten (piped
        stdin, where a line like ``@foo.py`` is data rather than a mention).
        """
        prompt = expand_mentions(user_input, self.cwd) if expand else user_input
        self._append_message(ModelRequest(parts=[UserPromptPart(content=prompt)]))
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
                function_tools=self._tool_definitions, allow_text_output=True
            )

            content = ""  # accumulated text, kept if the stream is interrupted
            attempt = 0

            while True:
                started = False  # this attempt has yielded something to the caller
                try:
                    async with model_request_stream(
                        model, request_messages, model_request_parameters=parameters
                    ) as stream:
                        async for event in stream:
                            if isinstance(event, PartStartEvent):
                                part = event.part
                                if isinstance(part, ThinkingPart) and part.content:
                                    started = True
                                    yield ReasoningDelta(part.content)
                                elif isinstance(part, TextPart) and part.content:
                                    started = True
                                    content += part.content
                                    yield TextDelta(part.content)
                            elif isinstance(event, PartDeltaEvent):
                                delta = event.delta
                                if isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                                    started = True
                                    yield ReasoningDelta(delta.content_delta)
                                elif isinstance(delta, TextPartDelta) and delta.content_delta:
                                    started = True
                                    content += delta.content_delta
                                    yield TextDelta(delta.content_delta)
                        response = stream.get()
                    break
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
                except Exception as exc:  # noqa: BLE001 — classified by retry.is_transient
                    attempt += 1
                    if started or attempt >= retry.MAX_ATTEMPTS or not retry.is_transient(exc):
                        raise
                    delay = retry.backoff(attempt)
                    yield ModelRetry(attempt, retry.MAX_ATTEMPTS, delay, retry.describe(exc))
                    await asyncio.sleep(delay)

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

            def persist() -> None:
                """Re-persist the tool request with the slots filled so far."""
                self._replace_message(record_id, tool_request)

            for slot, call in zip(returns, calls):
                args = _parse_args(call.args)
                name = call.tool_name

                # A name outside this agent's tool set is rejected before the
                # agent-handled table below, so excluding write_todos or
                # start_new_session from a toolset actually disables them.
                if name not in self.toolset:
                    yield ToolStart(call.tool_call_id, name, args)
                    slot.content = f"Error: unknown tool {name!r}"
                    persist()
                    yield ToolEnd(call.tool_call_id, name, slot.content)
                    continue

                handler = self._AGENT_HANDLED.get(name)
                if handler is not None:
                    async for event in handler(self, call, args, slot, persist):
                        yield event
                        if isinstance(event, SessionHandoff):
                            # Any later calls in this response keep their
                            # "Interrupted by user." placeholders.
                            return
                    continue

                yield ToolStart(call.tool_call_id, name, args)
                result, denied = await tools.run_tool(name, args, self.cwd, self.mode,
                                                      self.confirm, self.toolset,
                                                      safe_commands=self.config.safe_commands)

                slot.content = result
                persist()
                yield ToolEnd(call.tool_call_id, name, result, denied=denied)
            # loop again so the model can react to tool results
