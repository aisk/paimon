"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import asyncio
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Sequence

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

from . import agents, aside, compaction, retry, tools
from .config import Config
from .llm import NoModelError, build_model, user_agent
from .mentions import expand_mentions
from .prompt import build_system_prompt
from .skills import Skill, SkillDiagnostic, discover_skills, expand_skill_command
from .session import (
    Session,
    SessionIncompleteError,
    agents_message,
    agents_text,
    is_agents_message,
    is_summary_message,
)

# Transient compaction failures tolerated in one turn before it is left off.
_MAX_COMPACTION_FAILURES = 3


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
class RequestStats:
    """Output speed and cache usage of one finished model request, from
    provider-reported usage. ``input_tokens`` includes the cached portion
    (pydantic-ai normalizes providers that report them separately), so the
    cache hit rate is ``cache_read_tokens / input_tokens``. Providers that
    do not report caching leave both cache fields at zero.
    """

    output_tokens: int
    seconds: float
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ToolBudgetExhausted:
    """The turn hit its tool-call budget: the remaining calls were refused
    with an explicit result and the turn ends without another model request.
    """

    limit: int


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


@dataclass
class AgentsNotice:
    """A status line about the agents this session started.

    Not replay-only: it is written into the history at the top of a turn, so
    it is both yielded live and rebuilt when the session is resumed.
    """

    text: str


# Everything ``Agent.run`` and ``replay_events`` can yield. Renderers dispatch
# on isinstance; the alias exists so a type checker can flag an unhandled one.
AgentEvent = (
    TextDelta | ReasoningDelta | ToolStart | ToolEnd | TodosUpdate
    | SessionHandoff | RequestStats | ToolBudgetExhausted | TurnEnd
    | ContextCompacted | ContextCompactionFailed | ModelRetry | UserInput
    | CompactionNotice | AgentsNotice
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
    Tools run serially, so each call replays as its ToolStart immediately
    followed by its ToolEnd (looked up in the next request), matching the live
    order — not all starts of a batch first. The persisted ``outcome`` field
    restores denied styling.
    """
    events: list[AgentEvent] = []
    consumed: set[str] = set()  # tool returns already replayed with their call
    for index, message in enumerate(messages):
        if is_summary_message(message):
            events.append(CompactionNotice())
            continue
        if is_agents_message(message):
            events.append(AgentsNotice(agents_text(message)))
            continue
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content:
                    events.append(UserInput(part.content))
                elif (isinstance(part, ToolReturnPart)
                      and part.tool_call_id not in consumed
                      and part.tool_name != "write_todos"):
                    # A return whose call is gone (compaction cut mid-turn):
                    # better an unpaired result than a silent hole.
                    events.append(ToolEnd(part.tool_call_id, part.tool_name,
                                          str(part.content or "(no output)"),
                                          denied=part.outcome == "denied"))
        elif isinstance(message, ModelResponse):
            following = messages[index + 1] if index + 1 < len(messages) else None
            returns: dict[str, ToolReturnPart] = {}
            if isinstance(following, ModelRequest):
                returns = {part.tool_call_id: part for part in following.parts
                           if isinstance(part, ToolReturnPart)}
            for part in message.parts:
                if isinstance(part, ThinkingPart) and part.content:
                    events.append(ReasoningDelta(part.content))
                elif isinstance(part, TextPart) and part.content:
                    events.append(TextDelta(part.content))
                elif isinstance(part, ToolCallPart):
                    args = _parse_args(part.args)
                    result = returns.get(part.tool_call_id)
                    if result is not None:
                        consumed.add(part.tool_call_id)
                    if part.tool_name == "write_todos":
                        todos = tools.normalize_todos(args.get("todos"))
                        # Live, a write_todos that actually ran yields only a
                        # TodosUpdate; one that was rejected or never executed
                        # yields a normal failed call.
                        if todos is not None and (result is None or result.outcome == "success"):
                            events.append(TodosUpdate(todos))
                            continue
                    events.append(ToolStart(part.tool_call_id, part.tool_name, args))
                    if result is not None:
                        events.append(ToolEnd(part.tool_call_id, part.tool_name,
                                              str(result.content or "(no output)"),
                                              denied=result.outcome == "denied"))
    return events


def _strip_foreign_thinking(history: list[ModelMessage], model: Model) -> list[ModelMessage]:
    """Adapt thinking parts produced by a different provider/model.

    Preserved thinking is only valid replayed verbatim to the model that
    produced it, so a foreign message's readable thinking is converted to a
    plain text block — the rationale it carries is still context — while its
    signatures and encrypted/redacted payloads are dropped, never sent to a
    provider that did not produce them (cf. pi's transformMessages, which
    converts rather than deletes readable thinking).

    Only the request being built is touched; the persisted history keeps the
    original parts.
    """
    current = (model.system, model.model_name)
    sanitized: list[ModelMessage] = []
    for message in history:
        if (isinstance(message, ModelResponse)
                and (message.provider_name, message.model_name) != current
                and any(isinstance(part, ThinkingPart) for part in message.parts)):
            parts = []
            for part in message.parts:
                if not isinstance(part, ThinkingPart):
                    parts.append(part)
                elif part.content:
                    parts.append(TextPart(content=f"<thinking>\n{part.content}\n</thinking>"))
                # else: encrypted or redacted thinking with nothing readable
            if not parts:
                continue  # nothing a different provider can use
            message = dataclasses.replace(message, parts=parts)
        sanitized.append(message)
    return sanitized


# Re-exported so UI code can keep importing it from here.
ConfirmFn = tools.ConfirmFn

# Takes the messages a user queued while a turn was already running, clearing
# the queue as it goes. The loop calls it before every model request, so what
# it returns reaches the model at the next step rather than the next turn.
PendingFn = Callable[[], list[str]]


class Agent:
    """One conversation against one session.

    Built through :meth:`open`, which is where the session file is created or
    resumed and its lock taken; the constructor itself only wires up state a
    caller already holds, so it can neither fail nor leave a lock behind.

    An agent holds its session until :meth:`close`, which ``with`` calls on the
    way out::

        with Agent.open(cwd=path) as agent:
            async for event in agent.run("..."):
                ...
    """

    def __init__(self, session: Session, system_prompt: str, *, cwd: Optional[Path] = None,
                 confirm: Optional[ConfirmFn] = None, mode: str = "yolo",
                 config: Optional[Config] = None,
                 toolset: Optional[dict[str, tools.Tool]] = None,
                 model_override: Optional[str] = None,
                 skills: Sequence[Skill] = (),
                 skill_diagnostics: Sequence[SkillDiagnostic] = (),
                 agent_types: Sequence[agents.AgentType] = (),
                 agent_type_diagnostics: Sequence[SkillDiagnostic] = ()):
        self.cwd = Path(cwd or Path.cwd())
        # The skills listed in the system prompt and expandable by /skill:name,
        # with whatever discovery had to complain about (for the UI to show).
        self.skills = list(skills)
        self.skill_diagnostics = list(skill_diagnostics)
        # The agent types spawn_agent may name, discovered like skills; the
        # launcher reads them off whoever called spawn_agent.
        self.agent_types = list(agent_types)
        self.agent_type_diagnostics = list(agent_type_diagnostics)
        self.confirm = confirm
        self.mode = mode
        self.config = config or Config.load()
        # Per-agent model choice. One Config instance is shared by every agent
        # in the process, so writing config.model would repoint all of them;
        # this overrides the model for this agent alone, credentials included
        # (they come from the config either way).
        self.model_override = model_override
        # Set by the UI when this agent may start and talk to other agents.
        # None everywhere else (headless, tests), where the agent tools refuse
        # rather than pretend.
        self.supervisor = None
        # Set by the UI too: the hook the loop pulls queued user messages from.
        # None where nobody can type while a turn runs (headless, tests).
        self.pending: Optional[PendingFn] = None
        # Per-agent tool state, kept off the tool functions so one agent's
        # shell overflow files stay invisible to the next one. The session
        # rides along for the history tools.
        self.tool_context = tools.ToolContext(session=session)
        self.todos: list[dict] = []
        self.session = session
        self.system_prompt = system_prompt
        self.history: list[ModelMessage] = session.messages()
        # This agent's tool set; None means everything in tools.REGISTRY.
        self.toolset = dict(tools.REGISTRY if toolset is None else toolset)
        # The registry's spawn_agent knows no types; this agent's copy lists
        # the ones it discovered, so the schema below advertises them. A
        # toolset without spawn_agent (subagents, headless) skips it whole.
        if "spawn_agent" in self.toolset and self.agent_types:
            self.toolset["spawn_agent"] = agents.spawn_tool_with_types(
                self.toolset["spawn_agent"], self.agent_types)
        self.tool_schemas = tools.schemas(self.toolset)
        self._tool_definitions = tools.definitions(self.toolset)
        self._cached_model: Optional[tuple[tuple, Model]] = None
        # (history length, provider-reported tokens) after the last completed
        # request: the authoritative context size at that point, used by
        # count_context_tokens so the chars/4 heuristic only covers what was
        # appended since. None until a request reports usage, and again after
        # compaction reshapes the history.
        self._usage_anchor: Optional[tuple[int, int]] = None

    @classmethod
    def open(cls, cwd: Optional[Path] = None, *, session: Optional[Session] = None,
             confirm: Optional[ConfirmFn] = None, mode: str = "yolo",
             config: Optional[Config] = None,
             append_system_prompt: Optional[str] = None,
             toolset: Optional[dict[str, tools.Tool]] = None,
             model_override: Optional[str] = None,
             parent_session_id: Optional[str] = None) -> "Agent":
        """Start a new session, or resume ``session``, and take its lock.

        ``append_system_prompt`` is added to the end of a new session's system
        prompt and persisted with it, so a resumed session keeps it. Resuming
        with it set raises ``ValueError``: the persisted role is immutable.
        The dynamic parts of the prompt (date, environment, AGENTS.md, the
        skill list) are rebuilt on resume — a session picked up months later
        must not keep telling the model the old date or stale project rules —
        and when they changed, the rebuilt snapshot is appended to the log, so
        it always records the prompt each turn actually ran with.
        ``parent_session_id`` marks the new session as a subagent's, which
        keeps it out of the session listings its parent shows up in.

        Raises ``SessionBusyError`` when the session is already open (here or
        in another process) and ``SessionIncompleteError`` when a resumed log
        has no system prompt snapshot — both ``SessionError``, and neither
        leaves a lock held.
        """
        cwd = Path(cwd or Path.cwd())
        if session is not None and append_system_prompt:
            raise ValueError("append_system_prompt only applies to a new session")
        config = config or Config.load()
        skills, skill_diagnostics = discover_skills(
            cwd, extra_paths=config.skills, include_defaults=config.include_default_skills)
        agent_types, agent_type_diagnostics = agents.discover_agent_types(
            cwd, extra_paths=config.agents, include_defaults=config.include_default_agents)
        is_new = session is None
        if session is None:
            session = Session.create(cwd, parent_session_id)
        session.lock()
        # Everything after the lock can fail — a full disk while writing the
        # prompt, a message the current pydantic-ai cannot parse — and no
        # Agent is returned to unlock it, so the lock is released here.
        try:
            if is_new:
                appended = append_system_prompt.strip() if append_system_prompt else None
                system_prompt = build_system_prompt(cwd, skills)
                if appended:
                    system_prompt += f"\n\n{appended}"
                session.append_system_prompt(system_prompt, appended=appended)
            else:
                session.require_supported_format()
                stored, appended = session.system_prompt_parts()
                if stored is None:
                    raise SessionIncompleteError("Session does not contain a persisted system prompt")
                system_prompt = build_system_prompt(cwd, skills)
                if appended:
                    system_prompt += f"\n\n{appended}"
                if system_prompt != stored:
                    session.append_system_prompt(system_prompt, appended=appended)
            return cls(session, system_prompt, cwd=cwd, confirm=confirm, mode=mode,
                       config=config, toolset=toolset, model_override=model_override,
                       skills=skills, skill_diagnostics=skill_diagnostics,
                       agent_types=agent_types,
                       agent_type_diagnostics=agent_type_diagnostics)
        except BaseException:
            session.unlock()
            raise

    def close(self) -> None:
        """Release the session, so another agent may open it. Idempotent."""
        self.session.unlock()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Backstop for a caller that used neither ``with`` nor close(): the
        # claim is refcounted in this process, so a forgotten agent would keep
        # its session unopenable for as long as the process lives. Runs at
        # interpreter shutdown too, where anything may already be torn down.
        try:
            session = getattr(self, "session", None)
            if session is not None:
                session.unlock()
        except Exception:
            pass

    @property
    def model_name(self) -> Optional[str]:
        """The model this agent talks to: its own override, else the config's."""
        return self.model_override or self.config.model

    def _model(self) -> Model:
        """The configured model, rebuilt when login changes the config."""
        name = self.model_name
        if not name:
            raise NoModelError("No model configured; log in first")
        api_base, api_key = self.config.provider_auth(name)
        key = (name, api_base, api_key)
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
        window = compaction.context_window(self.model_name,
                                           self.config.compaction_context_window)
        if not force:
            if not self.config.compaction_enabled:
                return None
            # Nothing to compare against, so counting would be wasted work: an
            # unknown window disables auto-compaction outright.
            if window is None:
                return None
            tokens_before = await self.count_context_tokens()
            if not compaction.should_compact(tokens_before, window, self.config.compaction_reserve_tokens):
                return None
        else:
            tokens_before = await self.count_context_tokens()

        result = await compaction.compact(
            self.history,
            model=self._model(),
            keep_recent_tokens=self.config.compaction_keep_recent_tokens,
            tokens_before=tokens_before,
            tool_schemas=self.tool_schemas,
            system_prompt=self.system_prompt,
            window=window,
        )
        if result is None:
            return None

        # Mirrors how Session.messages() replays a compaction record, so the
        # append-message invariant (see _append_message) still holds afterwards.
        self.session.append_compaction(result.summary, result.kept_messages, result.tokens_before)
        self.history = result.messages
        # The provider's count described the old context; drop the anchor so
        # the fresh estimate below covers the compacted shape.
        self._usage_anchor = None
        result.tokens_after = await self.count_context_tokens()
        return result

    async def count_context_tokens(self) -> int:
        """Tokens of everything the next request would send.

        Anchored on the provider-reported usage of the last completed request
        when one exists — the provider's own count is authoritative and
        already includes the system prompt and tool schemas — with the chars/4
        heuristic applied only to messages appended since. Without an anchor
        the heuristic covers everything.

        Off the event loop: the count serializes history, which is hundreds of
        kilobytes on a long session.  It runs at the top of every model step,
        and every agent shares one loop, so counting inline stalls every other
        agent's streaming output for as long as it takes.
        """
        anchor = self._usage_anchor
        if anchor is not None and anchor[0] <= len(self.history):
            index, tokens = anchor
            suffix = list(self.history[index:])
            if not suffix:
                return tokens
            return tokens + await asyncio.to_thread(compaction.count_tokens, suffix)
        return await asyncio.to_thread(
            compaction.count_tokens, list(self.history), self.tool_schemas, self.system_prompt)

    async def compact_now(self) -> Optional[compaction.CompactionResult]:
        """Compact on demand; None when the history is too short to be worth it."""
        return await self._maybe_compact(force=True)

    async def ask_aside(self, question: str, *,
                        instructions: str = aside.DEFAULT_INSTRUCTIONS,
                        max_tokens: int = aside.DEFAULT_MAX_TOKENS) -> AsyncIterator[str]:
        """Ask a question over this agent's context, yielding the answer as text.

        Read-only: neither the question nor the answer enters ``self.history``
        or the session log, and no compaction is triggered. That is what lets
        it be asked while a turn is already waiting on the model: it takes a
        snapshot and opens a request of its own rather than joining the one in
        flight. A cancelled aside leaves nothing behind either.

        The history it sends is whatever exists when the first delta is asked
        for, so a turn running alongside it may add messages the answer never
        saw.
        """
        model = self._model()
        context = _strip_foreign_thinking(aside.usable_history(self.history), model)
        async for delta in aside.stream(model, self.system_prompt, context, question,
                                        instructions=instructions,
                                        tool_definitions=self._tool_definitions,
                                        max_tokens=max_tokens):
            yield delta

    # ---- Tools the agent loop runs itself ----------------------------------
    # These mutate agent-held state or end the turn, so they cannot go through
    # the stateless tools.run_tool. Each handler fills ``slot`` with the tool
    # result, calls ``persist()`` after every slot update, and yields the
    # events to forward; yielding SessionHandoff ends the whole turn.

    async def _run_write_todos(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                               persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        # Reports itself as TodosUpdate alone — no ToolStart/ToolEnd — which is
        # also the shape replay_events produces for it, so no renderer has to
        # special-case the name. A malformed argument is the exception: it
        # renders as a normal failed tool call, live and on replay alike.
        todos = tools.normalize_todos(args.get("todos"))
        if todos is None:
            yield ToolStart(call.tool_call_id, call.tool_name, args)
            slot.content = ("Error: 'todos' must be an array of "
                            "{content, status} objects.")
            slot.outcome = "failed"
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)
            return
        self.todos = todos
        slot.content = tools.render_todos(self.todos)
        slot.outcome = "success"
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
            slot.outcome = "failed"
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)
            return
        if not await self._permitted(call.tool_name, args):
            slot.content = "User denied this operation."
            slot.outcome = "denied"
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content, denied=True)
            return
        slot.content = ("Handoff accepted: this session ended here and a new "
                        "session continued with the provided prompt.")
        slot.outcome = "success"
        persist()
        yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)
        yield SessionHandoff(prompt_text)

    async def _permitted(self, name: str, args: dict) -> bool:
        """Gate a tool the loop runs itself, the way run_tool gates the rest.

        The enforcement point for everything in _AGENT_HANDLED: without a
        confirm hook a call that needs one is denied, so a headless agent
        cannot walk around the permission mode here either.
        """
        if tools.gate(name, args, self.mode, self.cwd, self.toolset,
                      safe_commands=self.config.safe_commands,
                      ctx=self.tool_context) != "confirm":
            return True
        return await self.confirm(name, args) if self.confirm else False

    async def _run_supervised(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                              persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        """The job tools: starting agents and commands, and reporting on both.

        They act on the pool of jobs the UI is running, which no stateless tool
        function can reach, and the supervisor is the one thing that knows
        whether a given job is busy \u2014 so they are dispatched from here. Most of
        them are not gated at all (they only reach jobs this same agent
        started); run_background is, because it starts a process.
        """
        yield ToolStart(call.tool_call_id, call.tool_name, args)
        if self.supervisor is None:
            slot.content = ("Error: this only works in the interactive UI; "
                            "do the work yourself instead.")
            slot.outcome = "failed"
        elif not await self._permitted(call.tool_name, args):
            slot.content = "User denied this operation."
            slot.outcome = "denied"
            persist()
            yield ToolEnd(call.tool_call_id, call.tool_name, slot.content, denied=True)
            return
        else:
            slot.content = await self.supervisor.handle(call.tool_name, args, caller=self)
            slot.outcome = "success"
        persist()
        yield ToolEnd(call.tool_call_id, call.tool_name, slot.content)

    _AGENT_HANDLED = {
        "write_todos": _run_write_todos,
        "start_new_session": _run_start_new_session,
        "spawn_agent": _run_supervised,
        "send_to_agent": _run_supervised,
        "run_background": _run_supervised,
        "read_job": _run_supervised,
        "wait_for_job": _run_supervised,
        "stop_job": _run_supervised,
    }

    def expand_input(self, text: str) -> str:
        """What a typed message becomes: a /skill:name command expands to the
        skill's content, then @path mentions are inlined."""
        def mentions(part: str) -> str:
            return expand_mentions(part, self.cwd)

        expanded = expand_skill_command(text, self.skills, expand_args=mentions)
        # Unchanged (the same object back) means it was not a skill command.
        return mentions(text) if expanded is text else expanded

    async def run(self, user_input: str, *, expand: bool = True,
                  max_tool_calls: Optional[int] = None) -> AsyncIterator[AgentEvent]:
        """Run one user turn to completion, yielding events along the way.

        ``expand=False`` skips @path expansion, for callers that assembled the
        prompt themselves and must not have unrelated text rewritten (piped
        stdin, where a line like ``@foo.py`` is data rather than a mention).

        ``max_tool_calls`` bounds the tool calls this turn may execute. It is
        enforced here, where every ToolCallPart is dispatched — including the
        agent-handled tools like write_todos that produce no ToolStart — so no
        tool can slip past the budget. On the budget every remaining call gets
        an explicit refusal persisted as its result, ToolBudgetExhausted is
        yielded and the turn ends.
        """
        prompt = self.expand_input(user_input) if expand else user_input
        # Agents this session started report in here, at the top of the next
        # turn, rather than by interrupting whatever the user is typing. It has
        # to be a persisted message: an event the model never sees would defeat
        # the point, which is to get it to call read_job.
        summary = self.supervisor.status_summary(self) if self.supervisor is not None else None
        if summary:
            self._append_message(agents_message(summary))
            yield AgentsNotice(summary)
        self._append_message(ModelRequest(parts=[UserPromptPart(content=prompt)]))

        # Every turn leaves exactly one terminal ``turn_end`` record in the
        # session: success, error (with the exception), max_tool_calls or
        # interrupted. The record is not a message, so it reaches audits and
        # ``paimon log`` without an error or a partial answer ever being
        # replayed to the model as a normal assistant response.
        outcome_recorded = False

        def record_outcome(outcome: str, *, error: Optional[str] = None,
                           partial_text: Optional[str] = None) -> None:
            nonlocal outcome_recorded
            if outcome_recorded:
                return
            outcome_recorded = True
            self.session.append_meta("turn_end", outcome=outcome, error=error,
                                     partial_text=partial_text)

        try:
            async for event in self._run(record_outcome, max_tool_calls):
                yield event
            # A session handoff is the one path that ends the loop without
            # its own record; it ended the turn deliberately.
            record_outcome("success")
        except asyncio.CancelledError:
            record_outcome("interrupted")
            raise
        except GeneratorExit:
            # The consumer abandoned the turn mid-iteration.
            record_outcome("interrupted")
            raise
        except Exception as exc:
            record_outcome("error", error=f"{type(exc).__name__}: {exc}")
            raise

    async def _run(self, record_outcome: Callable[..., None],
                   max_tool_calls: Optional[int]) -> AsyncIterator[AgentEvent]:
        """The model/tool loop of one turn; ``run`` wraps it to guarantee the
        terminal ``turn_end`` record whatever way the turn ends."""
        # A compaction that failed for a transient reason is retried on the
        # next step of this turn — the context only keeps growing, so giving
        # up on the first rate limit disables the safety net exactly when it
        # is needed. Anything else, and repeated transient failures, stop it
        # for the rest of the turn instead of paying for it every step.
        compaction_off = False
        compaction_failures = 0
        calls_made = 0  # every dispatched ToolCallPart counts, whatever its kind
        # A provider context-overflow error triggers one forced compaction and
        # one retry per turn; a second overflow means compaction cannot make
        # the request fit, and retrying again would loop.
        overflow_retried = False

        while True:
            # Messages the user queued while this turn was already running.
            # Appended as their own request so the model sees them at the next
            # step, without touching the stream that just ended or the tool
            # results already recorded. Before compaction, so they count
            # towards the context check and survive a history replacement.
            for queued in (self.pending() if self.pending is not None else []):
                self._append_message(ModelRequest(
                    parts=[UserPromptPart(content=self.expand_input(queued))]))
                yield UserInput(queued)

            if not compaction_off:
                try:
                    compacted = await self._maybe_compact()
                except Exception as exc:  # noqa: BLE001 - the normal request may still fit
                    compaction_failures += 1
                    compaction_off = (not retry.is_transient(exc)
                                      or compaction_failures >= _MAX_COMPACTION_FAILURES)
                    self.session.append_meta("compaction_failed", error=str(exc))
                    yield ContextCompactionFailed(str(exc))
                else:
                    compaction_failures = 0
                    if compacted:
                        yield ContextCompacted(compacted.tokens_before, compacted.tokens_after)

            model = self._model()

            def build_request() -> list[ModelMessage]:
                return [
                    ModelRequest(parts=[SystemPromptPart(content=self.system_prompt)]),
                    *_strip_foreign_thinking(self.history, model),  # noqa: B023 — rebuilt each step
                ]

            request_messages = build_request()
            parameters = ModelRequestParameters(
                function_tools=self._tool_definitions, allow_text_output=True
            )

            content = ""  # accumulated text, kept if the stream is interrupted
            attempt = 0
            stats: Optional[RequestStats] = None

            while True:
                started = False  # this attempt has yielded something to the caller
                first_event_at: Optional[float] = None
                try:
                    async with model_request_stream(
                        model,
                        request_messages,
                        model_settings={"extra_headers": {"User-Agent": user_agent()}},
                        model_request_parameters=parameters,
                    ) as stream:
                        async for event in stream:
                            if first_event_at is None:
                                first_event_at = time.monotonic()
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
                        if first_event_at is not None:
                            elapsed = time.monotonic() - first_event_at
                            usage = response.usage
                            if usage.output_tokens and elapsed > 0:
                                stats = RequestStats(usage.output_tokens, elapsed,
                                                     usage.input_tokens,
                                                     usage.cache_read_tokens,
                                                     usage.cache_write_tokens)
                    break
                except asyncio.CancelledError:
                    # Interrupted mid-stream: the partial text is persisted in
                    # the turn_end record for the log, but no ModelResponse is
                    # appended — a truncated answer replayed as a completed
                    # one misleads both the resumed UI and the model, and a
                    # history ending on the user request stays resumable.
                    record_outcome("interrupted", partial_text=content or None)
                    raise
                except Exception as exc:  # noqa: BLE001 — classified by retry.is_transient
                    attempt += 1
                    if (not started and not overflow_retried
                            and retry.is_context_overflow(exc)):
                        # The provider says the request exceeded the window:
                        # compact and retry once, with an audit record of the
                        # overflow that triggered it.
                        overflow_retried = True
                        self.session.append_meta("context_overflow",
                                                 error=f"{type(exc).__name__}: {exc}")
                        try:
                            compacted = await self._maybe_compact(force=True)
                        except Exception as compact_exc:  # noqa: BLE001 — original error still raises below
                            self.session.append_meta("compaction_failed",
                                                     error=str(compact_exc))
                            compacted = None
                        if compacted:
                            yield ContextCompacted(compacted.tokens_before,
                                                   compacted.tokens_after)
                            request_messages = build_request()
                            continue
                    if started or attempt >= retry.MAX_ATTEMPTS or not retry.is_transient(exc):
                        # Recorded here rather than in run()'s outer boundary
                        # because only this scope still holds the partial text
                        # the user already saw streaming.
                        record_outcome("error", error=f"{type(exc).__name__}: {exc}",
                                       partial_text=content or None)
                        raise
                    delay = retry.backoff(attempt)
                    self.session.append_meta("model_retry", attempt=attempt,
                                             error=retry.describe(exc))
                    yield ModelRetry(attempt, retry.MAX_ATTEMPTS, delay, retry.describe(exc))
                    await asyncio.sleep(delay)

            self._append_message(response)
            usage = response.usage
            if usage.input_tokens:
                # The provider's own count of this request plus its response —
                # the authoritative context size at this history length.
                self._usage_anchor = (len(self.history),
                                      usage.input_tokens + (usage.output_tokens or 0))
            if stats is not None:
                yield stats

            calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
            if not calls:
                record_outcome("success")
                yield TurnEnd()
                return

            # Pre-seed a tool result for every call up front, so even if we are
            # interrupted mid-execution the history never has a dangling tool call
            # (the API requires each tool_call_id to be answered). Unfinished ones
            # keep this placeholder.
            returns = [
                ToolReturnPart(tool_name=call.tool_name, content="Interrupted by user.",
                               tool_call_id=call.tool_call_id, outcome="interrupted")
                for call in calls
            ]
            tool_request = ModelRequest(parts=returns)
            record_id = self._append_message(tool_request)

            def persist() -> None:
                """Re-persist the tool request with the slots filled so far."""
                self._replace_message(record_id, tool_request)

            budget_hit = False
            for slot, call in zip(returns, calls):
                args = _parse_args(call.args)
                name = call.tool_name

                # The budget check comes before any dispatch, agent-handled
                # tools included: a refused call never executes, and its slot
                # records why instead of a generic interrupted placeholder.
                if max_tool_calls is not None and calls_made >= max_tool_calls:
                    budget_hit = True
                    yield ToolStart(call.tool_call_id, name, args)
                    slot.content = (f"Not executed: the run reached its tool call "
                                    f"budget (max_tool_calls={max_tool_calls}).")
                    persist()
                    yield ToolEnd(call.tool_call_id, name, slot.content)
                    continue
                calls_made += 1

                # A name outside this agent's tool set is rejected before the
                # agent-handled table below, so excluding write_todos or
                # start_new_session from a toolset actually disables them.
                if name not in self.toolset:
                    yield ToolStart(call.tool_call_id, name, args)
                    slot.content = f"Error: unknown tool {name!r}"
                    slot.outcome = "failed"
                    persist()
                    yield ToolEnd(call.tool_call_id, name, slot.content)
                    continue

                # One schema validation for every call — before gating and
                # before the agent-handled dispatch below — so a malformed
                # argument is a tool error the model can react to, never an
                # exception ending the turn (TOOLS-1).
                invalid = tools.validate_args(name, args, self.toolset)
                if invalid is not None:
                    yield ToolStart(call.tool_call_id, name, args)
                    slot.content = f"Error: invalid arguments for {name}: {invalid}"
                    slot.outcome = "failed"
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
                                                      safe_commands=self.config.safe_commands,
                                                      ctx=self.tool_context)

                slot.content = result
                slot.outcome = "denied" if denied else "success"
                persist()
                yield ToolEnd(call.tool_call_id, name, result, denied=denied)

            if budget_hit:
                record_outcome("max_tool_calls")
                yield ToolBudgetExhausted(max_tool_calls)
                return
            # loop again so the model can react to tool results
