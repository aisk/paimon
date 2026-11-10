"""The agent loop: stream from the LLM, run tool calls, repeat until done.

``Agent.run`` is UI-agnostic: it yields typed events that a CLI or a TUI can
render however it likes.
"""

import asyncio
import contextlib
import dataclasses
import functools
import time
import weakref
from pathlib import Path
from typing import AsyncIterator, Callable, Sequence

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

from . import codemode, compaction, logtext, retry, review
from .config import Config, RunOptions
from .errors import PaimonError
from .events import (
    AgentEvent,
    ContextCompacted,
    ContextCompactionFailed,
    JobNotice,
    ModelRetry,
    ReasoningDelta,
    RequestStats,
    TextDelta,
    TodosUpdate,
    ToolBudgetExhausted,
    ToolEnd,
    ToolStart,
    TurnEnd,
    UserInput,
    parse_args,
)
from .jobs import Jobs
from .llm import (NoModelError, ask_once, build_model, models_for, request_settings,
                  signed_in_plans, split_model_string)
from .mentions import expand_mentions
from .prompt import build_system_prompt
from .skills import Skill, SkillDiagnostic, discover_skills, expand_skill_command
from .session import (
    Session,
    SessionIncompleteError,
    job_message,
    shell_message,
)
from .tools.base import Tool, ToolContext
from .tools.permissions import USER_DENIAL, ConfirmFn, authorize, confirm_for, run_tool
from .tools.registry import (
    REGISTRY,
    SUBAGENT_DENIED,
    AskFn,
    definitions,
    execute_tool,
    normalize_todos,
    render_todos,
    schemas,
    validate_args,
    without,
)
from .tools.result import ToolError, ToolResult
from .tools.shell import user_shell

# Transient compaction failures tolerated in one turn before it is left off.
_MAX_COMPACTION_FAILURES = 3

_RECAP_MAX_TOKENS = 2_048


class AgentBusyError(PaimonError):
    """A turn or a compaction was started while another one of the same
    agent was still in flight.

    Both rewrite the history and the session log, and two of them at once
    would interleave their writes into a log that no longer replays to what
    either of them meant.
    """


# Marked so the model answers it rather than taking it for the next
# instruction and resuming the work.
RECAP_PROMPT = (
    "[recap] The user stepped away and is coming back to a screen full of tool "
    "output. In two to five short lines, say where things stand: what this "
    "session is working toward overall, what was just done, and the obvious "
    "next step. Answer in the language the user has been writing in. No "
    "headings, no code blocks, no preamble, and do not offer to help.\n\n"
    "Where does this session stand right now?"
)


def _settle(slot: ToolReturnPart, result: ToolResult) -> None:
    """Fill a call's pre-seeded return part with how it went.

    The status is the part's outcome, so the log records it rather than
    leaving it to be guessed from the text. A detail that says more than the
    status ("exit 2") goes in the part's metadata, which is persisted but
    never sent to the model, so a replay renders it exactly as live did.
    """
    slot.content = result.text
    slot.outcome = result.status
    metadata = dict(slot.metadata) if isinstance(slot.metadata, dict) else {}
    metadata.pop("detail", None)
    if result.detail != ("" if result.ok else result.status):
        metadata["detail"] = result.detail
    slot.metadata = metadata or None


def _held(toolset: dict[str, Tool] | None, config: Config, options: RunOptions) -> dict[str, Tool]:
    """The tools an agent opened with ``toolset`` under ``config`` and
    ``options`` really holds.

    The switches are applied here rather than by each caller, so
    --no-web-search and code mode reach every agent opened with them: new
    sessions, forks, subagents.
    """
    held = dict(REGISTRY if toolset is None else toolset)
    if options.no_web_search:
        held.pop("web_search", None)
    if not options.code_mode(config):
        held.pop(codemode.NAME, None)
    if codemode.NAME in held:
        held[codemode.NAME] = codemode.tool(held)
    return held


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


def _failures_as_text(history: list[ModelMessage]) -> list[ModelMessage]:
    """The history with every failed tool return sent as a plain result.

    pydantic-ai sends a "failed" outcome to providers without a native error
    channel (every OpenAI-compatible one) as a JSON string, {"error": "..."},
    which turns a failing command's output into one line of escaped newlines.
    The result text already says what went wrong, so every provider gets it
    verbatim instead. That gives up Anthropic's is_error flag and Gemini's
    error field: an accepted trade for one behaviour across providers.

    Only the request being built is touched, on copies; the history and the
    log keep the outcome that really happened.
    """
    sent: list[ModelMessage] = []
    for message in history:
        if isinstance(message, ModelRequest) and any(
                isinstance(part, ToolReturnPart) and part.outcome == "failed"
                for part in message.parts):
            message = dataclasses.replace(message, parts=[
                dataclasses.replace(part, outcome="success")
                if isinstance(part, ToolReturnPart) and part.outcome == "failed" else part
                for part in message.parts])
        sent.append(message)
    return sent


# Takes what the user queued while a turn was already running, clearing the
# queue as it goes. The loop calls it before every model request, so what it
# returns reaches the model at the next step rather than the next turn. A
# plain string is a typed prompt and goes through the usual expansion; a
# ready-made request is one paimon assembled (a "!" shell run) and is appended
# verbatim, since @path in a command's output is output, not a mention.
PendingFn = Callable[[], list[str | ModelRequest]]


class Agent:
    """One conversation against one session.

    Built through :meth:`open`, which is where the session file is created or
    resumed and its lock taken; the constructor itself only wires up state a
    caller already holds, so it can neither fail nor leave a lock behind.

    An agent holds its session until :meth:`close`, which ``with`` calls on the
    way out::

        with Agent.open(cwd=path) as agent:
            async for event in agent.run_until_idle("..."):
                ...
    """

    def __init__(self, session: Session, system_prompt: str, *, cwd: Path | None = None,
                 confirm: ConfirmFn | None = None, ask: AskFn | None = None,
                 mode: str = "yolo", config: Config | None = None,
                 run_options: RunOptions | None = None,
                 toolset: dict[str, Tool] | None = None,
                 model_override: str | None = None,
                 skills: Sequence[Skill] = (),
                 skill_diagnostics: Sequence[SkillDiagnostic] = ()):
        self.cwd = Path(cwd or Path.cwd())
        # The skills listed in the system prompt and expandable by /skill:name,
        # with whatever discovery had to complain about (for the UI to show).
        self.skills = list(skills)
        self.skill_diagnostics = list(skill_diagnostics)
        self.confirm = confirm
        # How ask_user reaches the user. None where nobody is at the keyboard
        # (headless, tests): the tool then fails with a readable error.
        self.ask = ask
        # The agents and background commands this agent started, and what
        # became of them. It opens children through the two callables, and
        # knows nothing else about this agent. Through a weak reference: bound
        # methods would tie the agent into a cycle with its own jobs, and
        # __del__, the backstop that unlocks a forgotten agent's session,
        # would then wait for the cycle collector instead of the last reference.
        me = weakref.ref(self)
        self.jobs = Jobs(self.cwd, open_child=lambda model: me()._open_child(model),
                         check_model=lambda name: me()._built(name, None))
        self.mode = mode
        self.config = config or Config.load()
        # What the process was launched with on top of the config. Handed on
        # to every agent opened from this one.
        self.run_options = run_options or RunOptions()
        # Per-agent model choice over the configured one: the model the run
        # was launched on (--model), or the one a subagent was started on.
        # Credentials come from the config either way.
        self.model_override = model_override
        # Set by the UI: the hook the loop pulls queued user messages from.
        # None where nobody can type while a turn runs (headless, tests).
        self.pending: PendingFn | None = None
        # Per-agent tool state, kept off the tool functions so one agent's
        # shell overflow files stay invisible to the next one. The session
        # rides along for the history tools.
        self.tool_context = ToolContext(session=session)
        self.todos: list[dict] = []
        self.session = session
        self.system_prompt = system_prompt
        self.history: list[ModelMessage] = session.messages()
        # This agent's tool set; None means everything in REGISTRY.
        self.toolset = _held(toolset, self.config, self.run_options)
        self.tool_schemas = schemas(self.toolset)
        self._tool_definitions = definitions(self.toolset)
        self._cached_model: tuple[tuple, Model] | None = None
        self._cached_review_model: tuple[tuple, Model] | None = None
        # Refusals in a row from the auto-mode reviewer; see _review.
        self._review_blocks = 0
        # (history length, provider-reported tokens) after the last completed
        # request: the authoritative context size at that point, used by
        # count_context_tokens so the chars/4 heuristic only covers what was
        # appended since. None until a request reports usage, and again after
        # compaction reshapes the history.
        self._usage_anchor: tuple[int, int] | None = None
        # What is writing the history right now: "turn", "compaction" or None.
        # Only a guard: whoever drives the agent is expected to serialize the
        # two (the TUI's TurnDriver does), and this turns a caller that does
        # not into an error instead of a corrupted log.
        self._in_flight: str | None = None

    @classmethod
    def open(cls, cwd: Path | None = None, *, session: Session | None = None,
             confirm: ConfirmFn | None = None, ask: AskFn | None = None,
             mode: str = "yolo", config: Config | None = None,
             run_options: RunOptions | None = None,
             append_system_prompt: str | None = None,
             toolset: dict[str, Tool] | None = None,
             model_override: str | None = None,
             parent_session_id: str | None = None) -> "Agent":
        """Start a new session, or resume ``session``, and take its lock.

        ``run_options`` are the launch switches layered over ``config``;
        left out, nothing is overridden.

        ``append_system_prompt`` is added to the end of a new session's system
        prompt and persisted with it, so a resumed session keeps it. Resuming
        with it set raises ``ValueError``: the persisted role is immutable.
        The dynamic parts of the prompt (date, environment, AGENTS.md, the
        skill list) are rebuilt on resume — a session picked up months later
        must not keep telling the model the old date or stale project rules —
        and when they changed, the rebuilt snapshot is appended to the log, so
        it always records the prompt each turn actually ran with.
        ``parent_session_id`` marks the new session as a subagent's, which
        keeps it out of the session listings its parent shows up in. It
        applies to new sessions only and is ignored on resume, where the
        header already says.

        Raises ``SessionBusyError`` when the session is already open (here or
        in another process) and ``SessionIncompleteError`` when a resumed log
        has no system prompt snapshot — both ``SessionError``, and neither
        leaves a lock held.
        """
        cwd = Path(cwd or Path.cwd())
        if session is not None and append_system_prompt:
            raise ValueError("append_system_prompt only applies to a new session")
        config = config or Config.load()
        run_options = run_options or RunOptions()
        skills, skill_diagnostics = discover_skills(
            cwd, extra_paths=run_options.skill_paths(config),
            include_defaults=not run_options.no_skills)
        # The prompt only mentions tools this agent will actually be offered.
        tool_names = frozenset(_held(toolset, config, run_options))
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
                system_prompt = build_system_prompt(cwd, skills, tool_names)
                if appended:
                    system_prompt += f"\n\n{appended}"
                session.append_system_prompt(system_prompt, appended=appended)
            else:
                session.require_supported_format()
                stored, appended = session.system_prompt_parts()
                if stored is None:
                    raise SessionIncompleteError("Session does not contain a persisted system prompt")
                system_prompt = build_system_prompt(cwd, skills, tool_names)
                if appended:
                    system_prompt += f"\n\n{appended}"
                if system_prompt != stored:
                    session.append_system_prompt(system_prompt, appended=appended)
            return cls(session, system_prompt, cwd=cwd, confirm=confirm, ask=ask, mode=mode,
                       config=config, run_options=run_options, toolset=toolset,
                       model_override=model_override,
                       skills=skills, skill_diagnostics=skill_diagnostics)
        except BaseException:
            session.unlock()
            raise

    def close(self) -> None:
        """Stop what this agent started and release its session. Idempotent.

        The jobs go with it: their ids only mean anything inside this
        conversation, so nothing would ever hear from them again.
        """
        self.jobs.close()
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
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        self._mode = mode
        self.jobs.set_mode(mode)

    @property
    def model_name(self) -> str | None:
        """The model this agent talks to: its own override, else the config's."""
        return self.model_override or self.config.model

    def _model(self) -> Model:
        """The configured model, rebuilt when login changes the config."""
        name = self.model_name
        if not name:
            raise NoModelError("No model configured; log in first")
        self._cached_model = self._built(name, self._cached_model)
        return self._cached_model[1]

    def _review_model(self) -> Model:
        """The model auto mode reviews with: the config's own choice, else the
        one paired with this agent's model, else this agent's model itself."""
        name = self.config.review_model or review.default_model(self.model_name or "")
        if not name or name == self.model_name:
            return self._model()
        self._cached_review_model = self._built(name, self._cached_review_model)
        return self._cached_review_model[1]

    def _built(self, name: str, cached: tuple[tuple, Model] | None) -> tuple[tuple, Model]:
        """``cached`` if it still matches the config, else ``name`` built anew."""
        api_base, api_key = self.config.provider_auth(name)
        key = (name, api_base, api_key)
        if cached is None or cached[0] != key:
            cached = (key, build_model(*key))
        return cached

    def context_window(self) -> int | None:
        """The context window compaction works against, None when unknown."""
        try:
            model = self._model()
        except Exception:  # noqa: BLE001 - the request itself reports an unusable model
            model = None
        return compaction.context_window(model, self.config.compaction_context_window)

    # The two methods below are the only paths that write conversation state.
    # They keep the invariant that ``self.history`` always equals what
    # replaying the session log would produce.

    def _append_message(self, message: ModelMessage) -> str:
        self.history.append(message)
        return self.session.append_message(message)

    def _persist_slot(self, record_id: str, index: int, request: ModelRequest) -> None:
        """Persist slot ``index`` of a tool request already present in
        ``self.history``, after its result was filled in place."""
        self.session.append_tool_result(record_id, index, request)

    async def _maybe_compact(self, *, force: bool = False) -> compaction.CompactionResult | None:
        """Compact the context if it is close to full.

        ``force`` is the manual path: the user asked for it, so neither the
        auto-compaction toggle nor the token threshold applies.  It still keeps
        the recent window, and so returns None on a history short enough that
        there is nothing to summarize.
        """
        window = self.context_window()
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
        # Measured before anything is committed, the way count_context_tokens
        # would measure it with no anchor. This is the last await: the commit
        # below runs in one go, so a compaction cancelled at any point leaves
        # the history and the log as they were, and nothing can observe it
        # half done.
        tokens_after = await asyncio.to_thread(
            compaction.count_tokens, list(result.messages), self.tool_schemas, self.system_prompt)

        # Mirrors how Session.messages() replays a compaction record, so the
        # append-message invariant (see _append_message) still holds afterwards.
        self.session.append_compaction(result.summary, result.kept_messages, result.tokens_before,
                                       checkpoint=result.checkpoint)
        self.history = result.messages
        # The provider's count described the old context.
        self._usage_anchor = None
        result.tokens_after = tokens_after
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

    async def compact_now(self) -> compaction.CompactionResult | None:
        """Compact on demand; None when the history is too short to be worth it.

        Raises ``AgentBusyError`` while a turn or another compaction of this
        agent is in flight. Cancelling it leaves the history and the log
        untouched.
        """
        self._claim("compaction")
        try:
            return await self._maybe_compact(force=True)
        finally:
            self._in_flight = None

    def _claim(self, what: str) -> None:
        """Mark ``what`` as the one writer of the history, or refuse."""
        if self._in_flight is not None:
            raise AgentBusyError(f"cannot start a {what}: a {self._in_flight} "
                                 "of this agent is still in flight")
        self._in_flight = what

    def _context(self, model: Model) -> list[ModelMessage]:
        """What a request over this conversation opens with: the system prompt,
        then the history as ``model`` can take it.

        Every request that sends the history goes through here (turn steps,
        the recap); the reviewer and compaction write their own text.
        """
        return [
            ModelRequest(parts=[SystemPromptPart(content=self.system_prompt)]),
            *_failures_as_text(_strip_foreign_thinking(self.history, model)),
        ]

    async def recap(self) -> str:
        """Where the session stands, in a few lines, for a user coming back to it.

        Read-only: neither the question nor the answer enters ``self.history``
        or the session log, and no compaction is triggered. Only between
        turns, where the history is a complete request on its own.

        The request is built from the history before the first await, and a
        compaction swaps the history in one synchronous step, so a recap sees
        the history either before a compaction or after it, never half way.

        The request opens exactly like a turn's (same prompt, tools and
        history), so a provider that caches prefixes has this one cached.
        """
        model = self._model()
        question = ModelRequest(parts=[UserPromptPart(content=RECAP_PROMPT)])
        answer = await ask_once(model, [*self._context(model), question],
                                max_tokens=_RECAP_MAX_TOKENS, tools=self._tool_definitions,
                                cache_key=self.session.id)
        return answer.strip()

    # ---- Commands the user runs themselves ---------------------------------

    async def run_shell(self, command: str) -> str:
        """Run a shell command on the user's behalf and return what it printed.

        The TUI's "!" prefix. No permission gate: the gate exists so the model
        cannot run something the user did not ask for, and here the user *is*
        the one asking. Nothing is recorded — the caller decides whether the
        run reaches the model now or at the running turn's next step.
        """
        return await user_shell(command, self.cwd, self.tool_context)

    def record_shell(self, command: str, output: str) -> None:
        """Persist a "!" run so the model sees what the user just did.

        Only between turns. While one is running the message belongs in the
        pending queue instead, which the loop drains between steps — appended
        here it would land between a tool call and its result. Raises
        ``AgentBusyError`` while a turn or a compaction is in flight.
        """
        if self._in_flight is not None:
            raise AgentBusyError(f"cannot record a shell run: a {self._in_flight} "
                                 "of this agent is still in flight")
        self._append_message(shell_message(command, output))

    # ---- Tools the agent loop runs itself ----------------------------------
    # These mutate agent-held state, so they cannot go through
    # the stateless run_tool. Each handler fills ``slot`` with the tool
    # result, calls ``persist()`` (which writes that slot) after every slot
    # update, and yields the events to forward.

    async def _run_write_todos(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                               persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        # Reports itself as TodosUpdate alone — no ToolStart/ToolEnd — which is
        # also the shape replay_events produces for it, so no renderer has to
        # special-case the name. A malformed argument is the exception: it
        # renders as a normal failed tool call, live and on replay alike.
        todos = normalize_todos(args.get("todos"))
        if todos is None:
            yield ToolStart(call.tool_call_id, call.tool_name, args)
            result = ToolResult("Error: 'todos' must be an array of "
                                "{content, status} objects.", "failed")
            _settle(slot, result)
            persist()
            yield ToolEnd.of(call.tool_call_id, call.tool_name, result)
            return
        self.todos = todos
        _settle(slot, ToolResult(render_todos(self.todos)))
        persist()
        yield TodosUpdate(list(self.todos))

    async def _run_ask_user(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                            persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        # The one tool whose result comes from the user. No ask hook means no
        # user to answer (headless leaves the tool out of its toolset, so this
        # is a narrowed-toolset edge, not the normal path); the error tells the
        # model to carry on rather than wait for an answer that cannot come.
        yield ToolStart(call.tool_call_id, call.tool_name, args)
        question = str(args.get("question") or "").strip()
        options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]
        if not question:
            result = ToolResult("Error: question is required.", "failed")
        elif len(options) > 9:
            result = ToolResult("Error: at most 9 options.", "failed")
        elif self.ask is None:
            result = ToolResult("Error: nobody is here to answer in this run. State your "
                                "assumption and continue.", "failed")
        else:
            answer = await self.ask(question, options)
            if answer is None:
                result = ToolResult("The user dismissed the question without answering. "
                                    "Proceed on your best judgment and say what you assumed.")
            else:
                result = ToolResult(f"User answered: {answer}")
        _settle(slot, result)
        persist()
        yield ToolEnd.of(call.tool_call_id, call.tool_name, result)

    async def _refusal(self, name: str, args: dict) -> str | None:
        """Authorize a tool the loop runs itself, the way run_tool does the rest:
        None when it may run, else what the model is told."""
        return await authorize(name, args, self.mode, self.cwd, self.confirm,
                               self._review, self.toolset,
                               safe_commands=self.run_options.safe_commands(self.config),
                               ctx=self.tool_context)

    async def _review(self, name: str, args: dict) -> str | None:
        """Put a call auto mode holds to the reviewer model.

        The user is the fallback, never the first stop: they are asked when no
        verdict came back, and when the reviewer has refused MAX_BLOCKS calls
        in a row. With nobody to ask, both are refusals.
        """
        try:
            verdict = await asyncio.wait_for(
                review.judge(self._review_model(), self.history, name, args, self.cwd,
                             subagent=self.session.parent_id is not None),
                review.TIMEOUT)
        except Exception as exc:  # noqa: BLE001 — any failure means no verdict
            verdict = None
            refusal = f"Denied: the auto mode reviewer was unavailable ({exc or type(exc).__name__})."
        else:
            if verdict.allow:
                self._review_blocks = 0
                return None
            self._review_blocks += 1
            refusal = (f"Denied by the auto mode reviewer: {verdict.reason} Do not look "
                       "for another way to do this; if it is needed, tell the user so.")
            if self._review_blocks < review.MAX_BLOCKS:
                return refusal
        if self.confirm is None:
            return refusal
        if not await self.confirm(name, args):
            return USER_DENIAL
        self._review_blocks = 0
        return None

    async def _run_job_tool(self, call: ToolCallPart, args: dict, slot: ToolReturnPart,
                            persist: Callable[[], None]) -> AsyncIterator[AgentEvent]:
        """The job tools: starting agents and commands, reading and stopping them.

        They act on the jobs this agent holds, which no stateless tool
        function can reach, so they are dispatched from here. Only
        run_background is gated: it starts a process, while the others reach
        nothing but what this same agent started.
        """
        yield ToolStart(call.tool_call_id, call.tool_name, args)
        if (refusal := await self._refusal(call.tool_name, args)) is not None:
            result = ToolResult(refusal, "denied")
        else:
            try:
                # list_models is about this agent's models, not its jobs.
                result = ToolResult(self._list_models() if call.tool_name == "list_models"
                                    else await self.jobs.call(call.tool_name, args))
            except ToolError as exc:
                result = exc.result()
        _settle(slot, result)
        persist()
        yield ToolEnd.of(call.tool_call_id, call.tool_name, result)

    def available_models(self) -> list[str]:
        """The models this agent can be put on, as qualified names.

        Read from the config on every call rather than baked into a prompt, so
        a provider signed in to mid-session is offered from the next call on.
        Only providers with a stored credential are listed, plus this agent's
        own: one that works off an environment variable alone is not known to
        work until it is tried.
        """
        providers = []
        if self.model_name:
            try:
                providers.append(split_model_string(self.model_name)[0])
            except ValueError:
                pass
        providers.extend(self.config.providers)
        providers.extend(signed_in_plans())
        return [f"{provider}:{name}"
                for provider in dict.fromkeys(providers)
                for name in models_for(provider)]

    def _list_models(self) -> str:
        """The models an agent can be started on, one qualified name per line."""
        lines = self.available_models()
        if not lines:
            return "No models are available; the user has to log in first."
        return (f"You are running on {self.model_name}. An agent can be started on:\n"
                + "\n".join(lines)
                + "\nOther models of the same providers work too, when named exactly.")

    def _open_child(self, model: str | None) -> "Agent":
        """A subagent of this one, for the job manager to run: its own session,
        marked as this one's child, on ``model`` or else this agent's model."""

        # Weak, like the jobs' hold on this agent: the closure lives on the
        # child, and a strong reference back would make the child a cycle.
        child_ref: "weakref.ReferenceType[Agent] | None" = None

        async def confirm(name: str, args: dict) -> bool:
            # For a child nothing put on screen; a UI that shows it asks there
            # instead. Through the parent, its hook looked up per call: whoever
            # can ask for this conversation answers for the child too, and
            # with nobody to ask (headless) it is denied, exactly as it would
            # be for the parent. The request is marked with the child's title,
            # so the parent's UI can say it is not the parent's own call; the
            # job is looked up per call because its id is only given once the
            # child is open.
            if not self.confirm:
                return False
            origin = self.jobs.title_of(child_ref and child_ref()) or "agent"
            return await confirm_for(self.confirm, origin, name, args)

        child = Agent.open(
            cwd=self.cwd, mode=self.mode, config=self.config,
            run_options=self.run_options, confirm=confirm,
            # Narrowed from this agent's own set, so a parent that was given
            # less never hands more to what it starts.
            toolset=without(self.toolset, SUBAGENT_DENIED),
            model_override=model or self.model_override,
            # Marked as this session's child so it stays out of the session
            # lists and out of `paimon -c`.
            parent_session_id=self.session.id)
        child_ref = weakref.ref(child)
        return child

    def _take_notices(self) -> list[JobNotice]:
        """Move the waiting notices into the history, one message each."""
        notices = self.jobs.take_notices()
        for text in notices:
            self._append_message(job_message(text))
        return [JobNotice(text) for text in notices]

    async def _run_code(self, code: str,
                        spend: Callable[[], str | None]) -> tuple[str, bool, list[dict]]:
        """Run a run_code script: (result, completed, the calls it made).

        Each call a script makes is this agent's own: it comes off the turn's
        budget through ``spend`` and passes the same validation and gate as
        one the model made directly. A call that does not get through raises
        inside the script, with the refusal as its message.
        """
        calls: list[dict] = []
        # Gathered calls reach the gate together, and two of them asking the
        # user at once would put one prompt over the other. Only the asking
        # is serialized; what was let through runs concurrently.
        asking = asyncio.Lock()

        async def call(name: str, args: dict) -> str:
            # "cancelled" is what stays when the script ends before the call does.
            record = {"name": name, "status": "cancelled", "seconds": 0.0,
                      "detail": logtext.summarize_call(name, args, limit=logtext.LOG_DETAIL_WIDTH)}
            calls.append(record)
            started = time.perf_counter()
            try:
                problem = spend()
                if problem is None:
                    invalid = validate_args(name, args, self.toolset)
                    problem = invalid and f"invalid arguments for {name}: {invalid}"
                if problem is not None:
                    record["status"] = "error"
                    raise codemode.ToolCallError(problem)
                async with asking:
                    refusal = await self._refusal(name, args)
                if refusal is not None:
                    record["status"] = "denied"
                    raise codemode.ToolCallError(refusal)
                result = await execute_tool(name, args, self.cwd, mode=self.mode,
                                            registry=self.toolset, ctx=self.tool_context)
                # A failed call still returns its text to the script, which
                # reads the error the way the model would; only the record
                # says it failed.
                record["status"] = "ok" if result.ok else "error"
                return result.text
            finally:
                record["seconds"] = round(time.perf_counter() - started, 3)

        result, completed = await codemode.run(code, self.toolset, call)
        return result, completed, calls

    _AGENT_HANDLED = {
        "write_todos": _run_write_todos,
        "ask_user": _run_ask_user,
        "spawn_agent": _run_job_tool,
        "list_models": _run_job_tool,
        "run_background": _run_job_tool,
        "read_job": _run_job_tool,
        "stop_job": _run_job_tool,
    }

    def expand_input(self, text: str) -> str:
        """What a typed message becomes: a /skill:name command expands to the
        skill's content, then @path mentions are inlined."""
        def mentions(part: str) -> str:
            return expand_mentions(part, self.cwd)

        expanded = expand_skill_command(text, self.skills, expand_args=mentions)
        # Unchanged (the same object back) means it was not a skill command.
        return mentions(text) if expanded is text else expanded

    async def run(self, user_input: str | None, *, expand: bool = True,
                  max_tool_calls: int | None = None) -> AsyncIterator[AgentEvent]:
        """Run one user turn to completion, yielding events along the way.

        ``user_input`` may be None: a wake-up turn, run so the model reacts to
        jobs that ended — the notices injected below are its only new input.
        When there is nothing to report either, the turn never happened: no
        events, nothing persisted, no model request.

        ``expand=False`` skips @path expansion, for callers that assembled the
        prompt themselves and must not have unrelated text rewritten (piped
        stdin, where a line like ``@foo.py`` is data rather than a mention).

        ``max_tool_calls`` bounds the tool calls this turn may execute. It is
        enforced here, where every ToolCallPart is dispatched — including the
        agent-handled tools like write_todos that produce no ToolStart — so no
        tool can slip past the budget. On the budget every remaining call gets
        an explicit refusal persisted as its result, ToolBudgetExhausted is
        yielded and the turn ends.

        Raises ``AgentBusyError`` when a turn or a compaction of this agent is
        already in flight: the history has one writer at a time.
        """
        self._claim("turn")
        try:
            # Jobs this session started report in here, ahead of the prompt they
            # arrived before. Persisted messages, not just events: the notice is
            # how the model gets a child's answer at all.
            notices = self._take_notices()
            if user_input is None and not notices:
                return
            for notice in notices:
                yield notice
            self.jobs.tool_budget = max_tool_calls
            if user_input is not None:
                prompt = self.expand_input(user_input) if expand else user_input
                self._append_message(ModelRequest(parts=[UserPromptPart(content=prompt)]))

            # Every turn leaves exactly one terminal ``turn_end`` record in the
            # session: success, error (with the exception), max_tool_calls or
            # interrupted. The record is not a message, so it reaches audits and
            # ``paimon log`` without an error or a partial answer ever being
            # replayed to the model as a normal assistant response.
            outcome_recorded = False

            def record_outcome(outcome: str, *, error: str | None = None,
                               partial_text: str | None = None) -> None:
                nonlocal outcome_recorded
                if outcome_recorded:
                    return
                outcome_recorded = True
                self.session.append_meta("turn_end", outcome=outcome, error=error,
                                         partial_text=partial_text)

            try:
                # Closed with this one rather than left to the garbage
                # collector, so an abandoned turn's model stream is shut before
                # its turn_end is recorded and the agent is freed.
                async with contextlib.aclosing(self._run(record_outcome, max_tool_calls)) as events:
                    async for event in events:
                        yield event
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
        finally:
            self._in_flight = None

    async def run_until_idle(self, user_input: str | None, *, expand: bool = True,
                             max_tool_calls: int | None = None) -> AsyncIterator[AgentEvent]:
        """Run ``user_input``, then keep the agent going until it has nothing
        left to wait for, yielding the events of every turn.

        A turn can start agents (spawn_agent) whose answers arrive after it
        ends, as notices the model only sees in a later turn. This runs that
        later turn — a wake-up, ``run(None)`` — whenever notices are waiting,
        and waits while agents this one started are still running. It returns
        once nothing is running and nothing is left unsaid.

        Each turn gets ``max_tool_calls`` afresh, and a ToolBudgetExhausted
        ends the whole run after it is yielded. Background commands do not
        keep it waiting: one may run for as long as the process lives, and
        what becomes of it is reported by whatever runs next. Plain ``run``
        is still the call for a caller that drives the conversation itself,
        such as a UI that wakes the agent when its jobs move.
        """
        with self.jobs.watch() as moved:
            prompt, first = user_input, True
            while True:
                if not first:
                    moved.clear()
                    if not self.jobs.notices:
                        if not self.jobs.agents_running:
                            return
                        await moved.wait()
                        continue
                # Closed here rather than left to the garbage collector: until
                # a turn's generator is closed it holds the agent, and the next
                # turn would be refused.
                async with contextlib.aclosing(self.run(
                        prompt, expand=expand, max_tool_calls=max_tool_calls)) as events:
                    async for event in events:
                        yield event
                        if isinstance(event, ToolBudgetExhausted):
                            return
                prompt, first = None, False

    async def _run(self, record_outcome: Callable[..., None],
                   max_tool_calls: int | None) -> AsyncIterator[AgentEvent]:
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
        budget_hit = False

        def spend() -> str | None:
            """Take one call off the turn's budget: None, or the refusal when
            there is none left. Shared by the calls the model makes and the
            ones its run_code scripts make, so neither can slip past it."""
            nonlocal calls_made, budget_hit
            if max_tool_calls is not None and calls_made >= max_tool_calls:
                budget_hit = True
                return (f"Not executed: the run reached its tool call "
                        f"budget (max_tool_calls={max_tool_calls}).")
            calls_made += 1
            return None

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
            # Notices from jobs that ended meanwhile come in the same way.
            for notice in self._take_notices():
                yield notice
            for queued in (self.pending() if self.pending is not None else []):
                # A ready-made request is one the caller already showed the
                # user (a "!" run, whose output was on screen the moment the
                # command exited), so it is appended without an event of its
                # own — and without expansion, since @path in a command's
                # output is output rather than a mention.
                if isinstance(queued, ModelRequest):
                    self._append_message(queued)
                    continue
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
            request_messages = self._context(model)
            parameters = ModelRequestParameters(
                function_tools=self._tool_definitions, allow_text_output=True
            )

            content = ""  # accumulated text, kept if the stream is interrupted
            attempt = 0
            stats: RequestStats | None = None

            while True:
                started = False  # this attempt has yielded something to the caller
                first_event_at: float | None = None
                try:
                    async with model_request_stream(
                        model,
                        request_messages,
                        model_settings=request_settings(model, self.session.id),
                        model_request_parameters=parameters,
                    ) as stream:
                        async for event in stream:
                            if first_event_at is None:
                                first_event_at = time.perf_counter()
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
                            # perf_counter, not monotonic: Windows ticks monotonic
                            # every ~16ms, so a short response measured 0s and
                            # reported no stats at all.
                            elapsed = time.perf_counter() - first_event_at
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
                            request_messages = self._context(model)
                            continue
                    # Counted only here: the compact-and-retry above is its own
                    # allowance, not one of the transient retries.
                    attempt += 1
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

            for index, (slot, call) in enumerate(zip(returns, calls)):
                # Writes this one slot as it stands, so each result is durable
                # the moment it is settled without rewriting the others.
                persist = functools.partial(self._persist_slot, record_id, index, tool_request)
                args = parse_args(call.args)
                name = call.tool_name

                # A call refused before it runs is still answered, with a
                # failed result saying why: "failed" rather than "denied",
                # which means a permission refusal.
                refused: ToolResult | None = None
                if (over := spend()) is not None:
                    # The budget check comes before any dispatch,
                    # agent-handled tools included: a refused call never
                    # executes, and its slot records why instead of a generic
                    # interrupted placeholder.
                    refused = ToolResult(over, "failed", "over budget")
                elif name not in self.toolset:
                    # Rejected before the agent-handled table below, so
                    # excluding write_todos or ask_user from a toolset
                    # actually disables them.
                    refused = ToolResult(f"Error: unknown tool {name!r}", "failed")
                elif (invalid := validate_args(name, args, self.toolset)) is not None:
                    # One schema validation for every call — before gating
                    # and before the agent-handled dispatch below — so a
                    # malformed argument is a tool error the model can react
                    # to, never an exception ending the turn (TOOLS-1).
                    refused = ToolResult(f"Error: invalid arguments for {name}: {invalid}",
                                         "failed")
                if refused is not None:
                    yield ToolStart(call.tool_call_id, name, args)
                    _settle(slot, refused)
                    persist()
                    yield ToolEnd.of(call.tool_call_id, name, refused)
                    continue

                if name == codemode.NAME:
                    # Not in the table below: it needs the turn's budget, which
                    # only this scope holds. An interrupt leaves the slot its
                    # placeholder, like any other tool cut short.
                    yield ToolStart(call.tool_call_id, name, args)
                    text, completed, nested = await self._run_code(args["code"], spend)
                    result = ToolResult(text) if completed else ToolResult(text, "failed")
                    # Kept beside the result, where it is persisted for replay
                    # and never sent to the model.
                    slot.metadata = {"calls": nested}
                    _settle(slot, result)
                    persist()
                    yield ToolEnd.of(call.tool_call_id, name, result, nested)
                    continue

                handler = self._AGENT_HANDLED.get(name)
                if handler is not None:
                    async for event in handler(self, call, args, slot, persist):
                        yield event
                    continue

                yield ToolStart(call.tool_call_id, name, args)
                result = await run_tool(name, args, self.cwd, self.mode,
                                        self.confirm, self.toolset,
                                        safe_commands=self.run_options.safe_commands(self.config),
                                        ctx=self.tool_context,
                                        review=self._review)
                _settle(slot, result)
                persist()
                yield ToolEnd.of(call.tool_call_id, name, result)

            if budget_hit:
                record_outcome("max_tool_calls")
                yield ToolBudgetExhausted(max_tool_calls)
                return
            # loop again so the model can react to tool results
