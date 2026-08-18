"""Non-interactive mode: run one turn without the UI and exit.

``paimon -p "prompt"`` streams the assistant's answer to stdout and puts
everything else (tool calls, denials, notices) on stderr, so ``$(paimon -p ...)``
captures just the answer.

``--output-format json`` writes one JSON object per line to stdout instead:

    {"type": "init", "session_id": str|null, "model": str|null, "mode": str|null, "cwd": str|null}
    {"type": "reasoning_delta" | "text_delta", "text": str}
    {"type": "tool_use", "id": str, "name": str, "args": object}
    {"type": "tool_result", "id": str, "name": str, "result": str, "denied": bool}
    {"type": "todos", "todos": array}
    {"type": "agents", "text": str}
    {"type": "compacted", "tokens_before": int, "tokens_after": int}
    {"type": "compaction_failed", "error": str}
    {"type": "retry", "attempt": int, "max_attempts": int, "delay": float, "error": str}
    {"type": "result", "subtype": "success"|"error"|"interrupted"|"timeout"|"max_tool_calls",
     "is_error": bool, "session_id": str|null, "text": str, "denied": int, "error": str|null}

The first line is always ``init`` and the last is always ``result`` — including
when the run fails before the model is reached. Consumers must ignore unknown
event types and unknown fields; existing fields are only added to, never
removed or redefined. A new output shape would be a new ``--output-format``
value rather than a change to this one.

``--output-format result`` writes exactly one JSON object to stdout: the same
``result`` line the json format ends with, and nothing else. This is the cheap
format for a calling agent that only needs the outcome; the intermediate
events are still persisted in the session and can be inspected afterwards with
``paimon log``. The result carries ``log_end``, the session log's line count
at the end of the run, so a later ``paimon log <id> --after <log_end>`` shows
exactly what a follow-up run added. The same field-stability rule applies.

Exit codes: 0 success, 1 error, 2 usage, 4 tool budget exhausted (``result``
subtype ``max_tool_calls``), 124 timed out (``timeout``), 130 interrupted.
Either way the session is persisted and resumable via the reported id.
"""

import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from . import compaction, tools
from .agent import (
    Agent,
    AgentsNotice,
    ContextCompacted,
    ContextCompactionFailed,
    ModelRetry,
    ReasoningDelta,
    TextDelta,
    TodosUpdate,
    ToolBudgetExhausted,
    ToolEnd,
    ToolStart,
)
from .config import Config
from .mentions import expand_mentions
from .session import Session, SessionError, resume_hint

# Piped stdin beyond this is truncated rather than pushed into the context
# window (a stray `cat huge.log |` should not blow up the request).
MAX_STDIN_CHARS = 100_000

DETAIL_WIDTH = 120  # tool-call detail on a progress line

PIPED_TAG = "piped_input"

_DIM, _RESET = "\x1b[2m", "\x1b[0m"


def read_stdin(stream=None) -> str:
    """All of stdin as text. Only called when stdin is not a terminal."""
    stream = sys.stdin if stream is None else stream
    try:
        # Read bytes and decode here: sys.stdin's own encoding is ascii under
        # LC_ALL=C, which would fail on any non-ASCII byte in the pipe.
        raw = stream.buffer.read()
    except (AttributeError, ValueError):
        raw = (stream.read() or "").encode("utf-8", errors="replace")
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > MAX_STDIN_CHARS:
        text = text[:MAX_STDIN_CHARS] + f"\n... (truncated, {len(text) - MAX_STDIN_CHARS} more chars)"
    return text


def build_prompt(prompt: str, piped: str, cwd: Path) -> str:
    """Combine the -p prompt with piped stdin into one user message.

    Only the prompt gets @path expansion: piped content is data, and a line
    that happens to read ``@foo.py`` must not be swapped for that file.
    The instruction goes last, after the material it applies to.
    """
    prompt = prompt.strip()
    piped = piped.strip()
    if prompt:
        prompt = expand_mentions(prompt, cwd)
    if not piped:
        return prompt
    block = f"<{PIPED_TAG}>\n{piped}\n</{PIPED_TAG}>"
    return f"{block}\n\n{prompt}" if prompt else block


def _write(stream, text: str) -> None:
    """Write and flush, tolerating a closed or disconnected stream.

    Flushing every write keeps stdout and stderr interleaved in order when
    both are a terminal; a broken pipe (``paimon -p x | head -1``) must not
    abort a turn that is otherwise fine and still has history to persist.
    """
    if not text:
        return
    try:
        stream.write(text)
        stream.flush()
    except (BrokenPipeError, ValueError):
        pass


class TextRenderer:
    """Human-readable rendering: answer on stdout, everything else on stderr."""

    def __init__(self, out=None, err=None, config: Optional[Config] = None) -> None:
        self._out = sys.stdout if out is None else out
        self._err = sys.stderr if err is None else err
        self._config = config or Config()
        self._color = _use_color(self._err)
        self._out_open = False  # stdout has an unterminated line
        self._call_open = False  # a progress line awaits its outcome
        self._denied = 0
        self._session_id: Optional[str] = None

    def _note(self, text: str) -> None:
        self._end_call()
        _write(self._err, f"{_DIM}{text}{_RESET}\n" if self._color else f"{text}\n")

    def _end_call(self) -> None:
        if self._call_open:
            _write(self._err, "\n")
            self._call_open = False

    def begin(self, session_id: Optional[str] = None, model: Optional[str] = None,
              mode: Optional[str] = None, cwd: Optional[Path] = None,
              log_path: Optional[Path] = None) -> None:
        self._session_id = session_id

    async def handle(self, ev: object) -> None:
        if isinstance(ev, TextDelta):
            self._end_call()
            _write(self._out, ev.text)
            self._out_open = not ev.text.endswith("\n")

        elif isinstance(ev, ReasoningDelta):
            if self._config.show_reasoning:
                self._end_call()
                _write(self._err, f"{_DIM}{ev.text}{_RESET}" if self._color else ev.text)

        elif isinstance(ev, ToolStart):
            self._end_call()
            detail = tools.summarize_call(ev.name, ev.args, limit=DETAIL_WIDTH)
            line = f"· {ev.name}  {detail}".rstrip()
            _write(self._err, f"{_DIM}{line}{_RESET}" if self._color else line)
            self._call_open = True

        elif isinstance(ev, ToolEnd):
            if ev.denied:
                self._denied += 1
                _write(self._err, "  → denied\n" if self._call_open else "· denied\n")
                self._call_open = False
                if self._denied == 1:
                    self._note("paimon: --print never asks for confirmation; "
                               "rerun with --mode edit or --mode yolo to allow this")
            else:
                self._end_call()

        elif isinstance(ev, TodosUpdate):
            self._note(tools.render_todos(ev.todos))

        elif isinstance(ev, AgentsNotice):
            self._note(f"· agents: {ev.text}")

        elif isinstance(ev, ContextCompacted):
            self._note(f"· context compacted: {ev.tokens_before:,} → ~{ev.tokens_after:,} tokens")

        elif isinstance(ev, ContextCompactionFailed):
            self._note(f"paimon: context compaction failed, continuing without it: {ev.error}")

        elif isinstance(ev, ModelRetry):
            self._note(f"paimon: {ev.error}, retrying in {ev.delay:g}s "
                       f"({ev.attempt}/{ev.max_attempts - 1})")

    async def close(self) -> None:
        # No await anywhere in here: close() runs from a finally block while
        # the turn may be cancelling, and any real suspension point would
        # raise CancelledError again and drop the trailing output.
        self._end_call()
        if self._out_open:
            _write(self._out, "\n")
            self._out_open = False

    def finish(self, subtype: str = "success", error: Optional[str] = None) -> None:
        if error:
            self._note(f"paimon: {error}")
        if subtype == "interrupted":
            self._note("· interrupted")
        if self._session_id:
            self._note(f"· resume: {resume_hint(self._session_id)}")


class JsonRenderer:
    """One JSON object per line on stdout; see the module docstring."""

    _TYPES = {
        ReasoningDelta: "reasoning_delta",
        TextDelta: "text_delta",
        ToolStart: "tool_use",
        ToolEnd: "tool_result",
        TodosUpdate: "todos",
        AgentsNotice: "agents",
        ContextCompacted: "compacted",
        ContextCompactionFailed: "compaction_failed",
        ModelRetry: "retry",
    }

    def __init__(self, out=None, config: Optional[Config] = None) -> None:
        self._out = sys.stdout if out is None else out
        self._blocks: list[str] = []
        self._current = ""
        self._denied = 0
        self._session_id: Optional[str] = None
        self._log_path: Optional[Path] = None

    def _emit(self, payload: dict) -> None:
        # default=str keeps malformed tool arguments from aborting the run.
        _write(self._out, json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def begin(self, session_id: Optional[str] = None, model: Optional[str] = None,
              mode: Optional[str] = None, cwd: Optional[Path] = None,
              log_path: Optional[Path] = None) -> None:
        self._session_id = session_id
        self._log_path = log_path
        self._emit({"type": "init", "session_id": session_id, "model": model,
                    "mode": mode, "cwd": str(cwd) if cwd else None})

    async def handle(self, ev: object) -> None:
        name = self._TYPES.get(type(ev))
        if name is None:  # TurnEnd is implied by the result line
            return

        if isinstance(ev, TextDelta):
            self._current += ev.text
            self._emit({"type": name, "text": ev.text})
        elif isinstance(ev, ReasoningDelta):
            # Emitted regardless of config.show_reasoning: that flag is
            # display-only, and a machine stream should not lose data to it.
            self._emit({"type": name, "text": ev.text})
        elif isinstance(ev, ToolStart):
            self._flush_block()
            self._emit({"type": name, "id": ev.id, "name": ev.name, "args": ev.args})
        elif isinstance(ev, ToolEnd):
            self._denied += ev.denied
            self._emit({"type": name, "id": ev.id, "name": ev.name,
                        "result": ev.result, "denied": ev.denied})
        elif isinstance(ev, TodosUpdate):
            self._emit({"type": name, "todos": ev.todos})
        elif isinstance(ev, AgentsNotice):
            self._emit({"type": name, "text": ev.text})
        elif isinstance(ev, ContextCompacted):
            self._emit({"type": name, "tokens_before": ev.tokens_before,
                        "tokens_after": ev.tokens_after})
        elif isinstance(ev, ContextCompactionFailed):
            self._emit({"type": name, "error": ev.error})
        elif isinstance(ev, ModelRetry):
            self._emit({"type": name, "attempt": ev.attempt,
                        "max_attempts": ev.max_attempts, "delay": ev.delay, "error": ev.error})

    def _flush_block(self) -> None:
        if self._current:
            self._blocks.append(self._current)
            self._current = ""

    async def close(self) -> None:
        self._flush_block()  # no await here; see TextRenderer.close

    def _log_end(self) -> Optional[int]:
        """Line count of the session log, the cursor for ``paimon log --after``."""
        if self._log_path is None:
            return None
        try:
            with self._log_path.open("rb") as file:
                return sum(1 for _ in file)
        except OSError:
            return None

    def finish(self, subtype: str = "success", error: Optional[str] = None) -> None:
        self._flush_block()
        self._emit({"type": "result", "subtype": subtype, "is_error": subtype != "success",
                    "session_id": self._session_id, "text": "\n\n".join(self._blocks),
                    "denied": self._denied, "error": error, "log_end": self._log_end()})


class ResultRenderer(JsonRenderer):
    """Only the final ``result`` line; the format for callers that just want the outcome."""

    def _emit(self, payload: dict) -> None:
        if payload.get("type") == "result":
            super()._emit(payload)


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _make_renderer(output_format: str, config: Config):
    if output_format == "json":
        return JsonRenderer(config=config)
    if output_format == "result":
        return ResultRenderer(config=config)
    return TextRenderer(config=config)


def _install_interrupt(task: "asyncio.Task") -> None:
    """Make Ctrl-C cancel the turn rather than tear down the loop.

    Cancelling lets Agent.run write back the partial response (so the session
    log keeps no dangling tool call) and lets the shell tool kill its process
    tree. The handler is removed as it fires, so a second Ctrl-C hits the
    default handler and exits hard even if cleanup wedges.
    """
    loop = asyncio.get_running_loop()

    def cancel() -> None:
        try:
            loop.remove_signal_handler(signal.SIGINT)
        except (NotImplementedError, RuntimeError):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGINT, signal.default_int_handler)
        task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, cancel)
    except NotImplementedError:  # Windows event loops
        with contextlib.suppress(ValueError, OSError):
            signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(cancel))


class _ToolBudgetExceeded(Exception):
    """The turn asked for more tool calls than --max-tool-calls allows.

    Control flow inside this module rather than a failure it reports, so
    it stays off PaimonError: a caller catching that must not swallow the
    signal that ends the turn here.
    """


async def _run_turn(agent: Agent, renderer, text: str,
                    max_tool_calls: Optional[int] = None) -> None:
    try:
        # The budget itself is enforced inside Agent.run, at the point every
        # tool call is dispatched — counting renderer events here would miss
        # the agent-handled tools that never produce a ToolStart.
        async for ev in agent.run(text, expand=False, max_tool_calls=max_tool_calls):
            await renderer.handle(ev)
            if isinstance(ev, ToolBudgetExhausted):
                raise _ToolBudgetExceeded
    finally:
        await renderer.close()


async def _drive(agent: Agent, renderer, text: str, *,
                 timeout: Optional[float] = None,
                 max_tool_calls: Optional[int] = None) -> int:
    turn = asyncio.ensure_future(_run_turn(agent, renderer, text, max_tool_calls))
    _install_interrupt(turn)
    try:
        await asyncio.wait_for(turn, timeout)
    except asyncio.CancelledError:
        renderer.finish(subtype="interrupted")
        return 130
    except asyncio.TimeoutError:
        # The agent recorded the cancellation as "interrupted" — only this
        # scope knows it was the --timeout deadline, so refine the outcome.
        agent.session.append_meta("turn_end", outcome="timeout",
                                  error=f"timed out after --timeout {timeout:g}s")
        renderer.finish(subtype="timeout", error=f"timed out after --timeout {timeout:g}s")
        return 124
    except _ToolBudgetExceeded:
        renderer.finish(subtype="max_tool_calls",
                        error=f"stopped after reaching --max-tool-calls {max_tool_calls}")
        return 4
    except Exception as exc:  # noqa: BLE001 — report any failure and exit non-zero
        renderer.finish(subtype="error", error=f"{type(exc).__name__}: {exc}")
        return 1
    renderer.finish()
    return 0


def fail(message: str, output_format: str = "text", config: Optional[Config] = None) -> int:
    """Report a failure that happened before a turn could start."""
    renderer = _make_renderer(output_format, config or Config())
    renderer.begin()
    renderer.finish(subtype="error", error=message)
    return 1


def _relax_encoding() -> None:
    """Never let an unencodable character kill a half-written answer.

    Under LC_ALL=C stdout is ascii, and the first non-ASCII token of a reply
    would raise mid-stream.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")


def run(*, prompt: str, piped: str, cwd: Path, mode: str, session: Optional[Session],
        output_format: str = "text", config: Optional[Config] = None,
        model: Optional[str] = None, timeout: Optional[float] = None,
        max_tool_calls: Optional[int] = None,
        append_system_prompt: Optional[str] = None) -> int:
    """Run one turn and return the process exit code.

    ``model`` overrides the configured model for this run only; nothing is
    written back to the config file. ``timeout`` and ``max_tool_calls`` bound
    an unattended run; on either limit the partial turn is persisted and the
    result names the limit that was hit. ``append_system_prompt`` extends a new
    session's system prompt (the CLI rejects it together with resume).
    """
    _relax_encoding()
    try:
        config = config or Config.load()
    except Exception as exc:  # corrupt config file, unreadable config home
        return fail(str(exc), output_format)
    if model:
        config.model = model
    renderer = _make_renderer(output_format, config)

    # Checked before constructing the Agent, which would otherwise create a
    # session file and probe the environment for a run that cannot happen.
    if not config.model:
        renderer.begin()
        renderer.finish(subtype="error", error="no model configured; run 'paimon' once to log in")
        return 1

    try:
        text = build_prompt(prompt, piped, cwd)
        # No agent tools here: -p runs exactly one turn and asyncio.run tears
        # the loop down the moment it returns, cancelling any agent still
        # working at whatever point it had reached — including, mid-cleanup,
        # the kill of a shell command's process group. Offering the model a
        # tool that cannot be finished is worse than not having it.
        agent = Agent.open(cwd=cwd, session=session, confirm=None, mode=mode, config=config,
                           append_system_prompt=append_system_prompt,
                           toolset=tools.without(tools.REGISTRY, tools.SUPERVISED_TOOLS))
    except SessionError as exc:  # busy in another process, or no persisted system prompt
        renderer.begin(session.id if session else None)
        renderer.finish(subtype="error", error=str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 — any startup failure must keep the protocol
        # An unusable data directory, a session whose messages no longer
        # validate, a failing system prompt build: machine formats promise a
        # first init line and a final result line no matter what. A session
        # that was never created leaves session_id and log_end null.
        # KeyboardInterrupt and SystemExit keep their meaning by staying
        # outside Exception.
        renderer.begin(session.id if session else None)
        renderer.finish(subtype="error", error=f"{type(exc).__name__}: {exc}")
        return 1

    renderer.begin(agent.session.id, config.model, mode, cwd, log_path=agent.session.path)
    # Auto-compaction silently doing nothing looks exactly like it working —
    # until the context overflows. An unattended run deserves the warning up
    # front; stderr, so the stdout protocols are untouched.
    if (config.compaction_enabled and compaction.context_window(
            agent.model_name, config.compaction_context_window) is None):
        _write(sys.stderr,
               f"paimon: context window unknown for {agent.model_name}; auto-compaction "
               "is off (set compaction.context_window in the config to enable it)\n")
    try:
        return asyncio.run(_drive(agent, renderer, text,
                                  timeout=timeout, max_tool_calls=max_tool_calls))
    except KeyboardInterrupt:  # before the handler was installed, or a second Ctrl-C
        renderer.finish(subtype="interrupted")
        return 130
    finally:
        # Synchronous: a cancelled coroutine's finally may never run.
        agent.session.unlock()
