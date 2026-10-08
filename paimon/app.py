"""Textual TUI for the Paimon agent.

The app is a container for panes: it owns the config, the theme, the command
palette, the global bindings and the status bar. Everything belonging to one
conversation lives in ``SessionPane``, everything belonging to an agent one
of them started in ``AgentPane``, and everything belonging to one background
command in ``CommandPane``.
"""

from functools import partial

from textual import events, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.content import Content
from textual.widgets import ContentSwitcher, Static

from . import herdr
from .agent import Agent, Job
from .agentpane import AgentPane
from .errors import PaimonError
from .login import LoginScreen, PickerScreen
from .pane import Pane, SessionPane
from .session import SessionError
from .tabs import PaneTabs
from .commandpane import CommandPane
from .ui import PromptInput

# Every pane holds a live agent and its context, or a live process, so panes
# cost model requests, memory and file descriptors, not just a row in the
# strip. A soft cap keeps a runaway loop of "one more session" from taking the
# app down with it.
MAX_PANES = 8


class PaneLimitError(PaimonError):
    """Every pane is taken, so nothing more can be put on screen."""


class PaimonApp(App):
    CSS_PATH = "app.tcss"

    # The prompt is a TextArea and claims most keys, so every app-wide binding
    # that has to work while typing is priority.
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "interrupt", "Interrupt"),
        Binding("shift+tab", "cycle_mode", "Cycle permission mode", priority=True),
        Binding("ctrl+t", "new_pane", "New pane", priority=True),
        # Takes ctrl+w from the prompt's delete-word-left, which keeps its
        # ctrl+backspace alias.
        Binding("ctrl+w", "close_pane", "Close pane", priority=True),
        Binding("ctrl+pageup", "prev_pane", "Previous pane", priority=True),
        Binding("ctrl+pagedown", "next_pane", "Next pane", priority=True),
        Binding("ctrl+g", "goto_attention", "Go to a pane waiting on you", priority=True),
    ]

    def get_system_commands(self, screen) -> list[SystemCommand]:
        return [
            *super().get_system_commands(screen),
            SystemCommand(
                "Login / switch provider",
                "Reconfigure model, API base and API key",
                self.action_login,
            ),
            SystemCommand(
                "Switch model",
                "Change to another model of a provider you are logged in to",
                self.action_switch_model,
            ),
            SystemCommand(
                "Toggle thinking display",
                "Show or hide the model's reasoning stream (it is generated either way)",
                self.action_toggle_reasoning,
            ),
            SystemCommand(
                "Toggle idle recap",
                "Offer a short recap after a turn once you go quiet (it costs an extra request)",
                self.action_toggle_recap,
            ),
            SystemCommand(
                "Compact context",
                "Summarize the earlier conversation now instead of waiting for the context to fill",
                self.action_compact,
            ),
            SystemCommand("New session", "Start a new empty session", self.action_new_session),
            SystemCommand("Fork session", "Copy this conversation into a new session and continue there",
                          self.action_fork_session),
            SystemCommand("Resume session", "Pick an earlier session in this directory to resume",
                          self.action_resume_session),
            SystemCommand("New pane", "Open another session alongside this one",
                          self.action_new_pane),
            SystemCommand("Close pane", "Close this session's pane", self.action_close_pane),
            *self._skill_commands(),
        ]

    def _skill_commands(self) -> list[SystemCommand]:
        """One palette entry per skill the current conversation knows.

        Picking one types ``/skill:name `` into the prompt rather than sending
        it, so arguments can follow; Enter then expands it.
        """
        pane = self._session()
        if pane is None:
            return []
        return [
            SystemCommand(f"Skill: {skill.name}", skill.description,
                          partial(self.action_use_skill, skill.name))
            for skill in pane.agent.skills
        ]

    def action_use_skill(self, name: str) -> None:
        if (pane := self._session()) is None:
            return
        prompt = pane.query_one(PromptInput)
        prompt.clear()
        prompt.insert(f"/skill:{name} ")
        prompt.focus()

    def __init__(self, agent: Agent, *, resumed: bool = False, pick_session: bool = False,
                 reporter: herdr.Reporter | None = None, resume_flags: tuple[str, ...] = ()) -> None:
        self._persist_theme_changes = False
        super().__init__()
        self.config = agent.config
        # Set only when running in a Herdr pane. ``resume_flags`` are the
        # launch options a resumed session needs to behave like this one.
        self._herdr = reporter
        self._resume_flags = resume_flags
        self._resume_pane: SessionPane | None = None
        pane = SessionPane(agent, resumed=resumed, id="pane-1")
        self._panes = [pane]
        self._current = pane
        self._next_pane = 2
        self._switcher = ContentSwitcher(pane, initial=pane.id, id="panes")
        self._tabs = PaneTabs()
        self._pick_session = pick_session
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self._persist_theme_changes = True

    # ---- panes --------------------------------------------------------------

    @property
    def pane(self) -> Pane:
        """The pane on screen. A conversation, or a background command."""
        return self._current

    @property
    def panes(self) -> list[Pane]:
        return list(self._panes)

    @property
    def sessions(self) -> list[SessionPane]:
        """The panes that hold a conversation. Not every pane does."""
        return [pane for pane in self._panes if isinstance(pane, SessionPane)]

    def _session(self) -> SessionPane | None:
        """The conversation a global action applies to.

        The current pane when it is one, and otherwise the first conversation
        there is: the palette and the key bindings stay usable while a command's
        output is on screen, rather than silently doing nothing. None only in
        the moment between the last conversation closing and its replacement
        being mounted.
        """
        if isinstance(self._current, SessionPane):
            return self._current
        sessions = self.sessions
        return sessions[0] if sessions else None

    def _sync_panes(self) -> None:
        """Redraw everything that shows more than one pane at a time."""
        self._tabs.sync(self._panes, self._current)
        # A visible strip opens with a rule, which is all the separation the
        # status bar needs; the bar drops its own bottom margin so the tabs
        # cost two rows rather than three.
        if self.is_mounted:
            self.screen.set_class(self._tabs.display, "-tabs-bottom")
        self.refresh_statusbar()
        self._report_herdr()

    def _report_herdr(self) -> None:
        """Tell Herdr what this process as a whole is doing.

        A Herdr pane has one state and paimon has many conversations, so the
        most urgent one wins: an agent in a tab the user is not looking at is
        still paimon working, or paimon waiting on an answer.
        """
        if self._herdr is None:
            return
        sessions = self.sessions
        if any(pane.needs_confirm for pane in self._panes):
            state = herdr.BLOCKED
        elif any(pane.is_busy or any(job.agent is not None and job.running
                                     for job in pane.agent.jobs.values()) for pane in sessions):
            # An agent still running in the background counts: its caller is
            # woken when it ends, so "idle" now would announce the work done
            # twice. A background command does not, or a dev server left
            # running would never let the pane go idle.
            state = herdr.WORKING
        else:
            state = herdr.IDLE
        # One pane restores one session: the one on screen, or the last one
        # that was while a command's output is. Never an agent another
        # conversation started, which is not resumed on its own.
        roots = [pane for pane in sessions if pane.agent.session.parent_id is None]
        if self._current in roots:
            self._resume_pane = self._current
        elif self._resume_pane not in roots:
            self._resume_pane = roots[0] if roots else None
        pane = self._resume_pane
        # An untouched session has nothing to come back to. Herdr keeps the
        # last command it was given, so until this one has a turn a restart
        # reopens the session before it.
        if pane is None or not pane.agent.history:
            self._herdr.report(herdr.Report(state))
            return
        session_id = pane.agent.session.id
        resume = [herdr.NAME, "--resume", session_id, "--mode", pane.mode]
        # Read now rather than at launch: it follows a login made from inside
        # the app.
        if self.config.model:
            resume += ["--model", self.config.model]
        self._herdr.report(herdr.Report(state, session_id, (*resume, *self._resume_flags)))

    def _switch_to(self, pane: Pane) -> None:
        self._current = pane
        self._switcher.current = pane.id
        self._sync_panes()
        pane._focus_input()

    def on_pane_state_changed(self, event: Pane.StateChanged) -> None:
        """A pane started or finished something the strip or the bar shows."""
        event.stop()
        # A pane's driver posts this as it unwinds too, which on the way out is
        # after the screen it would redraw has already been pruned.
        if self.screen_stack:
            self._sync_panes()

    def on_pane_tabs_selected(self, event: PaneTabs.Selected) -> None:
        event.stop()
        self._switch_to(event.pane)

    async def action_new_pane(self) -> None:
        source = self._session()
        if source is None:
            return
        if len(self._panes) >= MAX_PANES:
            self.pane.notice(Content.from_markup(
                "[$text-warning]Already at $max panes[/]", max=str(MAX_PANES)))
            return
        try:
            agent = Agent.open(cwd=source.cwd, mode=source.mode, config=self.config)
        except SessionError as exc:
            self.pane.notice(Content.from_markup(
                "[$text-error b]Cannot open a pane:[/] $body", body=str(exc)))
            return
        pane = self._make_pane(agent)
        # Current before mounting: the pane focuses its prompt in on_mount, and
        # only does so if it is the one on screen.
        self._current = pane
        await self._switcher.add_content(pane, set_current=True)
        self._sync_panes()

    def _make_pane(self, agent: Agent) -> SessionPane:
        """Register a pane for an agent. The caller mounts it."""
        pane = SessionPane(agent, id=f"pane-{self._next_pane}")
        self._next_pane += 1
        self._panes.append(pane)
        return pane

    async def action_close_pane(self) -> None:
        if len(self._panes) == 1:
            self.pane.notice(Content.from_markup(
                "[$text-muted]The last pane stays open — Ctrl+C quits[/]"))
            return
        await self._drop_pane(self._current)

    async def _drop_pane(self, pane: Pane) -> None:
        """Close a pane and take it off the screen. The one path out."""
        if pane not in self._panes:
            return
        index = self._panes.index(pane)
        pane.close()
        self._panes.remove(pane)
        if not self._panes:
            # The app always has one conversation in it.
            await self._replace_last_pane(pane)
        current = self._panes[min(index, len(self._panes) - 1)]
        await pane.remove()
        if self._current is pane:
            self._switch_to(current)
        else:
            self._sync_panes()

    async def _replace_last_pane(self, closed: Pane) -> None:
        try:
            agent = Agent.open(cwd=closed.cwd, mode=closed.mode, config=self.config)
        except SessionError:
            self.exit()  # nothing left to show, and no session to show it in
            return
        await self._switcher.add_content(self._make_pane(agent))

    # ---- background jobs ----------------------------------------------------

    async def open_job(self, owner: SessionPane, job_id: str, job: Job) -> None:
        """Open a background pane for an agent or a command ``owner``'s agent started.

        It is mounted hidden and never focused: the user asked for work to be
        done, not for their keyboard to move.
        """
        if len(self._panes) >= MAX_PANES:
            raise PaneLimitError(f"all {MAX_PANES} panes are in use; close one first")
        pane_id = f"pane-{self._next_pane}"
        if job.kind == "agent":
            pane = AgentPane(owner.agent, job_id, job, id=pane_id)
        else:
            pane = CommandPane(job_id, job, cwd=owner.cwd, mode=owner.mode, id=pane_id)
        self._next_pane += 1
        self._panes.append(pane)
        try:
            await self._switcher.add_content(pane)
        except BaseException:
            self._panes.remove(pane)
            raise
        self._sync_panes()

    def _step_pane(self, step: int) -> None:
        if len(self._panes) > 1:
            index = self._panes.index(self._current)
            self._switch_to(self._panes[(index + step) % len(self._panes)])

    def action_next_pane(self) -> None:
        self._step_pane(1)

    def action_prev_pane(self) -> None:
        self._step_pane(-1)

    def action_goto_attention(self) -> None:
        """Jump to the next pane blocked on a confirmation or a question.

        A background pane waiting for permission blocks whoever is waiting on
        it, so there has to be one key that always lands on it. The current
        pane comes last, for a panel there that lost the keyboard to a click.
        """
        index = self._panes.index(self._current)
        rotated = self._panes[index + 1:] + self._panes[:index + 1]
        for pane in rotated:
            if pane.needs_confirm:
                self._switch_to(pane)
                pane.focus_attention()
                return

    def save_config(self, **fields) -> None:
        """Persist config fields without blocking or crashing the UI.

        save() waits on the cross-process lock (up to 10s when another
        instance is stuck) and fsyncs twice, and it raises ConfigError on a
        corrupt file, so it runs on a worker thread and a failure lands as a
        notice instead of an exception tearing down the app.
        """
        def _write() -> None:
            try:
                self.config.save(**fields)
            except PaimonError as exc:
                self.call_from_thread(
                    self.pane.notice,
                    Content.from_markup("[$text-error b]Config not saved:[/] $body",
                                        body=str(exc)))
        self.run_worker(_write, thread=True, group="config-save")

    def _watch_theme(self, theme_name: str) -> None:
        super()._watch_theme(theme_name)
        if self._persist_theme_changes:
            self.save_config(theme=theme_name)

    def compose(self) -> ComposeResult:
        yield self._tabs
        yield self._switcher
        yield Static(id="statusbar")

    def on_mount(self) -> None:
        self._sync_panes()
        if not self.config.model:
            self.action_login()
        elif self._pick_session:
            self.action_resume_session()

    def on_unmount(self) -> None:
        """Hand back what outlives the process, on every way out.

        A background command is in its own process group, so nothing kills it
        for us: without this it is reparented to init and keeps running after
        paimon is gone. Sessions are unlocked here for symmetry, and because
        the app can be torn down and rebuilt inside one process.
        """
        for pane in self._panes:
            pane.shutdown()
        if self._herdr is not None:
            self._herdr.release()

    def on_key(self, event: events.Key) -> None:
        """Keys that reached the app were claimed by no pane.

        Focus can land outside every pane — clicking the status bar clears it —
        and the stray-typing handler only bubbles from inside a pane, so hand
        the key to the current one.
        """
        self.pane.on_key(event)

    # ---- pane actions -------------------------------------------------------

    # The palette and the key bindings live on the app, but every one of these
    # acts on a single conversation, so they only route to the current pane.

    # A pane showing an agent or a command is nobody's to steer, so each of
    # these is routed to a conversation rather than to whatever is on screen.

    def action_new_session(self) -> None:
        if (pane := self._session()) is not None:
            pane.new_session()

    def action_fork_session(self) -> None:
        if (pane := self._session()) is not None:
            pane.fork_session()

    def action_resume_session(self) -> None:
        if (pane := self._session()) is not None:
            pane.resume_session()

    def action_compact(self) -> None:
        if (pane := self._session()) is not None:
            pane.compact()

    def action_cycle_mode(self) -> None:
        if (pane := self._session()) is not None:
            pane.cycle_mode()

    def action_interrupt(self) -> None:
        if isinstance(self.pane, SessionPane):
            self.pane.interrupt()

    def action_toggle_reasoning(self) -> None:
        # Flipped on the instance up front: the write happens on a worker
        # thread, and the UI must reflect the toggle immediately.
        self.config.show_reasoning = not self.config.show_reasoning
        self.save_config(show_reasoning=self.config.show_reasoning)
        state = "streamed live" if self.config.show_reasoning else "folded"
        self.pane.notice(Content.from_markup(f"[$text-muted]Thinking: {state}[/]"))

    def action_toggle_recap(self) -> None:
        """Turn the after-idle recap off (or back on), for every pane.

        Config is process-wide, so one switch covers all panes; turning it off
        also drops recaps already armed, which check the flag only when armed.
        """
        self.config.recap_enabled = not self.config.recap_enabled
        self.save_config(recap_enabled=self.config.recap_enabled)
        if not self.config.recap_enabled:
            for pane in self.sessions:
                pane._cancel_recap()
        state = "on" if self.config.recap_enabled else "off"
        self.pane.notice(Content.from_markup(f"[$text-muted]Idle recap: {state}[/]"))

    # ---- login --------------------------------------------------------------

    def _config_is_busy(self) -> bool:
        """Whether a running turn makes it unsafe to rewrite the config.

        Config is process-wide, and a turn re-reads the model at the top of
        every step (Agent._model), so a login landing mid-turn silently
        swaps providers between two tool calls. It is refused while any pane
        is running a turn.
        """
        return any(pane.is_busy for pane in self.panes)

    def action_login(self) -> None:
        if self._config_is_busy():
            self.pane.notice(Content.from_markup("[$text-muted]Busy — log in after this turn[/]"))
            return

        def _done(completed: bool | None) -> None:
            if completed:
                self.pane.notice(
                    Content.from_markup(
                        "[$text-success b]Logged in.[/]  [$text-muted]$model[/]",
                        model=self.config.model or "",
                    )
                )
            elif not self.config.model:
                self.pane.notice(Content.from_markup("[$text-warning]Login cancelled — no model configured.[/]"))
                self.exit()
            self.refresh_statusbar()
            self._report_herdr()
            self.pane._focus_input()

        self.push_screen(LoginScreen(), _done)

    # ---- model --------------------------------------------------------------

    def action_switch_model(self) -> None:
        """Change the configured model without asking for credentials again."""
        pane = self._session()
        if pane is None:
            return
        if self._config_is_busy():
            self.pane.notice(Content.from_markup("[$text-muted]Busy — switch model after this turn[/]"))
            return

        def _done(model: str | None) -> None:
            # A turn may have started while the picker was up.
            if not model or self._config_is_busy():
                return
            self.config.model = model
            self.save_config(model=model)
            self.pane.notice(Content.from_markup("[$text-muted]Model: $model[/]", model=model))
            self.refresh_statusbar()
            self._report_herdr()

        self.push_screen(PickerScreen("Switch model", pane.agent.available_models()), _done)

    # ---- status bar ---------------------------------------------------------

    def refresh_statusbar(self, tokens: int | None = None) -> None:
        # A redraw queued behind a closing pane can arrive after the screen it
        # would draw on has been pruned, on the way out. Nothing to draw then.
        bars = self.query("#statusbar")
        if not bars:
            return
        pane = self.pane
        if isinstance(pane, SessionPane):
            parts = self._session_status(pane, tokens)
        elif isinstance(pane, AgentPane):
            agent = pane.job.agent
            parts = [f"agent {pane.job_id}", f"{pane.mode} mode", agent.model_name or "no model",
                     f"session {agent.session.id[:8]}"]
        else:
            parts = [f"command {pane.job_id}", pane.status_text, pane.command.command]
        # A pane blocked on a confirmation the user cannot see blocks whatever
        # is waiting on it, so the count follows them to every other pane.
        waiting = sum(1 for other in self._panes if other is not pane and other.needs_confirm)
        # Assembled rather than marked up: model names and session ids are not
        # markup and must not be parsed as any.
        line = Content("  ·  ".join(parts))
        if waiting:
            line = line.append_text("  ·  ").append_text(
                f"{waiting} waiting on you (ctrl+g)", "$text-warning")
        bars.first(Static).update(line)

    def _session_status(self, pane: SessionPane, tokens: int | None) -> list[str]:
        # The agent's model, not the config's: a pane may override it.
        parts = [f"{pane.mode} mode", pane.agent.model_name or "no model",
                 f"session {pane.agent.session.id[:8]}"]
        agents = sum(1 for job in pane.agent.jobs.values() if job.kind == "agent" and job.running)
        if agents:
            parts.append(f"{agents} agent{'s' if agents > 1 else ''} running")
        # The last measurement stands until a new one arrives: the bar is
        # redrawn whenever any pane changes state, not only after a turn.
        if tokens is None:
            tokens = pane._tokens
        else:
            pane._tokens = tokens
        if tokens is not None:
            window = pane.agent.context_window()
            if window:
                parts.append(f"context {tokens / 1000:.1f}k/{window / 1000:.0f}k ({tokens / window:.0%})")
            else:
                # Unknown window: auto-compaction cannot trigger, so say so
                # instead of looking like it is merely waiting to.
                parts.append(f"context ~{tokens / 1000:.1f}k tokens "
                             "(auto-compaction off: unknown context window)")
        if pane._tps is not None:
            parts.append(f"{pane._tps:.0f} tokens per second")
        if pane._cache_hit is not None:
            parts.append(f"cache hit {pane._cache_hit:.0%}")
        return parts

    @work(exclusive=True, group="statusbar")
    async def update_statusbar_tokens(self) -> None:
        pane = self.pane
        if not isinstance(pane, SessionPane):
            return
        tokens = await pane.agent.count_context_tokens()
        # The pane may have been swapped out while the count ran on a thread.
        if pane is self.pane:
            self.refresh_statusbar(tokens)
