"""Textual TUI for the Paimon agent.

The app is a container for panes: it owns the config, the theme, the command
palette, the global bindings and the status bar. Everything belonging to one
conversation lives in ``SessionPane``, and everything belonging to one
background command in ``TaskPane``.
"""

from functools import partial

from textual import events, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.content import Content
from textual.widgets import ContentSwitcher, Static

from . import compaction, tools
from .agent import Agent
from .config import DEFAULT_PROFILE, Config, list_profiles
from .errors import PaimonError
from .login import LoginScreen, PickerScreen, PromptScreen
from .pane import Pane, SessionPane
from .session import SessionError
from .jobs import CommandJob, Job
from .supervisor import Supervisor, SupervisorError
from .tabs import PaneTabs
from .taskpane import TaskPane
from .ui import PromptInput

# Every pane holds a live agent and its context, or a live process, so panes
# cost model requests, memory and file descriptors, not just a row in the
# strip. A soft cap keeps a runaway loop of "one more session" from taking the
# app down with it.
MAX_PANES = 8


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
        Binding("ctrl+g", "goto_attention", "Go to a pane awaiting confirmation", priority=True),
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
                "Switch profile",
                "Use another profile's account and model, or create a new one",
                self.action_switch_profile,
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

    def __init__(self, agent: Agent, *, resumed: bool = False, pick_session: bool = False) -> None:
        self._persist_theme_changes = False
        super().__init__()
        self.config = agent.config
        # Agents live in panes, so the supervisor borrows the app to open and
        # close them; everything else about them it owns itself.
        self._supervisor = Supervisor(launch=self._launch_agent, close=self._close_job,
                                      launch_task=self._launch_task, limit=MAX_PANES)
        pane = SessionPane(agent, job_id=self._supervisor.new_id(), resumed=resumed,
                           id="pane-1", supervisor=self._supervisor)
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
        there is: the palette and the key bindings stay usable while a task's
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
        # cost three rows rather than four.
        if self.is_mounted:
            self.screen.set_class(self._tabs.display, "-tabs-bottom")
        self.refresh_statusbar()

    def _switch_to(self, pane: Pane) -> None:
        self._current = pane
        self._switcher.current = pane.id
        self._sync_panes()
        pane._focus_input()

    def on_pane_state_changed(self, event: Pane.StateChanged) -> None:
        """A pane started or finished something the strip or the bar shows."""
        event.stop()
        # A job's driver posts this as it unwinds too, which on the way out is
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

    def _make_pane(self, agent: Agent, *, parent=None) -> SessionPane:
        """Register a pane for an agent. The caller mounts it."""
        pane = SessionPane(agent, job_id=self._supervisor.new_id(), parent=parent,
                           id=f"pane-{self._next_pane}", supervisor=self._supervisor)
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
        # A job whose pane is gone is over, but what it produced stays
        # readable: whoever started it may still be about to ask.
        self._supervisor.released(pane.job)
        self._panes.remove(pane)
        if not self._panes:
            # Closing a pane stops the agents it started, so the last two can
            # go at once. The app always has one conversation in it.
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

    # ---- agents -------------------------------------------------------------

    async def _launch_agent(self, job_id: str, parent, model: str | None) -> Job:
        """Open a background pane for an agent another pane asked for.

        It is mounted hidden and never focused: a pane the user did not open
        must not take the keyboard, or their next keystroke answers a
        confirmation they never saw.
        """
        if len(self._panes) >= MAX_PANES:
            raise SupervisorError(f"all {MAX_PANES} panes are in use; close one first")
        # Only conversations have an agent, so a task's pane is never asked
        # about one; the fallback is for an agent whose own pane has gone.
        owner = next((pane for pane in self.sessions if pane.agent is parent), None)
        if owner is None and (owner := self._session()) is None:
            raise SupervisorError("there is no conversation to start an agent from")
        agent = Agent.open(
            cwd=owner.cwd, mode=owner.mode, config=self.config, model_override=model,
            # Marked as this session's child so it stays out of the session
            # lists and out of `paimon -c`.
            parent=owner.agent.session.id,
            toolset=tools.without(tools.REGISTRY, tools.SUBAGENT_DENIED),
        )
        pane = SessionPane(agent, job_id=job_id, parent=parent,
                           id=f"pane-{self._next_pane}", supervisor=self._supervisor)
        self._next_pane += 1
        self._panes.append(pane)
        try:
            await self._switcher.add_content(pane)
        except BaseException:
            # Agent.open took the session lock, and nothing else will ever
            # release it: without this the session stays busy for the life of
            # the process, even to `paimon -r`.
            self._panes.remove(pane)
            agent.session.unlock()
            raise
        self._sync_panes()
        return pane.job

    async def _launch_task(self, job_id: str, command, description: str) -> Job:
        """Open a background pane for a command a session asked to run.

        Hidden and unfocused for the same reason a spawned agent's pane is: the
        user asked for a command, not for their keyboard to move.
        """
        if len(self._panes) >= MAX_PANES:
            raise SupervisorError(f"all {MAX_PANES} panes are in use; close one first")
        owner = self._session()
        if owner is None:
            raise SupervisorError("there is no conversation to attach a command to")
        job = CommandJob(job_id, command, description, parent=owner.agent)
        pane = TaskPane(job, cwd=owner.cwd, mode=owner.mode, id=f"pane-{self._next_pane}")
        self._next_pane += 1
        self._panes.append(pane)
        try:
            await self._switcher.add_content(pane)
        except BaseException:
            self._panes.remove(pane)
            raise
        job.start()
        self._sync_panes()
        return job

    def _close_job(self, job: Job) -> None:
        """The supervisor stopped a job: take the pane showing it down too."""
        pane = next((pane for pane in self._panes if pane.job is job), None)
        if pane is None:
            return
        pane.close()  # cancel the turn, release the session, stop the command
        self.call_later(self._drop_pane, pane)

    def _step_pane(self, step: int) -> None:
        if len(self._panes) > 1:
            index = self._panes.index(self._current)
            self._switch_to(self._panes[(index + step) % len(self._panes)])

    def action_next_pane(self) -> None:
        self._step_pane(1)

    def action_prev_pane(self) -> None:
        self._step_pane(-1)

    def action_goto_attention(self) -> None:
        """Jump to the next pane blocked on a confirmation.

        A background pane waiting for permission blocks whoever is waiting on
        it, so there has to be one key that always lands on it.
        """
        index = self._panes.index(self._current)
        rotated = self._panes[index + 1:] + self._panes[:index + 1]
        for pane in rotated:
            if pane.needs_confirm:
                self._switch_to(pane)
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

    # A task's pane has no session, no mode and no turn, so each of these is
    # routed to a conversation rather than to whatever is on screen.

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
        every step (Agent._model), so a login or profile switch landing
        mid-turn silently swaps providers between two tool calls. Both refuse
        while any pane is running a turn.
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
            self.pane._focus_input()

        self.push_screen(LoginScreen(), _done)

    # ---- profiles -----------------------------------------------------------

    _NEW_PROFILE = "New profile…"

    def _apply_config(self, config: Config) -> None:
        """Config is process-wide, so every pane's agent moves with it."""
        self.config = config
        for pane in self.sessions:
            pane.agent.config = config

    @work
    async def action_switch_profile(self) -> None:
        if self._config_is_busy():
            self.pane.notice(Content.from_markup("[$text-muted]Busy — switch profiles after this turn[/]"))
            return
        current = self.config.profile
        labels = {f"{name} (current)" if name == current else name: name
                  for name in list_profiles()}
        choice = await self.push_screen_wait(
            PickerScreen("Switch profile", [*labels, self._NEW_PROFILE]))
        if choice == self._NEW_PROFILE:
            choice = await self.push_screen_wait(PromptScreen("New profile name"))
        # An unlisted typed name is accepted too: switching to a profile that
        # does not exist yet is how one gets created.
        name = labels.get(choice, choice) if choice else None
        if name is None or name == current:
            self.pane._focus_input()
            return
        try:
            switched = Config.load(name)
        except (ValueError, PaimonError) as exc:
            self.pane.notice(Content.from_markup("[$text-error b]Cannot switch:[/] $body", body=str(exc)))
            self.pane._focus_input()
            return
        previous_config = self.config
        self._apply_config(switched)
        if not self.config.model:
            completed = await self.push_screen_wait(LoginScreen())
            if not completed:
                self._apply_config(previous_config)
                self.pane.notice(Content.from_markup("[$text-muted]Profile switch cancelled[/]"))
                self.pane._focus_input()
                return
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self.pane.notice(Content.from_markup(
            "[$text-success b]Profile:[/] $name  [$text-muted]$model[/]",
            name=name, model=self.config.model or ""))
        self.refresh_statusbar()
        self.pane._focus_input()

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
        else:
            parts = [f"task {pane.job.job_id}", pane.status_text, pane.command.command]
        # A pane blocked on a confirmation the user cannot see blocks whatever
        # is waiting on it, so the count follows them to every other pane.
        waiting = sum(1 for other in self._panes if other is not pane and other.needs_confirm)
        # Assembled rather than marked up: model names and session ids are not
        # markup and must not be parsed as any.
        line = Content("  ·  ".join(parts))
        if waiting:
            line = line.append_text("  ·  ").append_text(
                f"{waiting} awaiting confirmation (ctrl+g)", "$text-warning")
        bars.first(Static).update(line)

    def _session_status(self, pane: SessionPane, tokens: int | None) -> list[str]:
        # The agent's model, not the config's: a pane may override it.
        parts = [f"{pane.mode} mode", pane.agent.model_name or "no model",
                 f"session {pane.agent.session.id[:8]}"]
        if self.config.profile != DEFAULT_PROFILE:
            parts.insert(1, f"profile {self.config.profile}")
        # The last measurement stands until a new one arrives: the bar is
        # redrawn whenever any pane changes state, not only after a turn.
        if tokens is None:
            tokens = pane._tokens
        else:
            pane._tokens = tokens
        if tokens is not None:
            window = compaction.context_window(pane.agent.model_name,
                                               self.config.compaction_context_window)
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
