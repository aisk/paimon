"""Textual TUI for the Paimon agent.

The app is a container for panes: it owns the config, the theme, the command
palette, the global bindings and the status bar. Everything belonging to one
conversation lives in ``SessionPane``.
"""

from textual import events, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.content import Content
from textual.widgets import Static

from . import compaction
from .agent import Agent
from .config import DEFAULT_PROFILE, Config, list_profiles
from .login import LoginScreen, PickerScreen, PromptScreen
from .pane import SessionPane


class PaimonApp(App):
    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("escape", "interrupt", "Interrupt"),
        # priority so the focused PromptInput (a TextArea) can't swallow the key
        Binding("shift+tab", "cycle_mode", "Cycle permission mode", priority=True),
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
                "Compact context",
                "Summarize the earlier conversation now instead of waiting for the context to fill",
                self.action_compact,
            ),
            SystemCommand("New session", "Start a new empty session", self.action_new_session),
            SystemCommand("Fork session", "Copy this conversation into a new session and continue there",
                          self.action_fork_session),
            SystemCommand("Resume session", "Pick an earlier session in this directory to resume",
                          self.action_resume_session),
        ]

    def __init__(self, agent: Agent, *, resumed: bool = False, pick_session: bool = False) -> None:
        self._persist_theme_changes = False
        super().__init__()
        self.config = agent.config
        self._pane = SessionPane(agent, resumed=resumed, id="pane-1")
        self._pick_session = pick_session
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self._persist_theme_changes = True

    # ---- panes --------------------------------------------------------------

    @property
    def pane(self) -> SessionPane:
        """The pane on screen. There is exactly one until tabs land."""
        return self._pane

    @property
    def panes(self) -> list[SessionPane]:
        return [self._pane]

    def _watch_theme(self, theme_name: str) -> None:
        super()._watch_theme(theme_name)
        if self._persist_theme_changes:
            self.config.save(theme=theme_name)

    def compose(self) -> ComposeResult:
        yield self._pane
        yield Static(id="statusbar")

    def on_mount(self) -> None:
        self.refresh_statusbar()
        if not self.config.model:
            self.action_login()
        elif self._pick_session:
            self.action_resume_session()

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

    def action_new_session(self) -> None:
        self.pane.new_session()

    def action_fork_session(self) -> None:
        self.pane.fork_session()

    def action_resume_session(self) -> None:
        self.pane.resume_session()

    def action_compact(self) -> None:
        self.pane.compact()

    def action_cycle_mode(self) -> None:
        self.pane.cycle_mode()

    def action_interrupt(self) -> None:
        self.pane.interrupt()

    def action_toggle_reasoning(self) -> None:
        self.config.save(show_reasoning=not self.config.show_reasoning)
        state = "streamed live" if self.config.show_reasoning else "folded"
        self.pane._add(Content.from_markup(f"[$text-muted]Thinking: {state}[/]"))

    # ---- login --------------------------------------------------------------

    def _config_is_busy(self) -> bool:
        """Whether a running turn makes it unsafe to rewrite the config.

        Config is process-wide, and a turn re-reads the model at the top of
        every step (Agent._model), so a login or profile switch landing
        mid-turn silently swaps providers between two tool calls. Both refuse
        while any pane is running a turn.
        """
        return any(pane.is_turn_running for pane in self.panes)

    def action_login(self) -> None:
        if self._config_is_busy():
            self.pane._add(Content.from_markup("[$text-muted]Busy — log in after this turn[/]"))
            return

        def _done(completed: bool | None) -> None:
            if completed:
                self.pane._add(
                    Content.from_markup(
                        "[$text-success b]Logged in.[/]  [$text-muted]$model[/]",
                        model=self.config.model or "",
                    )
                )
            elif not self.config.model:
                self.pane._add(Content.from_markup("[$text-warning]Login cancelled — no model configured.[/]"))
                self.exit()
            self.refresh_statusbar()
            self.pane._focus_prompt()

        self.push_screen(LoginScreen(), _done)

    # ---- profiles -----------------------------------------------------------

    _NEW_PROFILE = "New profile…"

    def _apply_config(self, config: Config) -> None:
        """Config is process-wide, so every pane's agent moves with it."""
        self.config = config
        for pane in self.panes:
            pane.agent.config = config

    @work
    async def action_switch_profile(self) -> None:
        if self._config_is_busy():
            self.pane._add(Content.from_markup("[$text-muted]Busy — switch profiles after this turn[/]"))
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
            self.pane._focus_prompt()
            return
        try:
            switched = Config.load(name)
        except ValueError as exc:
            self.pane._add(Content.from_markup("[$text-error b]Cannot switch:[/] $body", body=str(exc)))
            self.pane._focus_prompt()
            return
        previous_config = self.config
        self._apply_config(switched)
        if not self.config.model:
            completed = await self.push_screen_wait(LoginScreen())
            if not completed:
                self._apply_config(previous_config)
                self.pane._add(Content.from_markup("[$text-muted]Profile switch cancelled[/]"))
                self.pane._focus_prompt()
                return
        if self.config.theme in self.available_themes:
            self.theme = self.config.theme
        self.pane._add(Content.from_markup(
            "[$text-success b]Profile:[/] $name  [$text-muted]$model[/]",
            name=name, model=self.config.model or ""))
        self.refresh_statusbar()
        self.pane._focus_prompt()

    # ---- status bar ---------------------------------------------------------

    def refresh_statusbar(self, tokens: int | None = None) -> None:
        pane = self.pane
        # The agent's model, not the config's: a pane may override it.
        parts = [f"{pane.mode} mode", pane.agent.model_name or "no model",
                 f"session {pane.agent.session.id[:8]}"]
        if self.config.profile != DEFAULT_PROFILE:
            parts.insert(1, f"profile {self.config.profile}")
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
            parts.append(f"{pane._tps:.0f} tps")
        self.query_one("#statusbar", Static).update(Content("  ·  ".join(parts)))

    @work(exclusive=True, group="statusbar")
    async def update_statusbar_tokens(self) -> None:
        pane = self.pane
        tokens = await pane.agent.count_context_tokens()
        # The pane may have been swapped out while the count ran on a thread.
        if pane is self.pane:
            self.refresh_statusbar(tokens)
