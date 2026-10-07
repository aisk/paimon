"""Login flow: pick provider → pick model → enter api_base → enter api_key.

Provider and model lists come from pydantic-ai's static ``KnownModelName``
catalog; no network calls are made. The catalog keeps every model a provider
ever served, so the model list is cut down to the current ones. The picker
accepts free-typed entries, so unlisted providers, brand-new model names and
the models left out still work.

The provider doubles as the wire dialect, so the catalog is narrowed to the
ones whose SDK ships with Paimon — offering the rest would only produce an
import error later.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from textual.content import Content

from paimon import llm
from paimon.errors import PaimonError
from paimon.llm import CHATGPT_PROVIDER, is_provider_available


def _providers() -> list[str]:
    names = {name.split(":", 1)[0] for name in llm.known_models() if ":" in name}
    return sorted(name for name in names | {CHATGPT_PROVIDER} if is_provider_available(name))


class PickerScreen(ModalScreen[Optional[str]]):
    """Filterable list picker. Type to filter, Up/Down to move, Enter to select."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title: str, options: list[str]) -> None:
        super().__init__()
        self._title = title
        self._options = options
        self._filtered: list[str] = options

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-box"):
            yield Static(self._title, id="picker-title")
            yield Input(placeholder="Type to filter, ↑↓ to move, Enter to select", id="picker-filter")
            yield OptionList(id="picker-list")

    def on_mount(self) -> None:
        self._populate("")
        self.query_one("#picker-filter", Input).focus()

    def _populate(self, query: str) -> None:
        q = query.strip().lower()
        self._filtered = [o for o in self._options if q in o.lower()]
        ol = self.query_one("#picker-list", OptionList)
        ol.clear_options()
        for o in self._filtered:
            ol.add_option(Option(o, id=o))
        if self._filtered:
            ol.action_first()

    @on(Input.Changed)
    def _on_filter(self, event: Input.Changed) -> None:
        self._populate(event.value)

    @on(Input.Submitted)
    def _on_filter_submit(self, event: Input.Submitted) -> None:
        event.prevent_default()
        event.stop()
        self._confirm()

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    @on(OptionList.OptionHighlighted)
    def _on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        # Keep filter input focused so typing keeps narrowing the list.
        self.query_one("#picker-filter", Input).focus()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()
            return
        if event.key == "down":
            self.query_one("#picker-list", OptionList).action_cursor_down()
            event.prevent_default()
            event.stop()
        elif event.key == "up":
            self.query_one("#picker-list", OptionList).action_cursor_up()
            event.prevent_default()
            event.stop()

    def _confirm(self) -> None:
        ol = self.query_one("#picker-list", OptionList)
        if ol.highlighted is not None and 0 <= ol.highlighted < len(self._filtered):
            self.dismiss(self._filtered[ol.highlighted])
            return
        # Nothing matched: accept the typed text verbatim, so entries missing
        # from the static catalog (new models, unlisted providers) still work.
        typed = self.query_one("#picker-filter", Input).value.strip()
        if typed:
            self.dismiss(typed)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptScreen(ModalScreen[Optional[str]]):
    """Single-line text input. Enter returns the value, Escape cancels."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title: str, *, password: bool = False, placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._password = password
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-screen-box"):
            yield Static(self._title, id="prompt-screen-title")
            yield Input(
                placeholder=self._placeholder,
                password=self._password,
                id="prompt-screen-input",
            )

    def on_mount(self) -> None:
        self.query_one("#prompt-screen-input", Input).focus()

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        event.prevent_default()
        event.stop()
        self.dismiss(event.value)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatGPTLoginScreen(ModalScreen[bool]):
    """Browser sign-in for the ChatGPT plan. Returns True once the credential
    is stored, False when cancelled or refused."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, profile: str) -> None:
        super().__init__()
        self._profile = profile
        self._pasted: asyncio.Future = asyncio.get_running_loop().create_future()

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-screen-box"):
            yield Static("Sign in with ChatGPT", id="prompt-screen-title")
            yield Static("Opening the browser…", id="chatgpt-login-status")
            yield Input(placeholder="or paste the final redirect URL here", id="prompt-screen-input")

    def on_mount(self) -> None:
        self.query_one("#prompt-screen-input", Input).focus()
        self._flow()

    def _show_url(self, url: str) -> None:
        from paimon import chatgpt

        lead = ("Finish signing in in the browser. If it did not open, visit:"
                if chatgpt.open_browser(url) else "Open this address in a browser to sign in:")
        self.query_one("#chatgpt-login-status", Static).update(
            Content.from_markup("$lead\n\n$url", lead=lead, url=url))

    @work
    async def _flow(self) -> None:
        from paimon import chatgpt

        try:
            await chatgpt.login(self._profile, self._show_url, self._pasted)
        except PaimonError as exc:
            self.app.pane.notice(Content.from_markup(  # type: ignore[attr-defined]
                "[$text-error b]ChatGPT sign-in failed:[/] $body", body=str(exc)))
            self.dismiss(False)
            return
        self.dismiss(True)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        event.prevent_default()
        event.stop()
        if event.value.strip() and not self._pasted.done():
            self._pasted.set_result(event.value)

    def action_cancel(self) -> None:
        # Dismissing unmounts the screen, which cancels the login worker and
        # with it the callback server.
        self.dismiss(False)


class LoginScreen(ModalScreen[bool]):
    """Multi-step login. Returns True on completion, False if cancelled anywhere."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def compose(self) -> ComposeResult:
        # The login flow is driven by sub-screens; this surface is just a backdrop.
        yield Static("Login required — opening provider selection…", id="login-status")

    def on_mount(self) -> None:
        self._flow()

    @work
    async def _flow(self) -> None:
        provider = await self.app.push_screen_wait(PickerScreen("Select provider", _providers()))
        if not provider:
            self.dismiss(False)
            return

        model = await self.app.push_screen_wait(PickerScreen(f"Select model · {provider}", llm.models_for(provider)))
        if not model:
            self.dismiss(False)
            return

        config = self.app.config  # type: ignore[attr-defined] (pushed only by PaimonApp)
        fields: dict = {"model": f"{provider}:{model}"}
        if provider == CHATGPT_PROVIDER:
            # The plan's credential is a browser login the sign-in stores
            # itself, so there is no endpoint or key to ask for.
            if not await self.app.push_screen_wait(ChatGPTLoginScreen(config.profile)):
                self.dismiss(False)
                return
        else:
            api_base = await self.app.push_screen_wait(
                PromptScreen(
                    "API base (leave blank for provider default)",
                    placeholder="https://api.example.com/v1",
                )
            )
            if api_base is None:
                self.dismiss(False)
                return

            api_key = await self.app.push_screen_wait(
                PromptScreen("API key", password=True, placeholder="sk-…")
            )
            if api_key is None:
                self.dismiss(False)
                return
            fields["api_base"] = api_base.strip() or None
            fields["api_key"] = api_key.strip() or None
        try:
            # On a thread: save() can wait on the cross-process config lock.
            await asyncio.to_thread(config.save, **fields)
        except PaimonError as exc:
            # The credentials just typed must not die with the write. Apply
            # them to this run and let the user repair the file afterwards.
            config.model = fields["model"]
            entry = config.providers.setdefault(provider, {})
            for key in ("api_base", "api_key"):
                if key not in fields:
                    continue
                if fields[key] is None:
                    entry.pop(key, None)
                else:
                    entry[key] = fields[key]
            if not entry:
                config.providers.pop(provider, None)
            self.app.pane.notice(Content.from_markup(  # type: ignore[attr-defined]
                "[$text-warning b]Logged in for this run only, config not saved:[/] $body",
                body=str(exc)))
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
