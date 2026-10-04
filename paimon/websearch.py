"""Web search, behind the web_search tool.

A backend is a blocking ``(query, max_results) -> list[Result]``. There is one
today, ddgs, a metasearch library that needs no API key and picks among
several public search engines by itself. Adding another means writing one more
such function and choosing between them in ``search``.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_RESULTS = 5
MAX_RESULTS = 10

# ddgs applies its timeout to each HTTP request, and its "auto" backend may try
# several engines in turn, so one search can take a multiple of it. The total
# bound is what keeps an unreachable network from looking like a hang.
_REQUEST_TIMEOUT = 15
_TOTAL_TIMEOUT = 45.0


class WebSearchError(Exception):
    """A search that failed for a reason the model can be told about."""


@dataclass(frozen=True)
class Result:
    title: str
    url: str
    snippet: str


Backend = Callable[[str, int], list[Result]]


def _ddgs(query: str, max_results: int) -> list[Result]:
    # Imported here: ddgs pulls in lxml and a native HTTP client, which no
    # run that never searches should pay for at startup.
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

    try:
        rows = DDGS(timeout=_REQUEST_TIMEOUT).text(query, max_results=max_results, backend="auto")
    except RatelimitException as exc:
        raise WebSearchError(f"the search engines rate-limited the request ({exc}); try again later") from exc
    except TimeoutException as exc:
        raise WebSearchError(f"the search engines did not answer in time ({exc})") from exc
    except DDGSException as exc:
        # ddgs reports an empty result set as an error rather than an empty list.
        if "no results" in str(exc).lower():
            return []
        raise WebSearchError(str(exc)) from exc
    return [Result(title=str(row.get("title") or ""), url=str(row.get("href") or ""),
                   snippet=str(row.get("body") or ""))
            for row in rows[:max_results]]


def render(query: str, results: list[Result]) -> str:
    """The numbered plain-text list the model reads."""
    if not results:
        return f"No results for {query!r}."
    blocks = []
    for index, result in enumerate(results, 1):
        lines = [f"{index}. {result.title or '(untitled)'}", f"   {result.url}"]
        snippet = " ".join(result.snippet.split())
        if snippet:
            lines.append(f"   {snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _start(backend: Backend, query: str, max_results: int) -> asyncio.Future:
    """Run the blocking backend in a daemon thread; the future carries its outcome.

    Not asyncio.to_thread: a search given up on (Esc, the total timeout) leaves
    its thread running, and the default executor's threads are joined when the
    loop shuts down and again at interpreter exit, so one stuck request would
    hold up leaving paimon. A daemon thread is simply abandoned.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def settle(result: object, error: object) -> None:
        if future.done():  # cancelled along with the caller
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def work() -> None:
        result = error = None
        try:
            result = backend(query, max_results)
        except BaseException as exc:  # noqa: BLE001 — handed to the caller, whatever it is
            error = exc
        try:
            loop.call_soon_threadsafe(settle, result, error)
        except RuntimeError:
            pass  # the loop is gone: nobody is waiting any more

    threading.Thread(target=work, name="paimon-web-search", daemon=True).start()
    return future


async def search(args: dict, backend: Optional[Backend] = None) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Error: query must not be empty"
    max_results = args.get("max_results")
    max_results = DEFAULT_RESULTS if max_results is None else max(1, min(int(max_results), MAX_RESULTS))
    future = _start(backend or _ddgs, query, max_results)
    # asyncio.wait rather than wait_for, which would report a timeout raised
    # inside the backend as this one.
    done, _ = await asyncio.wait([future], timeout=_TOTAL_TIMEOUT)
    if not done:
        future.cancel()
        return (f"Error: web search timed out after {_TOTAL_TIMEOUT:g} seconds; "
                "the network may be unreachable")
    try:
        results = future.result()
    except WebSearchError as exc:
        return f"Error: web search failed: {exc}"
    return render(query, results)
