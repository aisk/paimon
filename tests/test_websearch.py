import asyncio
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from paimon import tools, websearch
from paimon.agent import Agent
from paimon.config import Config
from paimon.websearch import Result, WebSearchError


def _results(count: int) -> list[Result]:
    return [Result(title=f"Title {i}", url=f"https://example.com/{i}", snippet=f"snippet {i}")
            for i in range(1, count + 1)]


class SearchTest(unittest.TestCase):
    def search(self, args: dict, backend) -> str:
        return asyncio.run(websearch.search(args, backend=backend))

    def test_results_come_back_as_a_numbered_list(self) -> None:
        result = self.search({"query": "paimon"}, lambda query, n: _results(2))
        self.assertEqual(result, (
            "1. Title 1\n   https://example.com/1\n   snippet 1\n\n"
            "2. Title 2\n   https://example.com/2\n   snippet 2"))

    def test_snippet_whitespace_is_collapsed_and_an_empty_one_is_left_out(self) -> None:
        rows = [Result("A", "https://a", "one\n  two\tthree"), Result("", "https://b", "")]
        result = self.search({"query": "q"}, lambda query, n: rows)
        self.assertEqual(result, "1. A\n   https://a\n   one two three\n\n2. (untitled)\n   https://b")

    def test_max_results_defaults_and_is_clamped(self) -> None:
        seen = []

        def backend(query: str, max_results: int) -> list[Result]:
            seen.append(max_results)
            return []

        self.search({"query": "q"}, backend)
        self.search({"query": "q", "max_results": 3}, backend)
        self.search({"query": "q", "max_results": 500}, backend)
        self.search({"query": "q", "max_results": 0}, backend)
        self.assertEqual(seen, [websearch.DEFAULT_RESULTS, 3, websearch.MAX_RESULTS, 1])

    def test_no_results_says_so(self) -> None:
        self.assertEqual(self.search({"query": "nothing"}, lambda query, n: []),
                         "No results for 'nothing'.")

    def test_an_empty_query_never_reaches_the_backend(self) -> None:
        def backend(query: str, max_results: int) -> list[Result]:
            raise AssertionError("backend called")

        self.assertEqual(self.search({"query": "  "}, backend), "Error: query must not be empty")

    def test_a_backend_failure_is_reported_readably(self) -> None:
        def backend(query: str, max_results: int) -> list[Result]:
            raise WebSearchError("rate-limited")

        self.assertEqual(self.search({"query": "q"}, backend), "Error: web search failed: rate-limited")

    def test_a_search_that_outlasts_the_total_timeout_is_given_up_on(self) -> None:
        def backend(query: str, max_results: int) -> list[Result]:
            time.sleep(0.2)
            return _results(1)

        started = time.monotonic()
        with patch.object(websearch, "_TOTAL_TIMEOUT", 0.01):
            result = self.search({"query": "q"}, backend)
        self.assertIn("timed out after 0.01 seconds", result)
        # The abandoned thread must not hold up the loop's shutdown either.
        self.assertLess(time.monotonic() - started, 0.15)

    def test_a_timeout_raised_by_the_backend_is_not_reported_as_the_total_one(self) -> None:
        def backend(query: str, max_results: int) -> list[Result]:
            raise TimeoutError("socket")

        with self.assertRaises(TimeoutError):
            self.search({"query": "q"}, backend)


class DdgsBackendTest(unittest.TestCase):
    """The ddgs adapter, against a stand-in for the library: no network."""

    def setUp(self) -> None:
        class DDGSException(Exception):
            pass

        class RatelimitException(DDGSException):
            pass

        class TimeoutException(DDGSException):
            pass

        self.calls: list = []
        self.outcome: object = []
        test = self

        class DDGS:
            def __init__(self, **kwargs) -> None:
                test.calls.append(("init", kwargs))

            def text(self, query: str, **kwargs) -> list[dict]:
                test.calls.append(("text", query, kwargs))
                if isinstance(test.outcome, Exception):
                    raise test.outcome
                return test.outcome

        exceptions = types.ModuleType("ddgs.exceptions")
        exceptions.DDGSException = DDGSException
        exceptions.RatelimitException = RatelimitException
        exceptions.TimeoutException = TimeoutException
        module = types.ModuleType("ddgs")
        module.DDGS = DDGS
        module.exceptions = exceptions
        self.exceptions = exceptions
        patcher = patch.dict(sys.modules, {"ddgs": module, "ddgs.exceptions": exceptions})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rows_are_mapped_and_the_auto_backend_is_asked_for(self) -> None:
        self.outcome = [{"title": "T", "href": "https://t", "body": "B"}, {"href": "https://u"}]
        self.assertEqual(websearch._ddgs("query", 5),
                         [Result("T", "https://t", "B"), Result("", "https://u", "")])
        self.assertEqual(self.calls, [
            ("init", {"timeout": websearch._REQUEST_TIMEOUT}),
            ("text", "query", {"max_results": 5, "backend": "auto"}),
        ])

    def test_more_rows_than_asked_for_are_cut(self) -> None:
        self.outcome = [{"title": str(i), "href": f"https://{i}", "body": ""} for i in range(8)]
        self.assertEqual(len(websearch._ddgs("query", 3)), 3)

    def test_the_no_results_error_is_an_empty_list(self) -> None:
        self.outcome = self.exceptions.DDGSException("No results found.")
        self.assertEqual(websearch._ddgs("query", 5), [])

    def test_rate_limits_timeouts_and_other_errors_are_named(self) -> None:
        for outcome, expected in (
            (self.exceptions.RatelimitException("202"), "rate-limited"),
            (self.exceptions.TimeoutException("slow"), "did not answer in time"),
            (self.exceptions.DDGSException("boom"), "boom"),
        ):
            self.outcome = outcome
            with self.assertRaises(WebSearchError) as caught:
                websearch._ddgs("query", 5)
            self.assertIn(expected, str(caught.exception))


class ToolTest(unittest.TestCase):
    def test_the_tool_is_never_gated(self) -> None:
        for mode in tools.MODES:
            self.assertEqual(tools.gate("web_search", {"query": "q"}, mode, Path.cwd()), "allow")

    def test_the_registry_entry_runs_the_search(self) -> None:
        with patch.object(websearch, "_ddgs", lambda query, n: _results(1)):
            result = asyncio.run(tools.execute_tool("web_search", {"query": "q"}, Path.cwd()))
        self.assertEqual(result, "1. Title 1\n   https://example.com/1\n   snippet 1")

    def test_the_query_is_the_call_summary(self) -> None:
        self.assertEqual(tools.summarize_call("web_search", {"query": "textual 8 release notes"}),
                         "textual 8 release notes")

    def test_subagents_and_headless_keep_it(self) -> None:
        self.assertNotIn("web_search", tools.SUBAGENT_DENIED)
        self.assertNotIn("web_search", (*tools.BACKGROUND_TOOLS, *tools.INTERACTIVE_TOOLS))


class DisabledTest(unittest.TestCase):
    def test_no_web_search_takes_the_tool_out_of_every_agents_toolset(self) -> None:
        config = Config(model="test:stub")
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            with Agent.open(cwd=cwd, config=config) as agent:
                self.assertIn("web_search", agent.toolset)
            config.web_search = False
            with Agent.open(cwd=cwd, config=config) as agent:
                self.assertNotIn("web_search", agent.toolset)
                self.assertNotIn("web_search", [s["function"]["name"] for s in agent.tool_schemas])


if __name__ == "__main__":
    unittest.main()
