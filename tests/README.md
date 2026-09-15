# Tests

Run from the repository root with `uv run pytest`; CI also runs `uv run ruff check .`.
Use pytest even for `unittest.TestCase` tests: `conftest.py` provides per-test
config/data isolation, disables telemetry and local skill/agent discovery, and
blocks real model requests. Tests opt into the behavior they exercise with a
local patch. `FunctionModel` stubs still exercise the real agent loop.

## Where tests belong

| Area | Files |
| --- | --- |
| Agent construction, prompts, session lifetime | `test_agent.py` |
| Agent tool execution and supervision wiring | `test_agent_tools.py` |
| Streaming, queued input, replay and persisted outcomes | `test_agent_events.py` |
| Compaction integration / compaction policy and helpers | `test_agent_compaction.py` / `test_compaction.py` |
| TUI panes, tabs and focus | `test_app.py` |
| TUI input, approvals, queuing and interruption | `test_app_input.py` |
| TUI resume, fork, handoff and profiles | `test_app_sessions.py` |
| TUI rendering, background jobs and recaps | `test_app_rendering.py`, `test_app_jobs.py`, `test_app_recap.py` |
| Tool dispatch, file edits, validation and history | `test_tools.py` |
| Permissions, shell processes/output and grep | `test_tools_permissions.py`, `test_tools_shell.py`, `test_tools_grep.py` |
| Job state and supervisor routing/ownership | `test_jobs.py`, `test_supervisor.py` |
| CLI arguments and end-to-end headless runs / renderer protocol | `test_cli.py` / `test_headless.py` |

Other modules follow the production module name. For example:

```sh
uv run pytest tests/test_agent_compaction.py
uv run pytest tests/test_app_input.py -k interrupt
```

## Test support and scope

- Reusable doubles and interactions live in `tests/support/`. Import them by
  package name; never import a collected test module or borrow a test class's
  helper. Keep helpers used by only one class next to that class.
- Prefer observable results: emitted events, persisted records, rendered output,
  permissions and process cleanup. A check that a value merely has the right
  type is rarely enough.
- Keep policy edge cases in the lower-level tests. Integration tests should
  prove wiring, ownership, persistence or user interaction. Similar scenarios
  at two layers are useful when they protect different failures.
- Control random inputs when testing randomized policy. TUI state waits use the
  shared bounded `_wait_for`; tests that assert sustained absence still need an
  observation interval. `settle` is only for the in-memory job doubles, not real
  processes or UI timers.
