---
name: paimon
description: >
  Delegate a coding task to Paimon, a terminal code agent, as a one-shot
  subprocess. Use when the user asks to run something through paimon, wants a
  task done by a different model or account, or wants to continue an earlier
  paimon session. Covers preflight, invocation, output parsing and resume.
---

# Driving Paimon

Paimon is a coding agent CLI. `paimon -p "task"` runs one turn without a UI
and exits; the conversation is persisted and can be resumed later.

## Preflight

```bash
paimon status --json
```

Exit 0 means ready; 1 means not logged in — report that to the user instead of
guessing credentials. If the user provided them:
`paimon login --model provider:name --api-key-env SOME_VAR`.

## One-shot run

```bash
paimon -p "fix the failing test in tests/test_foo.py" \
  --output-format json --mode edit --max-tool-calls 50 --timeout 600
```

- stdout is one JSON object per line; the last line is always
  `{"type": "result", ...}` with `text` (the final answer), `session_id`,
  `subtype` and `denied`. Ignore unknown event types and fields.
- Pick the mode for the task: `read` (analysis only, the default), `edit`
  (may edit inside the working directory), `yolo` (may also run commands).
  `-p` never asks for confirmation — anything the mode disallows is refused
  and counted in `denied`; a non-zero count usually means rerun with a more
  permissive mode.
- `--model provider:name` overrides the model for this run;
  `--profile NAME` switches to a separately configured account.
- `--append-system-prompt "You are a code reviewer. Only report findings."`
  adds a role definition on top of the base system prompt. New sessions only —
  it is persisted with the session (resuming keeps the role), so combining it
  with `-c`/`-r` is a usage error.
- Pipe data in on stdin (`git diff | paimon -p "review this"`) and reference
  files as `@path/to/file` inside the prompt.

## Resume

```bash
paimon -c -p "now also update the docs"       # latest session in this directory
paimon -r <session_id> -p "..."               # a specific one (id prefix is enough)
paimon sessions --json                        # enumerate resumable sessions here
```

## Exit codes

`0` done · `1` error · `2` bad usage · `4` hit `--max-tool-calls` ·
`124` hit `--timeout` · `130` interrupted. On 4 and 124 the partial turn is
persisted — resume the session to let it finish.
