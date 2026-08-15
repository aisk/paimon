---
name: paimon
description: >
  Delegate a coding task to Paimon, a terminal code agent, as a one-shot
  subprocess. Use when the user asks to run something through paimon, wants a
  task done by a different model or account, or wants to continue an earlier
  paimon session. Covers preflight, invocation, output parsing, resume and
  inspecting a run's event log.
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
  --output-format result --mode edit --max-tool-calls 50 --timeout 600
```

- stdout is exactly one JSON object: `{"type": "result", ...}` with `text`
  (the final answer), `session_id`, `subtype`, `denied` and `log_end` (a
  cursor into the session log, see below). Ignore unknown fields. Do not use
  `--output-format json` unless you need to consume events in real time — it
  streams every intermediate event and repeats the answer, which you would
  pay to read twice.
- Pick the mode for the task: `read` (analysis plus clearly read-only
  commands like `ls`/`git status`, including `cd dir && …` chains and pipes
  of recognized commands), `edit` (may edit inside the
  working directory), `yolo` (may also run commands; the default). `-p` never
  asks for confirmation: in read/edit anything the mode disallows is refused
  and counted in `denied`; a non-zero count usually means rerun with a more
  permissive mode. `--strict` restores ask-before-every-command, so under `-p`
  even read-only commands are refused.
- `--model provider:name` overrides the model for this run;
  `--profile NAME` switches to a separately configured account.
- `--append-system-prompt "You are a code reviewer. Only report findings."`
  adds a role definition on top of the base system prompt. New sessions only —
  it is persisted with the session (resuming keeps the role), so combining it
  with `-c`/`-r` is a usage error.
- Pipe data in on stdin (`git diff | paimon -p "review this"`) and reference
  files as `@path/to/file` inside the prompt.

## Resume

Always resume by id, using the `session_id` from the previous result:

```bash
paimon -r <session_id> -p "now also update the docs"   # id prefix is enough
paimon sessions --json                                 # enumerate resumable sessions here
```

Do not use `-c` (resume latest): it picks the newest session in the
directory, so any concurrent paimon run — yours or another agent's — can
slip in between and receive your follow-up.

## Inspecting what a run did

The result line is normally all you need. When it is not — `denied` is
non-zero, the run failed or timed out, or the answer looks off — read the
persisted event log instead of rerunning with a streaming format:

```bash
paimon log <session_id> --turns 1             # everything the last turn did
paimon log <session_id> --after <log_end>     # only what a later run added
paimon log <session_id> --tail 20             # the last 20 records
```

One clipped line per event, each prefixed `[seq]` (tool calls, tool results
with sizes, assistant text). `log_end` from a result is the seq cursor: pass
the previous run's `log_end` to `--after` to see exactly the new records.
Add `--json` for raw records or `--full` to un-clip content — both are
verbose, so filter first.

## Exit codes

`0` done · `1` error · `2` bad usage · `4` hit `--max-tool-calls` ·
`124` hit `--timeout` · `130` interrupted. On 4 and 124 the partial turn is
persisted — resume the session to let it finish.
