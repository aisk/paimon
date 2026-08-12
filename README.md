# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

English | [简体中文](README.zh-CN.md)

Paimon is a coding agent that lives in your terminal. It reads and edits files in the current directory and runs commands, asking before it touches anything. It also runs headless, so a stronger agent can drive it as a worker.

## Install

```bash
uv tool install paimon   # or: pip install paimon
```

## Getting started

```bash
paimon
```

Or run it without installing anything:

```bash
uvx paimon
```

The first launch asks for a provider, model, API base and key, and saves them to `~/.config/paimon/default/config.json`. Then just type what you want done.

While it runs: `Shift+Tab` switches how much the agent may do on its own (**read**: ask before writing files or running commands, except clearly read-only ones like `ls` or `git status`, which run without asking, **edit**: edits inside the working directory go through, **yolo**: never ask), `Esc` interrupts the current turn, `Ctrl+P` opens the command palette (switch provider or profile, new, fork or resume session, show the model's thinking, compact the context), `Ctrl+C` quits.

`Ctrl+T` opens another session in a pane of its own, `Ctrl+W` closes one, `Ctrl+PageUp` and `Ctrl+PageDown` move between them, and `Ctrl+G` jumps to a pane waiting for permission. The command palette docks the tabs at the top, the left or the right.

Write `@path/to/file` in a prompt to hand a file to the agent.

## Using Paimon as a subagent

Frontier models are good at planning and reviewing; the steps in between are often mechanical. Point Paimon at a cheaper model, let Claude Code or Codex write the plan and check the result, and pay frontier prices only for the parts that need them. A profile keeps that model's account separate:

```bash
paimon login --profile glm --model zai:glm-4.7 --api-key-env ZAI_API_KEY
paimon --profile glm -p "apply the plan in PLAN.md" --mode edit --output-format result
```

The bundled skill teaches the calling agent this workflow (check `paimon status --json`, run one-shot, read the single result object, resume its `session_id`, inspect what a run did with `paimon log`):

```bash
paimon install-skill                  # into Claude Code (~/.claude/skills/paimon)
paimon install-skill --target codex   # into Codex; --dest DIR for anywhere else
npx skills add aisk/paimon            # the same skill, via skills.sh
```

## Sessions

Every conversation is saved. Paimon prints the command that brings one back when you leave:

```bash
paimon -r            # choose a session started in this directory
paimon -r a1b2c3     # resume one by id
paimon -c            # resume the most recent one
paimon sessions      # list them (--json for machines)
paimon log a1b2c3    # what a session did, one line per event
```

`paimon log` prefixes every line with a stable seq number; `--after SEQ`, `--turns N` and `--tail N` narrow the window, `--json` and `--full` give the raw records.

## Other ways to run it

```bash
paimon --mode edit                  # start in a less cautious permission mode
paimon --strict                     # ask before every command, even read-only ones
paimon --web                        # the same UI in a browser (--port, default 8000)
paimon -p "what does cli.py do?"    # one answer on stdout, no UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # this model for this run only
paimon --profile work               # a separately configured account
```

`-p` never stops to ask, so anything the current mode would prompt for is refused instead (recognized read-only commands still run in read mode); pass `--mode edit` or `--mode yolo` if the run needs to change files. Add `--output-format result` for a single JSON object with the outcome (or `json` for one event per line), and `--timeout`/`--max-tool-calls` to bound an unattended run.

## Configuration

`~/.config/paimon/<name>/config.json` holds each profile's model settings (`default` unless `--profile` says otherwise). Two optional keys change how it behaves: auto-allowing read-only commands, and summarizing long conversations in place near the context limit.

```json
{
  "safe_commands": false,
  "compaction": {
    "enabled": true,
    "context_window": 128000,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000
  }
}
```

`safe_commands` (default `true`) lets read and edit modes run a small fixed set of clearly read-only commands (`ls`, `cat`, `git status`, …) without asking; `--strict` turns it off for one run. Recognized commands may be chained with `&&`, `;` or pipes, including `cd dir && …` when the directory stays inside the working directory and every link in the chain is `&&`. Redirects, `$()`/backtick substitution and background `&` still ask.

**This is a guardrail against agent mistakes, not a security boundary.** Recognized commands are still resolved through `PATH` and can still follow symlinks out of the working directory, and even a pure read pulls file contents into the model's context, so read-only is not confidentiality-safe. For real isolation, run Paimon inside a container or VM.

Sessions live in `~/.local/share/paimon/sessions/` (`PAIMON_DATA_HOME` overrides). File changes render nicer if [delta](https://github.com/dandavison/delta) is installed.
