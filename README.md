# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

Paimon is a coding agent that lives in your terminal. It reads and edits files in the current directory and runs commands, asking before it touches anything.

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

The first launch asks for a provider, model, API base and key, and saves them to `~/.config/paimon/config.json`. Then just type what you want done.

While it runs: `Shift+Tab` switches how much the agent may do on its own (**read**: ask before writing files or running commands, **edit**: edits inside the working directory go through, **yolo**: never ask), `Esc` interrupts the current turn, `Ctrl+P` opens the command palette (switch provider, new or resume session, show the model's thinking, compact the context), `Ctrl+C` quits.

Write `@path/to/file` in a prompt to hand a file to the agent.

## Sessions

Every conversation is saved. Paimon prints the command that brings one back when you leave:

```bash
paimon -r            # choose a session started in this directory
paimon -r a1b2c3     # resume one by id
paimon -c            # resume the most recent one
paimon sessions      # list them (--json for machines)
```

## Other ways to run it

```bash
paimon --mode edit                  # start in a less cautious permission mode
paimon --web                        # the same UI in a browser (--port, default 8000)
paimon -p "what does cli.py do?"    # one answer on stdout, no UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # this model for this run only
paimon --profile work               # a separately configured account
```

`-p` never stops to ask, so anything the current mode would prompt for is refused instead; pass `--mode edit` or `--mode yolo` if the run needs to change files. Add `--output-format json` for one JSON event per line, and `--timeout`/`--max-tool-calls` to bound an unattended run.

## Calling Paimon from another agent

Everything above works without a terminal, so a code agent (Claude Code, Codex, a script) can drive Paimon as a subprocess: check `paimon status --json`, log in with `paimon login --model provider:name --api-key-env VAR` if needed, run `paimon -p ... --output-format json`, and resume the session id from the final `result` line later. `paimon install-skill` (`--target claude|codex`, or `--dest DIR`) installs a skill file that teaches the calling agent this workflow.

## Configuration

`~/.config/paimon/config.json` holds the model settings. Long conversations can be summarized in place near the context limit by adding:

```json
{
  "compaction": {
    "enabled": true,
    "context_window": 128000,
    "reserve_tokens": 16384,
    "keep_recent_tokens": 20000
  }
}
```

Sessions live in `~/.local/share/paimon/sessions/` (`PAIMON_DATA_HOME` overrides). File changes render nicer if [delta](https://github.com/dandavison/delta) is installed.
