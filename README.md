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

While it runs: `Shift+Tab` switches how much the agent may do on its own (**read**: ask before writing files or running commands, **edit**: edits inside the working directory go through, **yolo**: never ask), `Esc` interrupts the current turn, `Ctrl+P` opens the command palette (switch provider or profile, new or resume session, show the model's thinking, compact the context), `Ctrl+C` quits.

Write `@path/to/file` in a prompt to hand a file to the agent.

## Using Paimon as a subagent

Frontier models are good at planning and reviewing; the steps in between are often mechanical. Point Paimon at a cheaper model, let Claude Code or Codex write the plan and check the result, and pay frontier prices only for the parts that need them. A profile keeps that model's account separate:

```bash
paimon login --profile glm --model zai:glm-4.7 --api-key-env ZAI_API_KEY
paimon --profile glm -p "apply the plan in PLAN.md" --mode edit --output-format json
```

The bundled skill teaches the calling agent this workflow (check `paimon status --json`, run one-shot, parse the JSON output, resume the session id from the final `result` line):

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

## Configuration

`~/.config/paimon/<name>/config.json` holds each profile's model settings (`default` unless `--profile` says otherwise). Long conversations can be summarized in place near the context limit by adding:

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
