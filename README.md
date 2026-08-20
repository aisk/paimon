# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

English | [简体中文](README.zh-CN.md)

Paimon is a coding agent that lives in your terminal. It reads and edits files in the current directory and runs commands. It also runs headless and imports as a library, so a stronger agent or a program of your own can drive it.

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

The first launch asks for a provider, model, API base and key. Then just type what you want done. Write `@path/to/file` in a prompt to hand a file to the agent.

While it runs: `Shift+Tab` switches how much the agent may do on its own (**read** asks before writing files or running commands, **edit** lets edits inside the working directory through, **yolo** never asks and is the default), `Esc` interrupts the current turn, `Ctrl+P` opens the command palette, `Ctrl+C` quits.

`Ctrl+T` opens another session in a pane of its own, `Ctrl+W` closes one, `Ctrl+PageUp` and `Ctrl+PageDown` move between them, and `Ctrl+G` jumps to a pane waiting for permission. Paimon can open panes itself: ask for two independent things and it starts a second agent in its own tab. It can also leave a command running in a tab of its own, a dev server or a watcher, instead of holding up a turn.

## Skills

Paimon loads [Agent Skills](https://agentskills.io) from `~/.config/paimon/skills`, `~/.agents/skills` and every `.agents/skills` from the working directory up to the repository root. Only each skill's name and description go into the system prompt; the model reads the `SKILL.md` when a task matches, and `/skill:name args` sends it explicitly (the `/` command palette lists them). More locations go in `config.json` as `"skills": ["~/.claude/skills"]` or on the command line with `--skill PATH`; `--no-skills` skips the default locations. When two skills share a name, explicit paths beat the project's, which beat the global ones.

## Using Paimon as a subagent

Frontier models are good at planning and reviewing; the steps in between are often mechanical. Point Paimon at a cheaper model and let Claude Code or Codex write the plan and check the result. A profile keeps that model's account separate:

```bash
paimon login --profile glm --model zai:glm-4.7 --api-key-env ZAI_API_KEY
paimon --profile glm -p "apply the plan in PLAN.md" --mode edit --output-format result
```

The bundled skill teaches the calling agent this workflow:

```bash
paimon install-skill                  # into Claude Code (~/.claude/skills/paimon)
paimon install-skill --target codex   # into Codex; --dest DIR for anywhere else
npx skills add aisk/paimon            # the same skill, via skills.sh
```

## Using Paimon as a library

The agent loop is importable, so a Python program can drive it without going through the CLI. `Agent.open()` starts or resumes a session and `agent.run()` yields typed events, one per text chunk, tool call and turn end, which the caller renders or filters however it likes:

```python
import asyncio

from paimon.agent import Agent, TextDelta

async def main():
    agent = Agent.open(mode="edit")
    async for event in agent.run("summarize the tests in this directory"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)

asyncio.run(main())
```

`Agent.open()` also takes a working directory, an async `confirm` callback for permission prompts, and a `toolset` to hand the model fewer tools or tools of your own. An agent holds its session until it goes away; to give it back at a definite moment, call `close()` or use the agent as a context manager. It writes the same session files as the CLI, so a run started in code can be resumed later with `paimon -r`.

## Sessions

Every conversation is saved, and long ones are summarized in place near the context limit. Paimon prints the command that brings a session back when you leave:

```bash
paimon -r            # choose a session started in this directory
paimon -r a1b2c3     # resume one by id
paimon -c            # resume the most recent one
paimon sessions      # list them (--json for machines)
paimon log a1b2c3    # what a session did, one line per event
```

## Other ways to run it

```bash
paimon --mode read                  # start in a more cautious permission mode (yolo is the default)
paimon --strict                     # ask before every command, even read-only ones
paimon --web                        # the same UI in a browser (--port, default 8000)
paimon -p "what does cli.py do?"    # one answer on stdout, no UI
cat log.txt | paimon -p "summarize this"
paimon --model zai:glm-4.7          # this model for this run only
paimon --profile work               # a separately configured account
```

`-p` never stops to ask, so with the default `yolo` mode it can already write files and run commands. Add `--output-format result` for a single JSON object with the outcome, which is what a calling program should read. `paimon --help` lists the rest.

## Configuration

Each profile keeps its model settings in `~/.config/paimon/<name>/config.json`, written by the first launch or by `paimon login`. Sessions live in `~/.local/share/paimon/sessions/`. File changes render nicer if [delta](https://github.com/dandavison/delta) is installed.

Read and edit modes run a small set of clearly read-only commands (`ls`, `cat`, `git status`, …) without asking; `--strict` turns that off. **This is a guardrail against agent mistakes, not a security boundary.** For real isolation, run Paimon inside a container or VM.

## Telemetry

Each launch sends one anonymous event to Google Analytics: a random install id, the launch mode, the version, the OS name and the configured provider and model name. Nothing from your sessions, prompts, files or credentials is included. Set `PAIMON_NO_TELEMETRY=1` or `DO_NOT_TRACK=1` to turn it off.
