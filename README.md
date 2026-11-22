# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

English | [简体中文](README.zh-CN.md)

Paimon is a coding agent that lives in your terminal. It reads and edits files in the current directory and runs commands. It also runs headless and imports as a library, so a stronger agent or a program of your own can drive it.

## Install

```bash
uv tool install paimon   # or: pip install paimon
```

Or run it without installing anything:

```bash
uvx paimon
```

## Getting started

```bash
paimon
```

The first launch asks for a provider, its API base and key, then a model. The first provider on offer, Kilo, needs no account: its free models run without a key, but the providers behind them may keep and train on your prompts and code, and use is limited to 200 requests an hour per IP address.

Then just type what you want done. Write `@path/to/file` in a prompt to hand a file or an image to the agent. A line starting with `!` runs in a shell instead of being sent.

While it runs: `Shift+Tab` switches how much the agent may do on its own (**read** only reads, **auto** edits inside the working directory and has a second model call approve everything else, **yolo** checks nothing and is the default), `Esc` interrupts the current turn, `Ctrl+P` opens the command palette, `Ctrl+T` opens another session in a pane of its own, `Ctrl+C` quits.

## Common commands

```bash
paimon -c                           # resume the most recent session here
paimon -r                           # choose a session to resume
paimon --mode read                  # start in a more cautious permission mode
paimon --model zai:glm-4.7          # this model for this run only
paimon --web                        # the same UI in a browser
paimon -p "what does cli.py do?"    # one answer on stdout, no UI
cat log.txt | paimon -p "summarize this"
paimon login zai --api-key-env ZAI_API_KEY
paimon install-skill                # teach Claude Code or Codex to use Paimon as a worker
```

`-p` never stops to ask, so with the default `yolo` mode it can already write files and run commands. The permission modes are a guardrail against agent mistakes, not a security boundary: for real isolation, run Paimon inside a container or VM.

## Documentation

- [Guide](https://aisk.github.io/paimon/guide.html): prompts and images, permission modes, panes and parallel agents, sessions, skills, and running Paimon from another agent or from Python.
- [Reference](https://aisk.github.io/paimon/reference.html): every switch, command, key, tool and `config.json` setting, and the headless output format.
- [Architecture](https://aisk.github.io/paimon/architecture.html): how the code is put together.

## Telemetry

Each launch sends one anonymous event to Google Analytics, with nothing from your sessions, prompts, files or credentials in it; the [reference](https://aisk.github.io/paimon/reference.html#telemetry) lists what it holds. Set `PAIMON_NO_TELEMETRY=1` or `DO_NOT_TRACK=1` to turn it off.
