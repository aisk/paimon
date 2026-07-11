# Paimon

A minimal terminal code agent built on **litellm** (LLM access) and **textual** (TUI).

## Features

- Streaming agent loop: LLM + tool calls until the task is done
- Six tools: `read_file`, `write_file`, `edit_file`, `glob`, `bash`, `write_todos`
- Live streaming output + reasoning display in a Textual UI
- Confirmation prompt before dangerous actions (`bash`, `write_file`, `edit_file`)
- Guided login (provider/model picker sourced from litellm's catalog), reachable from the command palette (`Ctrl+P`)

## Setup

On first launch (no config yet) Paimon walks you through provider → model → API base → API key. The choices and selected theme are stored in `~/.config/paimon/config.json` (override the directory with `PAIMON_CONFIG_HOME`). Re-run the flow anytime via the command palette (`Ctrl+P` → "Login / switch provider").

## Run

```bash
uv run paimon        # or: uv run main.py
uv run paimon -c     # continue the most recent session for this directory
```

Sessions are appended as JSONL under `~/.local/share/paimon/sessions/` (or
`$XDG_DATA_HOME/paimon/sessions/`) and separated by working directory. Override
the data directory with `PAIMON_DATA_HOME`. Use “New session” from the command
palette to leave the current history and start fresh.

## Layout

| File | Role |
|------|------|
| `paimon/config.py` | load/save `config.json` |
| `paimon/login.py`  | login flow screens |
| `paimon/tools.py`  | tool schemas + execution |
| `paimon/agent.py`  | UI-agnostic agent loop (yields typed events) |
| `paimon/app.py`    | Textual TUI |
