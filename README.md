# Paimon

A minimal terminal code agent built on **litellm** (LLM access) and **textual** (TUI).

## Features

- Streaming agent loop: LLM + tool calls until the task is done
- Six tools: `read_file`, `write_file`, `edit_file`, `glob`, `bash`, `write_todos`
- Live streaming output + reasoning display in a Textual UI
- Automatic context compaction that summarizes old history and keeps recent messages verbatim
- Confirmation prompt before dangerous actions (`bash`, `write_file`, `edit_file`)
- Guided login (provider/model picker sourced from litellm's catalog), reachable from the command palette (`Ctrl+P`)

## Setup

On first launch (no config yet) Paimon walks you through provider → model → API base → API key. The choices and selected theme are stored in `~/.config/paimon/config.json` (override the directory with `PAIMON_CONFIG_HOME`). Re-run the flow anytime via the command palette (`Ctrl+P` → "Login / switch provider").

## Run

```bash
uv run paimon        # or: uv run main.py
uv run paimon -c     # continue the most recent session for this directory
uv run paimon --yolo # run commands and file changes without confirmation
```

Sessions are appended as JSONL under `~/.local/share/paimon/sessions/` (or
`$XDG_DATA_HOME/paimon/sessions/`) and separated by working directory. Override
the data directory with `PAIMON_DATA_HOME`. Use “New session” from the command
palette to leave the current history and start fresh.

When a model's known context window has fewer than 16,384 tokens left, Paimon
automatically creates a checkpoint summary and keeps roughly the most recent
20,000 tokens unchanged. The full pre-compaction history remains in the JSONL
session file. Custom models that are not in LiteLLM's catalog can configure the
window and thresholds in `config.json`:

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

## Layout

| File | Role |
|------|------|
| `paimon/config.py` | load/save `config.json` |
| `paimon/compaction.py` | context sizing, cut selection, and checkpoint summaries |
| `paimon/login.py`  | login flow screens |
| `paimon/tools.py`  | tool schemas + execution |
| `paimon/agent.py`  | UI-agnostic agent loop (yields typed events) |
| `paimon/ui.py`     | reusable UI components |
| `paimon/app.py`    | Textual TUI |
