# Paimon

A minimal terminal code agent built on **litellm** (LLM access) and **textual** (TUI).

## Features (MVP)

- Streaming agent loop: LLM + tool calls until the task is done
- Four tools: `read_file`, `write_file`, `edit_file`, `bash`
- Live streaming output + reasoning display in a Textual UI
- Confirmation prompt before dangerous actions (`bash`, `write_file`, `edit_file`)

## Setup

Set these environment variables (uses an OpenAI-compatible endpoint via the
litellm `openai/` prefix):

| Variable | Description | Example |
|----------|-------------|---------|
| `PAIMON_API_KEY`  | API key for the endpoint | `tp-...` |
| `PAIMON_MODEL`    | litellm model id | `openai/mimo-v2.5-pro` |
| `PAIMON_API_BASE` | base URL of the endpoint | `https://token-plan-cn.xiaomimimo.com/v1` |

## Run

```bash
uv run paimon        # or: uv run main.py
```

## Layout

| File | Role |
|------|------|
| `paimon/config.py` | model config |
| `paimon/tools.py`  | tool schemas + execution |
| `paimon/agent.py`  | UI-agnostic agent loop (yields typed events) |
| `paimon/app.py`    | Textual TUI |
