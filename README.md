# Paimon

![Paimon](https://automaton-media.com/wp-content/uploads/2020/10/20201019-140524-header.jpg)

A minimal terminal code agent built on pydantic-ai and textual.

## Run

```bash
uv run paimon              # or: python -m paimon
uv run paimon -r [ID]      # pick a session to resume, or resume one by id prefix
uv run paimon --mode yolo  # permission mode: read (default), edit or yolo
uv run paimon --web        # serve the UI in a browser (--port, default 8000)
uv run paimon -p "..."     # one turn, no UI (--output-format json for JSONL)
cat log.txt | uv run paimon -p "summarize this"
```

First launch asks for a provider, model, API base and key, saved to
`~/.config/paimon/config.json`. Change them later with "Login / switch
provider" in the command palette (Ctrl+P).

## Notes

- `@path` in a prompt attaches that file: small files inline, larger ones are
  referenced by path for the model to read on demand.
- Shift+Tab cycles the permission mode: read (writes, commands and access
  outside the working directory ask for confirmation), edit (edits inside the
  working directory run without asking) and yolo (nothing asks). File changes
  are shown as a side-by-side diff (nicer if
  [delta](https://github.com/dandavison/delta) is installed).
- Prompts typed while the agent is busy are queued and sent when it finishes.
- `-p` never asks for confirmation: whatever the permission mode would prompt
  for is denied instead, so pass `--mode edit` or `--mode yolo` when the run
  needs to write files or run commands. The answer goes to stdout and
  everything else to stderr.
- Sessions are JSONL files under `~/.local/share/paimon/sessions/`, split by
  working directory (`PAIMON_DATA_HOME` overrides). A session can only be
  active in one process at a time.
- Near the context limit old history is summarized in place, once a window is
  set in `config.json`:

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
