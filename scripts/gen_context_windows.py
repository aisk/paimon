#!/usr/bin/env python3
"""Regenerate paimon/model_windows.py from the models.dev catalogue.

Development-only: it is committed to the repo but deliberately left out of the
wheel (pyproject lists `packages = ["paimon"]`), so the shipped package carries
the generated table and no network dependency.

    python scripts/gen_context_windows.py

The catalogue lists a model once per provider that serves it, and those copies
disagree: a model's own vendor reports the real window while some resellers
report a plan-limited one. We key the table on the bare model name and settle
disagreements by majority, which tracks the vendor's number and drops outliers
(claude-haiku-4-5 appears as 200k eleven times and 20k once).
"""

import collections
import json
import sys
import urllib.request
from pathlib import Path

SOURCE = "https://models.dev/api.json"
# models.dev rejects the default urllib agent.
USER_AGENT = "paimon-context-window-generator"
TARGET = Path(__file__).resolve().parent.parent / "paimon" / "model_windows.py"

HEADER = '''"""Context-window sizes by model name, generated from https://models.dev.

Do not edit by hand: run `python scripts/gen_context_windows.py` instead.

Names are the catalogue's own model ids, lowercased, on the assumption that
the same model has the same window wherever it is served. Where providers
disagreed about a model, the majority value won.
"""

# Generated from {source}, {count} models.
CONTEXT_WINDOWS: dict[str, int] = {{
'''


def collect(catalogue: dict) -> dict[str, int]:
    """Model name -> the window most providers agree on, ties going to the larger."""
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for provider in catalogue.values():
        for name, model in (provider.get("models") or {}).items():
            window = (model.get("limit") or {}).get("context")
            if isinstance(window, int) and window > 0:
                votes[name.strip().lower()][window] += 1
    return {
        name: max(counts.items(), key=lambda pair: (pair[1], pair[0]))[0]
        for name, counts in sorted(votes.items())
    }


def render(windows: dict[str, int]) -> str:
    body = "".join(f"    {name!r}: {window},\n" for name, window in windows.items())
    return HEADER.format(source=SOURCE, count=len(windows)) + body + "}\n"


def main() -> int:
    request = urllib.request.Request(SOURCE, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        catalogue = json.load(response)
    windows = collect(catalogue)
    if len(windows) < 1_000:
        print(f"refusing to write a suspiciously small table ({len(windows)} models)",
              file=sys.stderr)
        return 1
    TARGET.write_text(render(windows), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(Path.cwd())}: {len(windows)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
