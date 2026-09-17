from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def api_key() -> str:
    load_local_env()
    value = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    if not value:
        raise RuntimeError("Set GROQ_API_KEY in the environment or gitignored .env file")
    return value
