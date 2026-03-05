from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class JsonlLogger:
    """A tiny JSONL logger (no external deps)."""

    out_path: Path

    def __post_init__(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        # Also echo a concise line
        msg = {k: payload[k] for k in payload.keys() if k != "ts"}
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()
