from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


T = TypeVar("T")


def dataclass_from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            kwargs[f.name] = d[f.name]
    return cls(**kwargs)  # type: ignore[arg-type]
