"""Config loading with shell-style environment interpolation.

Config files use ``${VAR}`` and ``${VAR:-default}`` so that secrets stay in the
environment and never get committed. Interpolation runs over every string in
the loaded structure, recursively.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

#: ``${NAME}`` or ``${NAME:-fallback}``. The fallback may be empty.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_LLM_CONFIG = CONFIG_DIR / "pipeline.yaml"

#: Same file: the LLM block and the pipeline defaults live together.
DEFAULT_PIPELINE_CONFIG = DEFAULT_LLM_CONFIG


def interpolate(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""),
            value,
        )
    if isinstance(value, dict):
        return {key: interpolate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate(item) for item in value]
    return value


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file and interpolate environment references."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {resolved}")
    return interpolate(data)
