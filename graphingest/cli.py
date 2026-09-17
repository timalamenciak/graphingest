"""Small helpers shared by the command-line entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console defaults to cp1252, and CAMO's own descriptions contain
    en dashes and arrows. Without this, printing a rendered prompt raises
    UnicodeEncodeError on exactly the machines the hackathon runs on.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Log to the console, and to a file as well when one is given."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
