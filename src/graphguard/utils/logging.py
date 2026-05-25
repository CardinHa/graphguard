"""Rich-based logging utilities for GraphGuard."""

from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "magenta",
})

console = Console(theme=_THEME)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger that writes to the rich console."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
