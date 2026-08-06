"""Structured logging via structlog: colourised text locally, JSON when piped."""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return

    from app.config import settings

    lvl = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=lvl)

    # Pretty output for a TTY, JSON when redirected to a file or collector.
    renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if sys.stdout.isatty()
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(lvl),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = __name__):
    configure_logging()
    return structlog.get_logger(name)
